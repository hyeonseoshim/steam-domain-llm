from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_CANDIDATES = (
    "services/b-review-analysis/data/pseudo_labels/"
    "rare_topic_candidates.csv"
)

DEFAULT_GOLD = (
    "services/b-review-analysis/data/annotations/"
    "topic_golden_dataset.csv"
)

DEFAULT_OUTPUT = (
    "services/b-review-analysis/data/pseudo_labels/"
    "rare_positive_target_verification.csv"
)

TARGETS = {
    "pos__bugs": {
        "topic": "bugs",
        "definition": (
            "The review positively evaluates technical stability, "
            "the absence of bugs or crashes, or successful bug fixes. "
            "Examples include 'no bugs', 'never crashed', 'works without "
            "crashing', or 'the bugs have been fixed'."
        ),
        "reject_rule": (
            "Do not match when the review reports bugs or crashes, merely "
            "mentions a past buggy release without praising the current "
            "state, or discusses unrelated performance problems."
        ),
    },
    "pos__monetization": {
        "topic": "monetization",
        "definition": (
            "The review positively evaluates the game's monetization policy, "
            "such as fair DLC pricing, no microtransactions, no pay-to-win, "
            "reasonable premium pricing, free DLC, or willingly supporting "
            "the developer financially."
        ),
        "reject_rule": (
            "Do not match when DLC, premium currency, or purchases are merely "
            "mentioned without praise, or when pricing and monetization are "
            "criticized."
        ),
    },
}


def parse_topics(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {
        "", "nan", "none", "null", "[]", "-"
    }:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]
    except Exception:
        pass

    return [
        item.strip().strip('"').strip("'")
        for item in re.split(r"[,|;/]+", text)
        if item.strip()
    ]


def contains_target(value: Any, target: str) -> bool:
    if pd.isna(value):
        return False

    return target in {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def build_examples(
    gold: pd.DataFrame,
    topic: str,
    max_positive: int = 3,
    max_negative: int = 3,
) -> tuple[list[str], list[str]]:
    positive_rows = gold[
        gold["positive_topics"].apply(
            lambda value: topic in parse_topics(value)
        )
    ].head(max_positive)

    negative_rows = gold[
        gold["negative_topics"].apply(
            lambda value: topic in parse_topics(value)
        )
    ].head(max_negative)

    positive_examples = [
        str(text).strip()[:1500]
        for text in positive_rows["review_text_clean"]
        if str(text).strip()
    ]

    negative_examples = [
        str(text).strip()[:1500]
        for text in negative_rows["review_text_clean"]
        if str(text).strip()
    ]

    return positive_examples, negative_examples


def format_examples(
    heading: str,
    examples: list[str],
) -> str:
    if not examples:
        return f"{heading}\n- None available"

    lines = [heading]

    for index, example in enumerate(examples, start=1):
        lines.append(f"{index}. {example}")

    return "\n".join(lines)


def build_prompt(
    target_label: str,
    review_text: str,
    positive_examples: list[str],
    negative_examples: list[str],
) -> str:
    config = TARGETS[target_label]

    return f"""
You are verifying one specific signed topic label for a Steam review.

Target label: {target_label}

Definition:
{config["definition"]}

Important rejection rule:
{config["reject_rule"]}

{format_examples("Human-verified positive examples:", positive_examples)}

{format_examples("Human-verified opposite-polarity examples:", negative_examples)}

Review to verify:
{review_text}

Determine whether the review clearly contains the target label.

Return JSON only:
{{
  "target_label": "{target_label}",
  "is_match": true,
  "confidence": 0.0,
  "evidence": "short evidence from the review",
  "reason": "brief explanation"
}}

Rules:
- is_match must be true only when the target meaning and polarity are explicit.
- Do not infer positive sentiment merely from the presence of a keyword.
- confidence must be between 0 and 1.
- Use false when uncertain.
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            f"JSON 객체를 찾지 못했습니다: {cleaned[:300]}"
        )

    parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("응답이 JSON 객체가 아닙니다.")

    return parsed


def call_ollama(
    url: str,
    model: str,
    prompt: str,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 250,
            },
        },
        timeout=timeout,
    )

    response.raise_for_status()

    payload = response.json()
    raw_response = str(payload.get("response", "")).strip()

    return extract_json(raw_response), raw_response


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES,
    )
    parser.add_argument(
        "--gold",
        default=DEFAULT_GOLD,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    candidates = pd.read_csv(
        args.candidates,
        dtype={"recommendationid": str},
    )

    gold = pd.read_csv(
        args.gold,
        dtype={"recommendationid": str},
    )

    if "is_valid" in gold.columns:
        gold = gold[
            gold["is_valid"]
            .astype(str)
            .str.lower()
            .eq("true")
        ].copy()

    if "human_verified" in gold.columns:
        gold = gold[
            gold["human_verified"]
            .astype(str)
            .str.lower()
            .eq("true")
        ].copy()

    tasks = []

    for target_label in TARGETS:
        target_rows = candidates[
            candidates["target_labels"].apply(
                lambda value: contains_target(
                    value,
                    target_label,
                )
            )
        ]

        for _, row in target_rows.iterrows():
            tasks.append(
                {
                    "recommendationid": str(
                        row["recommendationid"]
                    ),
                    "appid": row.get("appid"),
                    "target_label": target_label,
                    "review_text_clean": str(
                        row["review_text_clean"]
                    ),
                    "voted_up": row.get("voted_up"),
                    "label": row.get("label"),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_rows = []

    if output_path.exists():
        existing = pd.read_csv(
            output_path,
            dtype={"recommendationid": str},
        )

        existing_rows = existing.to_dict("records")

    completed_keys = {
        (
            str(row.get("recommendationid")),
            str(row.get("target_label")),
        )
        for row in existing_rows
    }

    results = list(existing_rows)

    example_cache = {}

    for target_label, config in TARGETS.items():
        example_cache[target_label] = build_examples(
            gold,
            config["topic"],
        )

    total = len(tasks)

    for index, task in enumerate(tasks, start=1):
        key = (
            task["recommendationid"],
            task["target_label"],
        )

        if key in completed_keys:
            print(
                f"[{index}/{total}] 이미 완료: "
                f"{task['recommendationid']} "
                f"{task['target_label']}"
            )
            continue

        positive_examples, negative_examples = (
            example_cache[task["target_label"]]
        )

        prompt = build_prompt(
            target_label=task["target_label"],
            review_text=task["review_text_clean"],
            positive_examples=positive_examples,
            negative_examples=negative_examples,
        )

        result_row = dict(task)

        try:
            parsed, raw_response = call_ollama(
                url=args.ollama_url,
                model=args.model,
                prompt=prompt,
                timeout=args.timeout,
            )

            is_match = parsed.get("is_match", False)

            if isinstance(is_match, str):
                is_match = (
                    is_match.strip().lower() == "true"
                )

            try:
                confidence = float(
                    parsed.get("confidence", 0.0)
                )
            except Exception:
                confidence = 0.0

            confidence = max(
                0.0,
                min(1.0, confidence),
            )

            result_row.update(
                {
                    "is_match": bool(is_match),
                    "confidence": confidence,
                    "evidence": str(
                        parsed.get("evidence", "")
                    ).strip(),
                    "reason": str(
                        parsed.get("reason", "")
                    ).strip(),
                    "is_valid": True,
                    "error": "",
                    "raw_response": raw_response,
                    "teacher_model": args.model,
                    "verification_version": (
                        "rare_positive_binary_v1"
                    ),
                }
            )

        except Exception as error:
            result_row.update(
                {
                    "is_match": False,
                    "confidence": 0.0,
                    "evidence": "",
                    "reason": "",
                    "is_valid": False,
                    "error": str(error),
                    "raw_response": "",
                    "teacher_model": args.model,
                    "verification_version": (
                        "rare_positive_binary_v1"
                    ),
                }
            )

        results.append(result_row)

        pd.DataFrame(results).to_csv(
            output_path,
            index=False,
        )

        print(
            f"[{index}/{total}] "
            f"{task['target_label']} "
            f"match={result_row['is_match']} "
            f"confidence={result_row['confidence']}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    result_df = pd.DataFrame(results)

    print("\n정밀 검증 완료")
    print("전체 판정:", len(result_df))
    print("유효 출력:", int(
        result_df["is_valid"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    ))

    print("\n목표 라벨별 match 수:")
    print(
        result_df[
            result_df["is_match"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]
        ["target_label"]
        .value_counts()
        .sort_index()
    )

    print("\n저장:", output_path)


if __name__ == "__main__":
    main()
