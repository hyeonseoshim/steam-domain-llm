from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from generate_pseudo_labels_rag import (
    GoldenRetriever,
    build_examples_context,
    build_rules_context,
    build_topic_context,
    call_ollama,
    load_allowed_topics_from_golden,
    load_golden_examples,
    load_guidelines,
    parse_response,
    split_topics,
)


DEFAULT_GOLDEN_FILE = (
    "services/b-review-analysis/data/annotations/topic_golden_dataset.csv"
)
DEFAULT_GUIDELINES_FILE = (
    "services/b-review-analysis/data/annotations/topic_guidelines.json"
)
DEFAULT_OUTPUT_DIR = (
    "services/b-review-analysis/reports/prompt_strategy_eval"
)


def build_v1_prompt(review_text: str, allowed_topics: list[str]) -> str:
    allowed_list = ", ".join(allowed_topics)

    return f"""
You are a Steam game review topic classifier.

Classify the target review into positive_topics and negative_topics.

Allowed topics:
{allowed_list}

Strict rules:
- Use only the allowed topics.
- Do not create new topics.
- If there is no positive topic, output an empty list.
- If there is no negative topic, output an empty list.
- Output JSON only.
- Do not explain.
- Do not add markdown.

Output format:
{{"positive_topics": ["topic1"], "negative_topics": ["topic2"]}}

Target review:
{review_text}
""".strip()


def build_v2_prompt(
    review_text: str,
    allowed_topics: list[str],
    topic_definitions: dict[str, str],
    distinction_rules: list[str],
) -> str:
    allowed_list = ", ".join(allowed_topics)
    topic_context = build_topic_context(topic_definitions, allowed_topics)
    rules_context = build_rules_context(distinction_rules)

    return f"""
You are a Steam game review topic classifier.

Classify the target review into positive_topics and negative_topics.

Allowed topics:
{allowed_list}

Topic definitions:
{topic_context}

Distinction rules:
{rules_context}

Strict rules:
- Use only the allowed topics.
- Do not create new topics.
- If there is no positive topic, output an empty list.
- If there is no negative topic, output an empty list.
- Use a topic only when the review gives clear evidence.
- Output JSON only.
- Do not explain.
- Do not add markdown.

Output format:
{{"positive_topics": ["topic1"], "negative_topics": ["topic2"]}}

Target review:
{review_text}
""".strip()


def build_v3_prompt(
    review_text: str,
    allowed_topics: list[str],
    topic_definitions: dict[str, str],
    distinction_rules: list[str],
    examples: list[dict],
) -> str:
    allowed_list = ", ".join(allowed_topics)
    topic_context = build_topic_context(topic_definitions, allowed_topics)
    rules_context = build_rules_context(distinction_rules)
    examples_context = build_examples_context(examples)

    return f"""
You are a Steam game review topic classifier.

Classify the target review into positive_topics and negative_topics.

Allowed topics:
{allowed_list}

Topic definitions:
{topic_context}

Distinction rules:
{rules_context}

Similar human-verified examples:
{examples_context}

Strict rules:
- Use only the allowed topics.
- Do not create new topics.
- If there is no positive topic, output an empty list.
- If there is no negative topic, output an empty list.
- Use a topic only when the review gives clear evidence.
- Output JSON only.
- Do not explain.
- Do not add markdown.

Output format:
{{"positive_topics": ["topic1"], "negative_topics": ["topic2"]}}

Target review:
{review_text}
""".strip()


def signed_labels(
    positive_topics: list[str],
    negative_topics: list[str],
) -> list[str]:
    labels = []

    for topic in positive_topics:
        labels.append(f"pos__{topic}")

    for topic in negative_topics:
        labels.append(f"neg__{topic}")

    return labels


def evaluate_strategy(
    detail_df: pd.DataFrame,
    strategy: str,
    label_names: list[str],
) -> dict:
    strategy_df = detail_df[detail_df["strategy"] == strategy].copy()

    mlb = MultiLabelBinarizer(classes=label_names)
    mlb.fit([[]])

    y_true = mlb.transform(strategy_df["gold_labels"])
    y_pred = mlb.transform(strategy_df["pred_labels"])

    return {
        "strategy": strategy,
        "rows": len(strategy_df),
        "valid_rows": int(strategy_df["is_valid"].sum()),
        "invalid_rows": int((~strategy_df["is_valid"]).sum()),
        "valid_rate": float(strategy_df["is_valid"].mean()),
        "micro_precision": precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "micro_recall": recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "exact_match": accuracy_score(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="gemma3:27b")
    parser.add_argument("--golden-file", default=DEFAULT_GOLDEN_FILE)
    parser.add_argument("--guidelines-file", default=DEFAULT_GUIDELINES_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="0이면 골든셋 전체를 평가합니다.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rag-k", type=int, default=3)
    parser.add_argument("--rag-min-score", type=float, default=0.15)

    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_definitions, distinction_rules = load_guidelines(
        args.guidelines_file
    )
    allowed_topics = load_allowed_topics_from_golden(
        args.golden_file
    )

    missing_topics = sorted(
        set(allowed_topics) - set(topic_definitions.keys())
    )
    if missing_topics:
        raise ValueError(
            f"topic_guidelines.json에 없는 토픽: {missing_topics}"
        )

    golden_retrieval_df = load_golden_examples(
        args.golden_file
    )
    retriever = GoldenRetriever(golden_retrieval_df)

    eval_df = pd.read_csv(args.golden_file)

    if "is_valid" in eval_df.columns:
        eval_df = eval_df[
            eval_df["is_valid"].astype(str).str.lower() == "true"
        ].copy()

    if "human_verified" in eval_df.columns:
        eval_df = eval_df[
            eval_df["human_verified"].astype(str).str.lower() == "true"
        ].copy()

    eval_df = eval_df[
        eval_df["review_text_clean"].notna()
    ].copy()

    if args.limit > 0 and len(eval_df) > args.limit:
        eval_df = eval_df.sample(
            n=args.limit,
            random_state=args.seed,
        ).copy()

    eval_df = eval_df.reset_index(drop=True)

    print("평가 리뷰 수:", len(eval_df))
    print("모델:", args.model)
    print("RAG k:", args.rag_k)
    print("RAG 최소 유사도:", args.rag_min_score)

    records = []
    strategies = ["v1_topics_only", "v2_guidelines", "v3_rag"]

    for row_index, row in eval_df.iterrows():
        review_text = str(row["review_text_clean"]).strip()

        gold_positive = split_topics(row["positive_topics"])
        gold_negative = split_topics(row["negative_topics"])
        gold_signed = signed_labels(
            gold_positive,
            gold_negative,
        )

        recommendationid = row.get("recommendationid")

        rag_examples = retriever.retrieve(
            query=review_text,
            k=args.rag_k,
            min_score=args.rag_min_score,
            exclude_recommendationid=recommendationid,
            exclude_text=review_text,
        )

        prompts = {
            "v1_topics_only": build_v1_prompt(
                review_text,
                allowed_topics,
            ),
            "v2_guidelines": build_v2_prompt(
                review_text,
                allowed_topics,
                topic_definitions,
                distinction_rules,
            ),
            "v3_rag": build_v3_prompt(
                review_text,
                allowed_topics,
                topic_definitions,
                distinction_rules,
                rag_examples,
            ),
        }

        for strategy in strategies:
            try:
                raw_response = call_ollama(
                    prompt=prompts[strategy],
                    model=args.model,
                    ollama_url=args.ollama_url,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )

                (
                    pred_positive,
                    pred_negative,
                    is_valid,
                    invalid_reason,
                ) = parse_response(
                    raw_response,
                    allowed_topics,
                )

            except Exception as exc:
                raw_response = f"ERROR: {exc}"
                pred_positive = []
                pred_negative = []
                is_valid = False
                invalid_reason = "runtime_error"

            if not is_valid:
                pred_positive = []
                pred_negative = []

            pred_signed = signed_labels(
                pred_positive,
                pred_negative,
            )

            records.append(
                {
                    "row_index": row_index,
                    "recommendationid": recommendationid,
                    "appid": row.get("appid"),
                    "review_text_clean": review_text,
                    "strategy": strategy,
                    "gold_positive_topics": gold_positive,
                    "gold_negative_topics": gold_negative,
                    "pred_positive_topics": pred_positive,
                    "pred_negative_topics": pred_negative,
                    "gold_labels": gold_signed,
                    "pred_labels": pred_signed,
                    "is_valid": bool(is_valid),
                    "invalid_reason": invalid_reason,
                    "raw_response": (
                        raw_response
                        if is_valid
                        else "REJECTED_INVALID_OUTPUT"
                    ),
                    "rag_example_count": (
                        len(rag_examples)
                        if strategy == "v3_rag"
                        else 0
                    ),
                    "rag_examples": (
                        rag_examples
                        if strategy == "v3_rag"
                        else []
                    ),
                }
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(
            f"[{row_index + 1}/{len(eval_df)}] "
            f"evaluation completed"
        )

    detail_df = pd.DataFrame(records)

    json_columns = [
        "gold_positive_topics",
        "gold_negative_topics",
        "pred_positive_topics",
        "pred_negative_topics",
        "gold_labels",
        "pred_labels",
        "rag_examples",
    ]

    save_df = detail_df.copy()
    for col in json_columns:
        save_df[col] = save_df[col].apply(
            lambda value: json.dumps(
                value,
                ensure_ascii=False,
            )
        )

    safe_model = args.model.replace(":", "_").replace("/", "_")
    suffix = (
        f"{len(eval_df)}_{safe_model}"
    )

    detail_path = output_dir / (
        f"prompt_strategy_details_{suffix}.csv"
    )
    summary_path = output_dir / (
        f"prompt_strategy_summary_{suffix}.csv"
    )

    save_df.to_csv(detail_path, index=False)

    label_names = [
        f"pos__{topic}" for topic in allowed_topics
    ] + [
        f"neg__{topic}" for topic in allowed_topics
    ]

    summary_rows = [
        evaluate_strategy(
            detail_df,
            strategy,
            label_names,
        )
        for strategy in strategies
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print("\n평가 완료")
    print(summary_df.to_string(index=False))
    print("\n상세 결과:", detail_path)
    print("요약 결과:", summary_path)


if __name__ == "__main__":
    main()
