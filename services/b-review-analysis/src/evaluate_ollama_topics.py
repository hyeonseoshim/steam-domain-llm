from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]

GOLD_PATH = (
    PROJECT_DIR
    / "data"
    / "annotations"
    / "topic_golden_dataset.csv"
)

TOPIC_SCHEMA_PATH = (
    PROJECT_DIR
    / "config"
    / "topic_schema.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "ollama_topic_eval"
)

DEFAULT_MODELS = [
    "qwen3:14b",
    "phi4:14b",
    "gemma3:12b",
    "qwen3.5:9b",
    "qwen2.5:7b",
]

ALLOWED_TOPICS = [
    "gameplay",
    "story",
    "graphics",
    "audio",
    "controls",
    "ui_ux",
    "performance",
    "bugs",
    "difficulty_balance",
    "content",
    "replayability",
    "value",
    "multiplayer",
    "network_server",
    "updates_support",
    "monetization",
    "accessibility",
    "localization",
    "other",
]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def split_topics(value: Any) -> list[str]:
    if value is None:
        return []

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    return [
        topic.strip()
        for topic in text.split(",")
        if topic.strip()
    ]


def normalize_topics(topics: Any) -> list[str]:
    if not isinstance(topics, list):
        return []

    cleaned: list[str] = []

    for topic in topics:
        topic = str(topic).strip()

        if topic in ALLOWED_TOPICS and topic not in cleaned:
            cleaned.append(topic)

    if len(cleaned) > 1 and "other" in cleaned:
        cleaned.remove("other")

    return cleaned


def safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def build_system_prompt() -> str:
    schema = json.loads(
        TOPIC_SCHEMA_PATH.read_text(encoding="utf-8")
    )

    topic_guide = "\n".join(
        f"- {code}: {info['description']}"
        for code, info in schema.items()
    )

    return f"""
You are a strict Steam review topic classifier.

Classify the given English Steam review into positive and negative game-aspect topics.

Allowed topics:
{topic_guide}

Rules:
1. Use only the allowed topic codes.
2. Judge only from the review text. Do not use voted_up or any external information.
3. Put praised aspects in positive_topics.
4. Put criticized aspects in negative_topics.
5. Select the minimum set of topics with direct evidence.
6. Do not add a topic only because it is indirectly related.
7. Do not duplicate a cause and its consequence as separate topics.
8. If controls cause the game to be unenjoyable, choose controls, not gameplay, unless gameplay itself is independently criticized.
9. If crashes, launch failures, freezes, missing purchased content, or broken functions are mentioned, use bugs.
10. If FPS, lag, stuttering, loading, or optimization is mentioned, use performance.
11. If server disconnection, high ping, or connection failure is mentioned, use network_server.
12. If lack of online players or nobody to play with is mentioned, use multiplayer.
13. If DLC, microtransactions, pay-to-win, paid content, or monetization policy is evaluated, use monetization.
14. If price, worth, discount, waste of money, refund, or scam is evaluated, use value.
15. If content amount, maps, quests, levels, DLC content, or modes are evaluated, use content.
16. If replay value, another playthrough, repeated runs, or replay motivation is evaluated, use replayability.
17. If the same action repeats within one playthrough, use gameplay, not replayability.
18. If difficulty, fairness, enemy strength, balance, or no counterplay is evaluated, use difficulty_balance.
19. If menus, HUD, tutorial explanation, inventory, unclear information, or interface usability is evaluated, use ui_ux.
20. If keyboard, mouse, controller, camera, keybinding, zoom, or input behavior is evaluated, use controls.
21. If developer updates, patch support, abandonment, or support response is evaluated, use updates_support.
22. Do not choose updates_support only because the user looked for patches.
23. If translation or language support is evaluated, use localization.
24. If subtitles, text size, colorblind mode, or accessibility options are evaluated, use accessibility.
25. Use other only for meaningful overall praise or criticism without a specific topic.
26. In the same direction, do not combine other with specific topics.
27. A non-English review should be is_valid false.
28. If the review has no meaningful game evaluation, set is_valid false.
29. If is_valid is false, both topic lists must be empty.
30. Return only valid JSON. No markdown. No explanation.

Output JSON schema:
{{
  "positive_topics": ["gameplay"],
  "negative_topics": ["bugs"],
  "is_valid": true
}}
""".strip()


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()

    # ```json ... ``` 제거
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 본문 중 첫 JSON 객체만 추출
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def call_ollama(
    model: str,
    system_prompt: str,
    review_text: str,
    timeout: int,
) -> tuple[dict[str, Any] | None, str, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Classify this review. Return only JSON.\n\n"
                    f"Review:\n{review_text}"
                ),
            },
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 256,
        },
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))

    elapsed = time.perf_counter() - started_at
    content = body.get("message", {}).get("content", "")

    return extract_json(content), content, elapsed


def load_gold(limit: int | None) -> list[dict[str, Any]]:
    with GOLD_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if limit is not None:
        rows = rows[:limit]

    return rows


def signed_labels(
    positive_topics: list[str],
    negative_topics: list[str],
) -> set[str]:
    labels = set()

    for topic in positive_topics:
        labels.add(f"positive:{topic}")

    for topic in negative_topics:
        labels.add(f"negative:{topic}")

    return labels


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)

    json_success = sum(
        parse_bool(row["json_success"])
        for row in rows
    )

    valid_correct = 0
    positive_exact = 0
    negative_exact = 0
    topic_exact = 0
    all_exact = 0

    tp = 0
    fp = 0
    fn = 0

    per_label = {
        f"{direction}:{topic}": {"tp": 0, "fp": 0, "fn": 0}
        for direction in ["positive", "negative"]
        for topic in ALLOWED_TOPICS
    }

    latencies = []

    for row in rows:
        gold_positive = split_topics(row["gold_positive_topics"])
        gold_negative = split_topics(row["gold_negative_topics"])
        pred_positive = split_topics(row["pred_positive_topics"])
        pred_negative = split_topics(row["pred_negative_topics"])

        gold_valid = parse_bool(row["gold_is_valid"])
        pred_valid = parse_bool(row["pred_is_valid"])

        if gold_valid == pred_valid:
            valid_correct += 1

        if set(gold_positive) == set(pred_positive):
            positive_exact += 1

        if set(gold_negative) == set(pred_negative):
            negative_exact += 1

        if (
            set(gold_positive) == set(pred_positive)
            and set(gold_negative) == set(pred_negative)
        ):
            topic_exact += 1

        if (
            gold_valid == pred_valid
            and set(gold_positive) == set(pred_positive)
            and set(gold_negative) == set(pred_negative)
        ):
            all_exact += 1

        gold_signed = signed_labels(
            gold_positive,
            gold_negative,
        )

        pred_signed = signed_labels(
            pred_positive,
            pred_negative,
        )

        tp += len(gold_signed & pred_signed)
        fp += len(pred_signed - gold_signed)
        fn += len(gold_signed - pred_signed)

        for label in per_label:
            in_gold = label in gold_signed
            in_pred = label in pred_signed

            if in_gold and in_pred:
                per_label[label]["tp"] += 1
            elif not in_gold and in_pred:
                per_label[label]["fp"] += 1
            elif in_gold and not in_pred:
                per_label[label]["fn"] += 1

        try:
            latencies.append(float(row["latency_sec"]))
        except ValueError:
            pass

    micro_precision = tp / (tp + fp) if tp + fp else 0
    micro_recall = tp / (tp + fn) if tp + fn else 0

    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0
    )

    f1_values = []

    for counts in per_label.values():
        label_tp = counts["tp"]
        label_fp = counts["fp"]
        label_fn = counts["fn"]

        # 골든셋에 등장하지 않는 라벨은 macro 계산에서 제외
        if label_tp + label_fn == 0:
            continue

        precision = (
            label_tp / (label_tp + label_fp)
            if label_tp + label_fp
            else 0
        )

        recall = (
            label_tp / (label_tp + label_fn)
            if label_tp + label_fn
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0
        )

        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0

    avg_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    return {
        "n": total,
        "json_success_rate": json_success / total if total else 0,
        "is_valid_accuracy": valid_correct / total if total else 0,
        "positive_exact_match": positive_exact / total if total else 0,
        "negative_exact_match": negative_exact / total if total else 0,
        "topic_exact_match": topic_exact / total if total else 0,
        "all_exact_match": all_exact / total if total else 0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "avg_latency_sec": avg_latency,
    }


def run_model(
    model: str,
    gold_rows: list[dict[str, Any]],
    timeout: int,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt()
    output_path = OUTPUT_DIR / f"predictions_{safe_model_name(model)}.csv"

    fieldnames = [
        "model",
        "annotation_id",
        "review_text_clean",
        "gold_positive_topics",
        "gold_negative_topics",
        "gold_is_valid",
        "pred_positive_topics",
        "pred_negative_topics",
        "pred_is_valid",
        "json_success",
        "raw_response",
        "error",
        "latency_sec",
    ]

    existing_ids = set()

    if output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as file:
            existing_ids = {
                row["annotation_id"]
                for row in csv.DictReader(file)
            }

    file_exists = output_path.exists()

    with output_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for index, row in enumerate(gold_rows, start=1):
            annotation_id = str(row["annotation_id"])

            if annotation_id in existing_ids:
                continue

            review_text = row["review_text_clean"]

            try:
                parsed, raw_response, latency = call_ollama(
                    model=model,
                    system_prompt=system_prompt,
                    review_text=review_text,
                    timeout=timeout,
                )

                if parsed is None:
                    pred_positive = []
                    pred_negative = []
                    pred_is_valid = False
                    json_success = False
                else:
                    pred_positive = normalize_topics(
                        parsed.get("positive_topics", [])
                    )

                    pred_negative = normalize_topics(
                        parsed.get("negative_topics", [])
                    )

                    pred_is_valid = bool(
                        parsed.get("is_valid", False)
                    )

                    if not pred_is_valid:
                        pred_positive = []
                        pred_negative = []

                    json_success = True

                error = ""

            except Exception as exc:
                pred_positive = []
                pred_negative = []
                pred_is_valid = False
                json_success = False
                raw_response = ""
                latency = 0
                error = f"{type(exc).__name__}: {exc}"

            writer.writerow(
                {
                    "model": model,
                    "annotation_id": annotation_id,
                    "review_text_clean": review_text,
                    "gold_positive_topics": row["positive_topics"],
                    "gold_negative_topics": row["negative_topics"],
                    "gold_is_valid": row["is_valid"],
                    "pred_positive_topics": ",".join(pred_positive),
                    "pred_negative_topics": ",".join(pred_negative),
                    "pred_is_valid": pred_is_valid,
                    "json_success": json_success,
                    "raw_response": raw_response,
                    "error": error,
                    "latency_sec": round(latency, 4),
                }
            )

            print(
                f"{model} [{index}/{len(gold_rows)}] "
                f"annotation_id={annotation_id} "
                f"{'OK' if not error else 'ERROR'}"
            )

    return output_path


def write_summary(prediction_paths: list[Path]) -> None:
    summary_path = OUTPUT_DIR / "model_summary.csv"

    fieldnames = [
        "model",
        "n",
        "json_success_rate",
        "is_valid_accuracy",
        "positive_exact_match",
        "negative_exact_match",
        "topic_exact_match",
        "all_exact_match",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_f1",
        "avg_latency_sec",
    ]

    rows_to_write = []

    for path in prediction_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))

        if not rows:
            continue

        metrics = evaluate_rows(rows)
        metrics["model"] = rows[0]["model"]

        rows_to_write.append(metrics)

    rows_to_write.sort(
        key=lambda item: (
            item["micro_f1"],
            item["macro_f1"],
            item["all_exact_match"],
        ),
        reverse=True,
    )

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows_to_write:
            writer.writerow(
                {
                    key: (
                        round(row[key], 4)
                        if isinstance(row.get(key), float)
                        else row.get(key)
                    )
                    for key in fieldnames
                }
            )

    print()
    print(f"요약 저장: {summary_path}")

    for row in rows_to_write:
        print(
            f"{row['model']}: "
            f"micro_f1={row['micro_f1']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"exact={row['all_exact_match']:.4f}, "
            f"json={row['json_success_rate']:.4f}, "
            f"latency={row['avg_latency_sec']:.2f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
    )

    args = parser.parse_args()

    gold_rows = load_gold(args.limit)
    prediction_paths = []

    print(f"평가 행 수: {len(gold_rows)}")
    print(f"모델: {args.models}")

    for model in args.models:
        path = run_model(
            model=model,
            gold_rows=gold_rows,
            timeout=args.timeout,
        )
        prediction_paths.append(path)

    write_summary(prediction_paths)


if __name__ == "__main__":
    main()
