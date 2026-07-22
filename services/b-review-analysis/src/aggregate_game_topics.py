#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = (
    "services/b-review-analysis/data/predictions/"
    "review_topic_predictions_parts"
)

DEFAULT_OUTPUT_DIR = (
    "services/b-review-analysis/data/predictions/"
    "game_topic_summary"
)


TOPIC_MAPPING = {
    "accessibility": "other",
    "audio": "audio",
    "bugs": "bugs",
    "content": "content",
    "controls": "controls",
    "difficulty_balance": "difficulty_balance",
    "gameplay": "gameplay",
    "graphics": "graphics",
    "localization": "translation",
    "monetization": "monetization",
    "multiplayer": "online",
    "network_server": "online",
    "other": "other",
    "performance": "performance",
    "replayability": "content",
    "story": "story",
    "ui_ux": "ui",
    "updates_support": "updates_support",
    "value": "value",
}

API_TOPICS = {
    "gameplay",
    "graphics",
    "story",
    "audio",
    "controls",
    "performance",
    "bugs",
    "difficulty_balance",
    "content",
    "value",
    "monetization",
    "updates_support",
    "ui",
    "online",
    "translation",
    "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate review topic predictions by Steam appid."
        )
    )

    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Maximum top topics per sentiment",
    )

    return parser.parse_args()


def parse_topic_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    text = str(value).strip()

    if not text or text == "[]":
        return []

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(
            f"Topic value is not a list: {value}"
        )

    return [str(item) for item in parsed]


def map_topics(topics: list[str]) -> set[str]:
    mapped = set()

    for topic in topics:
        if topic not in TOPIC_MAPPING:
            raise ValueError(
                f"Unknown internal topic: {topic}"
            )

        api_topic = TOPIC_MAPPING[topic]

        if api_topic not in API_TOPICS:
            raise ValueError(
                f"Invalid API topic: {api_topic}"
            )

        mapped.add(api_topic)

    return mapped


def sorted_topic_counts(
    counter: Counter[str],
) -> dict[str, int]:
    return {
        topic: count
        for topic, count in sorted(
            counter.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    }


def select_top_topics(
    counter: Counter[str],
    top_n: int,
) -> list[str]:
    if not counter:
        return []

    candidates = list(counter.items())

    specific_topics = [
        item
        for item in candidates
        if item[0] != "other"
    ]

    # 구체적인 토픽이 존재하면 other는 상위 토픽에서 제외합니다.
    if specific_topics:
        candidates = specific_topics

    candidates.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )

    return [
        topic
        for topic, _ in candidates[:top_n]
    ]


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = sorted(
        input_dir.glob("part-*.csv.gz")
    )

    if not files:
        raise FileNotFoundError(
            f"No prediction files found in {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_counts: Counter[int] = Counter()

    positive_counts: dict[
        int,
        Counter[str],
    ] = defaultdict(Counter)

    negative_counts: dict[
        int,
        Counter[str],
    ] = defaultdict(Counter)

    total_rows = 0

    print("Input files:", len(files))
    print("Top N:", args.top_n)

    for file_index, path in enumerate(files, start=1):
        df = pd.read_csv(
            path,
            usecols=[
                "appid",
                "positive_topics",
                "negative_topics",
            ],
        )

        for row in df.itertuples(index=False):
            appid = int(row.appid)

            positive_topics = map_topics(
                parse_topic_list(
                    row.positive_topics
                )
            )

            negative_topics = map_topics(
                parse_topic_list(
                    row.negative_topics
                )
            )

            review_counts[appid] += 1
            positive_counts[appid].update(
                positive_topics
            )
            negative_counts[appid].update(
                negative_topics
            )

        total_rows += len(df)

        if (
            file_index == 1
            or file_index % 10 == 0
            or file_index == len(files)
        ):
            print(
                f"[{file_index}/{len(files)}] "
                f"rows={total_rows:,} "
                f"games={len(review_counts):,}"
            )

    rows = []
    api_data = {}

    for appid in sorted(review_counts):
        positive_counter = positive_counts[appid]
        negative_counter = negative_counts[appid]

        top_positive = select_top_topics(
            positive_counter,
            args.top_n,
        )

        top_negative = select_top_topics(
            negative_counter,
            args.top_n,
        )

        positive_topic_counts = (
            sorted_topic_counts(
                positive_counter
            )
        )

        negative_topic_counts = (
            sorted_topic_counts(
                negative_counter
            )
        )

        record = {
            "appid": appid,
            "review_count": review_counts[appid],
            "top_positive_topics": top_positive,
            "top_negative_topics": top_negative,
            "positive_topic_counts": (
                positive_topic_counts
            ),
            "negative_topic_counts": (
                negative_topic_counts
            ),
        }

        api_data[str(appid)] = record

        rows.append(
            {
                "appid": appid,
                "review_count": review_counts[appid],
                "top_positive_topics": json.dumps(
                    top_positive,
                    ensure_ascii=False,
                ),
                "top_negative_topics": json.dumps(
                    top_negative,
                    ensure_ascii=False,
                ),
                "positive_topic_counts": json.dumps(
                    positive_topic_counts,
                    ensure_ascii=False,
                ),
                "negative_topic_counts": json.dumps(
                    negative_topic_counts,
                    ensure_ascii=False,
                ),
            }
        )

    summary_df = pd.DataFrame(rows)

    csv_path = (
        output_dir
        / "game_topic_summary.csv.gz"
    )

    json_path = (
        output_dir
        / "game_topic_summary.json.gz"
    )

    audit_path = (
        output_dir
        / "aggregation_summary.json"
    )

    summary_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    with gzip.open(
        json_path,
        "wt",
        encoding="utf-8",
    ) as file:
        json.dump(
            api_data,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    aggregation_summary = {
        "input_files": len(files),
        "total_reviews": total_rows,
        "unique_appids": len(review_counts),
        "top_n": args.top_n,
        "internal_topic_count": len(TOPIC_MAPPING),
        "api_topic_count": len(API_TOPICS),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            aggregation_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nAggregation completed")
    print(
        json.dumps(
            aggregation_summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
