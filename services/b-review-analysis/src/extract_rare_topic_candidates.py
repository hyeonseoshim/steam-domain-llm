from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "data/processed/english_reviews_clean.csv"

DEFAULT_GOLD = (
    "services/b-review-analysis/data/annotations/"
    "topic_golden_dataset.csv"
)

DEFAULT_BASE_TRAIN = (
    "services/b-review-analysis/data/pseudo_labels/"
    "pseudo_topics_rag_v3_embedding_gemma3_27b_1000_train.csv"
)

DEFAULT_OUTPUT = (
    "services/b-review-analysis/data/pseudo_labels/"
    "rare_topic_candidates.csv"
)

DEFAULT_SUMMARY = (
    "services/b-review-analysis/data/pseudo_labels/"
    "rare_topic_candidate_summary.csv"
)


PATTERNS = {
    "pos__bugs": [
        r"\bno bugs?\b",
        r"\bbug[\s-]?free\b",
        r"\bno crashes?\b",
        r"\bnever crashed\b",
        r"\bwithout crashing\b",
        r"\bbugs? (?:were|are|have been) fixed\b",
        r"\bfixed (?:the )?bugs?\b",
        r"\bstable build\b",
        r"\bworks? perfectly\b",
        r"\bworks? flawlessly\b",
        r"\bno issues? so far\b",
    ],
    "pos__monetization": [
        r"\bno microtransactions?\b",
        r"\bnot pay[\s-]?to[\s-]?win\b",
        r"\bno pay[\s-]?to[\s-]?win\b",
        r"\bfair(?:ly)? priced dlc\b",
        r"\breasonably priced dlc\b",
        r"\bdlc (?:is|was) worth\b",
        r"\bworth (?:buying|the price).*dlc\b",
        r"\bfree dlc\b",
        r"\bmonetization (?:is|feels) fair\b",
        r"\bpremium (?:is|was) worth\b",
        r"\bhappy to support the dev",
    ],
    "pos__accessibility": [
        r"\baccessibility options?\b",
        r"\baccessibility settings?\b",
        r"\bcolor[\s-]?blind\b",
        r"\bcolour[\s-]?blind\b",
        r"\bremappable controls?\b",
        r"\bremap(?:ping)? keys?\b",
        r"\btext size option\b",
        r"\bsubtitle options?\b",
        r"\bdifficulty options?\b",
        r"\bassist mode\b",
        r"\bscreen reader\b",
        r"\bone[\s-]?handed\b",
        r"\bfull controller support\b",
    ],
    "pos__localization": [
        r"\bwell translated\b",
        r"\bgood translation\b",
        r"\bgreat translation\b",
        r"\bexcellent translation\b",
        r"\btranslation (?:is|was) (?:good|great|excellent|natural)\b",
        r"\blocalization (?:is|was) (?:good|great|excellent)\b",
        r"\bwell localized\b",
        r"\bproperly localized\b",
        r"\beverything is spelled correctly\b",
        r"\benglish (?:reads|sounds) naturally\b",
        r"\bgood language support\b",
    ],
    "neg__network_server": [
        r"\bservers?\b",
        r"\bdisconnect(?:ed|ing|s)?\b",
        r"\bconnection (?:issue|issues|problem|problems|error|errors)\b",
        r"\bconnection (?:lost|failed|timed out)\b",
        r"\bhigh ping\b",
        r"\blatency\b",
        r"\bpacket loss\b",
        r"\bde[\s-]?sync\b",
        r"\bd[\s-]?sync\b",
        r"\bonline (?:does not|doesn't|did not|didn't) work\b",
        r"\bmatchmaking (?:does not|doesn't|did not|didn't) work\b",
        r"\bcannot connect\b",
        r"\bcan't connect\b",
        r"\blogin (?:issue|issues|problem|problems|failed)\b",
    ],
    "neg__localization": [
        r"\bbad translation\b",
        r"\bpoor translation\b",
        r"\bterrible translation\b",
        r"\bbroken english\b",
        r"\bengrish\b",
        r"\bmistranslat(?:ed|ion)\b",
        r"\btranslation (?:is|was) (?:bad|poor|terrible|awful|broken)\b",
        r"\bpoor localization\b",
        r"\bbad localization\b",
        r"\btypos?\b",
        r"\bspelling errors?\b",
        r"\bno english\b",
        r"\bmissing english\b",
        r"\bno language support\b",
        r"\bsubtitle errors?\b",
    ],
}


def load_excluded_ids(gold_path: str, train_path: str) -> set[str]:
    excluded = set()

    for path_str in [gold_path, train_path]:
        path = Path(path_str)

        if not path.exists():
            continue

        df = pd.read_csv(
            path,
            usecols=["recommendationid"],
            dtype={"recommendationid": str},
        )

        excluded.update(
            df["recommendationid"]
            .dropna()
            .astype(str)
            .tolist()
        )

    return excluded


def compile_patterns() -> dict[str, re.Pattern]:
    return {
        label: re.compile(
            "|".join(f"(?:{pattern})" for pattern in patterns),
            flags=re.IGNORECASE,
        )
        for label, patterns in PATTERNS.items()
    }


def reservoir_add(
    reservoir: list[dict],
    row: dict,
    seen_count: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(row)
        return

    replacement_index = rng.randint(0, seen_count - 1)

    if replacement_index < capacity:
        reservoir[replacement_index] = row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--base-train", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)

    parser.add_argument("--per-target", type=int, default=120)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_path = Path(args.output)
    summary_path = Path(args.summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_ids = load_excluded_ids(
        args.gold,
        args.base_train,
    )

    compiled_patterns = compile_patterns()
    rng = random.Random(args.seed)

    reservoirs = {
        label: []
        for label in PATTERNS
    }

    seen_counts = {
        label: 0
        for label in PATTERNS
    }

    usecols = [
        "recommendationid",
        "appid",
        "review_text_clean",
        "voted_up",
        "label",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            args.input,
            usecols=usecols,
            dtype={
                "recommendationid": str,
                "appid": str,
            },
            chunksize=args.chunk_size,
        ),
        start=1,
    ):
        chunk = chunk[
            chunk["review_text_clean"].notna()
        ].copy()

        chunk["review_text_clean"] = (
            chunk["review_text_clean"]
            .astype(str)
            .str.strip()
        )

        chunk = chunk[
            chunk["review_text_clean"] != ""
        ].copy()

        chunk = chunk[
            ~chunk["recommendationid"]
            .astype(str)
            .isin(excluded_ids)
        ].copy()

        for target_label, pattern in compiled_patterns.items():
            mask = chunk["review_text_clean"].str.contains(
                pattern,
                na=False,
            )

            matched = chunk.loc[mask]

            for _, row in matched.iterrows():
                seen_counts[target_label] += 1

                candidate = {
                    "recommendationid": str(
                        row["recommendationid"]
                    ),
                    "appid": str(row["appid"]),
                    "review_text_clean": row[
                        "review_text_clean"
                    ],
                    "voted_up": row["voted_up"],
                    "label": row["label"],
                    "target_label": target_label,
                }

                reservoir_add(
                    reservoirs[target_label],
                    candidate,
                    seen_counts[target_label],
                    args.per_target,
                    rng,
                )

        print(
            f"[chunk {chunk_number}] "
            + ", ".join(
                f"{label}={seen_counts[label]}"
                for label in PATTERNS
            )
        )

    sampled_rows = []

    for target_label, rows in reservoirs.items():
        sampled_rows.extend(rows)

    sampled_df = pd.DataFrame(sampled_rows)

    if sampled_df.empty:
        raise RuntimeError(
            "키워드와 일치하는 후보 리뷰를 찾지 못했습니다."
        )

    # 같은 리뷰가 여러 목표 라벨 후보일 경우 한 번만 추론하도록 합칩니다.
    aggregated_rows = []

    for recommendationid, group in sampled_df.groupby(
        "recommendationid",
        sort=False,
    ):
        first = group.iloc[0]

        aggregated_rows.append(
            {
                "recommendationid": recommendationid,
                "appid": first["appid"],
                "review_text_clean": first[
                    "review_text_clean"
                ],
                "voted_up": first["voted_up"],
                "label": first["label"],
                "target_labels": "|".join(
                    sorted(
                        set(group["target_label"].tolist())
                    )
                ),
            }
        )

    candidate_df = pd.DataFrame(aggregated_rows)
    candidate_df.to_csv(output_path, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "target_label": label,
                "matched_in_source": seen_counts[label],
                "sampled_before_dedup": len(
                    reservoirs[label]
                ),
                "candidate_file_mentions": int(
                    candidate_df["target_labels"]
                    .str.split("|", regex=False)
                    .apply(lambda values: label in values)
                    .sum()
                ),
            }
            for label in PATTERNS
        ]
    )

    summary_df.to_csv(summary_path, index=False)

    print("\n후보 추출 완료")
    print("골든셋·기존 학습 데이터 제외 ID:", len(excluded_ids))
    print("최종 고유 후보 리뷰:", len(candidate_df))
    print("\n후보 요약:")
    print(summary_df.to_string(index=False))
    print("\n후보 파일:", output_path)
    print("요약 파일:", summary_path)


if __name__ == "__main__":
    main()
