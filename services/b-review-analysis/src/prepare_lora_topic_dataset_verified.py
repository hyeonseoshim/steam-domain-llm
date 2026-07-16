from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


INPUT_PATH = Path(
    "services/b-review-analysis/data/pseudo_labels/"
    "pseudo_topics_rag_v3_embedding_gemma3_27b_final_train_pool_verified.csv"
)

GUIDELINES_PATH = Path(
    "services/b-review-analysis/data/annotations/"
    "topic_guidelines.json"
)

OUTPUT_DIR = Path(
    "services/b-review-analysis/data/"
    "lora_topic_dataset_verified"
)

VALID_RATIO = 0.10
RANDOM_STATE = 42


def parse_topics(value) -> list[str]:
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

    try:
        parsed = ast.literal_eval(text)
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
        for item in text.split(",")
        if item.strip()
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_PATH,
        dtype={"recommendationid": str},
    )

    required_columns = [
        "recommendationid",
        "input_text",
        "positive_topics",
        "negative_topics",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df[df["input_text"].notna()].copy()

    df["input_text"] = (
        df["input_text"]
        .astype(str)
        .str.strip()
    )

    df = df[df["input_text"] != ""].copy()

    df = df.drop_duplicates(
        subset=["recommendationid"],
        keep="first",
    ).reset_index(drop=True)

    with open(
        GUIDELINES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        guidelines = json.load(file)

    topics = sorted(guidelines["topics"].keys())

    label_names = (
        [f"pos__{topic}" for topic in topics]
        + [f"neg__{topic}" for topic in topics]
    )

    label_to_id = {
        label: index
        for index, label in enumerate(label_names)
    }

    vectors = []
    positive_lists = []
    negative_lists = []

    for _, row in df.iterrows():
        positive = parse_topics(row["positive_topics"])
        negative = parse_topics(row["negative_topics"])

        vector = np.zeros(
            len(label_names),
            dtype=np.int64,
        )

        for topic in positive:
            label = f"pos__{topic}"

            if label not in label_to_id:
                raise ValueError(
                    f"허용되지 않은 라벨: {label}"
                )

            vector[label_to_id[label]] = 1

        for topic in negative:
            label = f"neg__{topic}"

            if label not in label_to_id:
                raise ValueError(
                    f"허용되지 않은 라벨: {label}"
                )

            vector[label_to_id[label]] = 1

        positive_lists.append(positive)
        negative_lists.append(negative)
        vectors.append(vector)

    y = np.stack(vectors)

    df["positive_topics"] = [
        json.dumps(value, ensure_ascii=False)
        for value in positive_lists
    ]

    df["negative_topics"] = [
        json.dumps(value, ensure_ascii=False)
        for value in negative_lists
    ]

    df["labels"] = [
        json.dumps(
            vector.tolist(),
            ensure_ascii=False,
        )
        for vector in vectors
    ]

    empty_indices = np.where(
        y.sum(axis=1) == 0
    )[0]

    non_empty_indices = np.where(
        y.sum(axis=1) > 0
    )[0]

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(empty_indices)

    target_valid_size = int(
        round(len(df) * VALID_RATIO)
    )

    valid_empty_size = max(
        1,
        int(round(len(empty_indices) * VALID_RATIO)),
    )

    valid_empty_size = min(
        valid_empty_size,
        len(empty_indices),
    )

    valid_non_empty_size = (
        target_valid_size - valid_empty_size
    )

    if valid_non_empty_size <= 0:
        raise ValueError(
            "검증셋의 non-empty 크기가 0 이하입니다."
        )

    non_empty_y = y[non_empty_indices]

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=valid_non_empty_size,
        random_state=RANDOM_STATE,
    )

    dummy_x = np.zeros(
        (len(non_empty_indices), 1),
        dtype=np.int64,
    )

    (
        train_non_empty_local,
        valid_non_empty_local,
    ) = next(
        splitter.split(dummy_x, non_empty_y)
    )

    train_non_empty_indices = (
        non_empty_indices[train_non_empty_local]
    )

    valid_non_empty_indices = (
        non_empty_indices[valid_non_empty_local]
    )

    valid_empty_indices = (
        empty_indices[:valid_empty_size]
    )

    train_empty_indices = (
        empty_indices[valid_empty_size:]
    )

    train_indices = np.concatenate(
        [
            train_non_empty_indices,
            train_empty_indices,
        ]
    )

    valid_indices = np.concatenate(
        [
            valid_non_empty_indices,
            valid_empty_indices,
        ]
    )

    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)

    train_df = df.iloc[train_indices].copy()
    valid_df = df.iloc[valid_indices].copy()

    train_y = y[train_indices]
    valid_y = y[valid_indices]

    save_columns = [
        "recommendationid",
        "appid",
        "input_text",
        "positive_topics",
        "negative_topics",
        "labels",
    ]

    save_columns = [
        column
        for column in save_columns
        if column in df.columns
    ]

    train_df[save_columns].to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    valid_df[save_columns].to_csv(
        OUTPUT_DIR / "valid.csv",
        index=False,
    )

    label_mapping = {
        "topics": topics,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "id_to_label": {
            str(index): label
            for label, index in label_to_id.items()
        },
        "num_topics": len(topics),
        "num_labels": len(label_names),
    }

    with open(
        OUTPUT_DIR / "label_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            label_mapping,
            file,
            ensure_ascii=False,
            indent=2,
        )

    distribution = pd.DataFrame(
        {
            "label": label_names,
            "total_count": y.sum(axis=0),
            "train_count": train_y.sum(axis=0),
            "valid_count": valid_y.sum(axis=0),
        }
    )

    distribution.to_csv(
        OUTPUT_DIR / "label_distribution.csv",
        index=False,
    )

    missing_in_train = distribution[
        (distribution["total_count"] > 0)
        & (distribution["train_count"] == 0)
    ]["label"].tolist()

    missing_in_valid = distribution[
        (distribution["total_count"] > 0)
        & (distribution["valid_count"] == 0)
    ]["label"].tolist()

    summary = {
        "source_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "topic_count": int(len(topics)),
        "label_count": int(len(label_names)),
        "total_empty_label_rows": int(
            (y.sum(axis=1) == 0).sum()
        ),
        "train_empty_label_rows": int(
            (train_y.sum(axis=1) == 0).sum()
        ),
        "valid_empty_label_rows": int(
            (valid_y.sum(axis=1) == 0).sum()
        ),
        "labels_missing_in_train": missing_in_train,
        "labels_missing_in_valid": missing_in_valid,
        "random_state": RANDOM_STATE,
        "valid_ratio": VALID_RATIO,
    }

    with open(
        OUTPUT_DIR / "dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("최종 LoRA 데이터 분할 완료")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("저장 폴더:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
