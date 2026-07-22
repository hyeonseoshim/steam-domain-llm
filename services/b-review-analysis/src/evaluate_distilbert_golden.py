#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default=(
            "services/b-review-analysis/models/"
            "distilbert_topic_lora/merged_model"
        ),
    )
    parser.add_argument(
        "--golden-path",
        default=(
            "services/b-review-analysis/data/annotations/"
            "topic_golden_dataset.csv"
        ),
    )
    parser.add_argument(
        "--label-mapping-path",
        default=(
            "services/b-review-analysis/data/"
            "lora_topic_dataset_verified/label_mapping.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "services/b-review-analysis/reports/"
            "distilbert_topic_lora/golden_eval"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.39)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)

    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def parse_topic_list(value: Any) -> list[str]:
    """골든셋 토픽 값을 정규화하여 리스트로 반환합니다."""

    def split_item(item: Any) -> list[str]:
        if item is None:
            return []

        text = str(item).strip()

        if text.lower() in {
            "",
            "nan",
            "none",
            "null",
            "[]",
        }:
            return []

        text = text.strip().strip("[](){}").strip()
        text = text.strip('"').strip("'").strip()

        # 세미콜론과 파이프도 쉼표로 통일
        text = text.replace(";", ",").replace("|", ",")

        topics = []

        for part in text.split(","):
            topic = part.strip()
            topic = topic.strip('"').strip("'").strip()

            if topic and topic.lower() not in {
                "nan",
                "none",
                "null",
            }:
                topics.append(topic)

        return topics

    if isinstance(value, (list, tuple, set)):
        parsed_items = list(value)

    else:
        if pd.isna(value):
            return []

        text = str(value).strip()

        if text.lower() in {
            "",
            "nan",
            "none",
            "null",
            "[]",
        }:
            return []

        parsed_items = None

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (
                json.JSONDecodeError,
                ValueError,
                SyntaxError,
            ):
                continue

            if isinstance(parsed, (list, tuple, set)):
                parsed_items = list(parsed)
            else:
                parsed_items = [parsed]

            break

        if parsed_items is None:
            parsed_items = [text]

    result = []

    for item in parsed_items:
        if isinstance(item, (list, tuple, set)):
            for nested_item in item:
                result.extend(split_item(nested_item))
        else:
            result.extend(split_item(item))

    return result


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    golden_df = pd.read_csv(args.golden_path)

    print("Original golden rows:", len(golden_df))

    if "human_verified" in golden_df.columns:
        golden_df = golden_df[
            golden_df["human_verified"].apply(parse_bool)
        ].copy()

    if "annotation_status" in golden_df.columns:
        golden_df = golden_df[
            golden_df["annotation_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("completed")
        ].copy()

    if "is_valid" in golden_df.columns:
        golden_df = golden_df[
            golden_df["is_valid"].apply(parse_bool)
        ].copy()

    golden_df = golden_df.reset_index(drop=True)

    print("Filtered valid golden rows:", len(golden_df))

    if "review_text_clean" in golden_df.columns:
        text_column = "review_text_clean"
    elif "input_text" in golden_df.columns:
        text_column = "input_text"
    elif "review_text" in golden_df.columns:
        text_column = "review_text"
    else:
        raise ValueError("Review text column not found")

    with open(args.label_mapping_path, encoding="utf-8") as file:
        mapping = json.load(file)

    label_names = mapping["label_names"]
    label_to_id = mapping["label_to_id"]
    num_labels = mapping["num_labels"]

    true_labels = np.zeros(
        (len(golden_df), num_labels),
        dtype=np.int32,
    )

    true_positive_topics = []
    true_negative_topics = []

    for row_index, row in golden_df.iterrows():
        positive_topics = parse_topic_list(
            row["positive_topics"]
        )
        negative_topics = parse_topic_list(
            row["negative_topics"]
        )

        true_positive_topics.append(positive_topics)
        true_negative_topics.append(negative_topics)

        for topic in positive_topics:
            label = f"pos__{topic}"

            if label not in label_to_id:
                raise ValueError(
                    f"Unknown positive topic: {topic}"
                )

            true_labels[
                row_index,
                label_to_id[label],
            ] = 1

        for topic in negative_topics:
            label = f"neg__{topic}"

            if label not in label_to_id:
                raise ValueError(
                    f"Unknown negative topic: {topic}"
                )

            true_labels[
                row_index,
                label_to_id[label],
            ] = 1

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Threshold:", args.threshold)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir
    )

    model.to(device)
    model.eval()

    texts = (
        golden_df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    logits_batches = []

    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[
                start : start + args.batch_size
            ]

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(**encoded)

            logits_batches.append(
                outputs.logits.detach().cpu().numpy()
            )

    logits = np.concatenate(logits_batches, axis=0)
    probabilities = sigmoid(logits)

    predictions = (
        probabilities >= args.threshold
    ).astype(np.int32)

    micro_precision, micro_recall, micro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="micro",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    exact_match = float(
        np.mean(
            np.all(
                predictions == true_labels,
                axis=1,
            )
        )
    )

    (
        per_label_precision,
        per_label_recall,
        per_label_f1,
        per_label_support,
    ) = precision_recall_fscore_support(
        true_labels,
        predictions,
        average=None,
        zero_division=0,
    )

    per_label_df = pd.DataFrame(
        {
            "label": label_names,
            "support": per_label_support.astype(int),
            "predicted_count": predictions.sum(
                axis=0
            ).astype(int),
            "precision": per_label_precision,
            "recall": per_label_recall,
            "f1": per_label_f1,
            "mean_probability": probabilities.mean(axis=0),
        }
    )

    per_label_path = output_dir / "per_label_metrics.csv"

    per_label_df.to_csv(
        per_label_path,
        index=False,
        encoding="utf-8",
    )

    prediction_rows = []

    for row_index, row in golden_df.iterrows():
        predicted_labels = [
            label_names[label_index]
            for label_index, selected in enumerate(
                predictions[row_index]
            )
            if selected == 1
        ]

        predicted_positive = [
            label.replace("pos__", "", 1)
            for label in predicted_labels
            if label.startswith("pos__")
        ]

        predicted_negative = [
            label.replace("neg__", "", 1)
            for label in predicted_labels
            if label.startswith("neg__")
        ]

        prediction_row = {
            "row_index": row_index,
            "review_text": texts[row_index],
            "true_positive_topics": json.dumps(
                true_positive_topics[row_index],
                ensure_ascii=False,
            ),
            "true_negative_topics": json.dumps(
                true_negative_topics[row_index],
                ensure_ascii=False,
            ),
            "predicted_positive_topics": json.dumps(
                predicted_positive,
                ensure_ascii=False,
            ),
            "predicted_negative_topics": json.dumps(
                predicted_negative,
                ensure_ascii=False,
            ),
            "exact_match": bool(
                np.array_equal(
                    predictions[row_index],
                    true_labels[row_index],
                )
            ),
        }

        for column in [
            "annotation_id",
            "recommendationid",
            "appid",
        ]:
            if column in golden_df.columns:
                prediction_row[column] = row[column]

        prediction_rows.append(prediction_row)

    predictions_path = output_dir / "predictions.csv"

    pd.DataFrame(prediction_rows).to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )

    summary = {
        "model_dir": args.model_dir,
        "golden_path": args.golden_path,
        "device": str(device),
        "golden_rows": len(golden_df),
        "num_labels": num_labels,
        "threshold": args.threshold,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "exact_match": exact_match,
        "true_labels_per_sample": float(
            true_labels.sum(axis=1).mean()
        ),
        "predicted_labels_per_sample": float(
            predictions.sum(axis=1).mean()
        ),
        "predicted_label_types": int(
            (predictions.sum(axis=0) > 0).sum()
        ),
    }

    summary_path = output_dir / "summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nGolden evaluation completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(summary_path)
    print(per_label_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
