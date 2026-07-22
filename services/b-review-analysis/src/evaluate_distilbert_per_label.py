#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

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
            "distilbert_topic_baseline"
        ),
    )
    parser.add_argument(
        "--valid-path",
        default=(
            "services/b-review-analysis/data/"
            "lora_topic_dataset_verified/valid.csv"
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
        "--output-path",
        default=(
            "services/b-review-analysis/reports/"
            "distilbert_topic_baseline/"
            "per_label_metrics_threshold_016.csv"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)

    return parser.parse_args()


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    args = parse_args()

    valid_df = pd.read_csv(args.valid_path)

    with open(args.label_mapping_path, encoding="utf-8") as file:
        mapping = json.load(file)

    label_names = mapping["label_names"]

    labels = np.vstack(
        valid_df["labels"].apply(
            lambda value: np.asarray(
                ast.literal_eval(value),
                dtype=np.int32,
            )
        )
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir
    )

    model.to(device)
    model.eval()

    texts = valid_df["input_text"].fillna("").astype(str).tolist()
    logits_list = []

    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start : start + args.batch_size]

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

            logits_list.append(
                outputs.logits.detach().cpu().numpy()
            )

    logits = np.concatenate(logits_list, axis=0)
    probabilities = sigmoid(logits)
    predictions = (probabilities >= args.threshold).astype(int)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average=None,
            zero_division=0,
        )
    )

    result_df = pd.DataFrame(
        {
            "label": label_names,
            "support": support.astype(int),
            "predicted_count": predictions.sum(axis=0).astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_probability": probabilities.mean(axis=0),
        }
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    supported = result_df[result_df["support"] > 0]

    print("=" * 90)
    print("Device:", device)
    print("Threshold:", args.threshold)
    print("Validation rows:", len(valid_df))
    print("True labels per review:", labels.sum(axis=1).mean())
    print(
        "Predicted labels per review:",
        predictions.sum(axis=1).mean(),
    )

    print("\nLowest F1 labels with validation support")
    print(
        supported.sort_values(
            ["f1", "support"],
            ascending=[True, False],
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nHighest F1 labels")
    print(
        supported.sort_values(
            ["f1", "support"],
            ascending=[False, False],
        )
        .head(15)
        .to_string(index=False)
    )

    print("\nLabels absent from Validation")
    print(
        result_df[result_df["support"] == 0][
            ["label", "support", "predicted_count"]
        ].to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
