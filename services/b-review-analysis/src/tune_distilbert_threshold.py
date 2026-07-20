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
        "--output-dir",
        default=(
            "services/b-review-analysis/reports/"
            "distilbert_topic_baseline"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.01)

    return parser.parse_args()


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)

    micro_precision, micro_recall, micro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    exact_match = np.mean(
        np.all(predictions == labels, axis=1)
    )

    predicted_labels_per_sample = predictions.sum(axis=1).mean()
    true_labels_per_sample = labels.sum(axis=1).mean()

    return {
        "threshold": float(threshold),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "exact_match": float(exact_match),
        "predicted_labels_per_sample": float(
            predicted_labels_per_sample
        ),
        "true_labels_per_sample": float(
            true_labels_per_sample
        ),
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_df = pd.read_csv(args.valid_path)

    with open(args.label_mapping_path, encoding="utf-8") as file:
        label_mapping = json.load(file)

    label_names = label_mapping["label_names"]
    num_labels = label_mapping["num_labels"]

    labels = np.vstack(
        valid_df["labels"].apply(
            lambda value: np.asarray(
                ast.literal_eval(value),
                dtype=np.int32,
            )
        )
    )

    if labels.shape[1] != num_labels:
        raise ValueError(
            f"Expected {num_labels} labels, "
            f"but found {labels.shape[1]}"
        )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Validation rows:", len(valid_df))
    print("Number of labels:", num_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir
    )

    model.to(device)
    model.eval()

    texts = valid_df["input_text"].fillna("").astype(str).tolist()
    all_logits = []

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

            all_logits.append(
                outputs.logits.detach().cpu().numpy()
            )

    logits = np.concatenate(all_logits, axis=0)
    probabilities = sigmoid(logits)

    thresholds = np.arange(
        args.min_threshold,
        args.max_threshold + args.step / 2,
        args.step,
    )

    results = [
        calculate_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]

    result_df = pd.DataFrame(results)

    best_micro = result_df.loc[
        result_df["micro_f1"].idxmax()
    ].to_dict()

    best_macro = result_df.loc[
        result_df["macro_f1"].idxmax()
    ].to_dict()

    threshold_csv = output_dir / "threshold_sweep.csv"
    result_df.to_csv(
        threshold_csv,
        index=False,
        encoding="utf-8",
    )

    best_result = {
        "model_dir": args.model_dir,
        "validation_rows": len(valid_df),
        "num_labels": num_labels,
        "label_names": label_names,
        "best_micro_f1": best_micro,
        "best_macro_f1": best_macro,
    }

    best_json = output_dir / "best_threshold.json"

    with open(best_json, "w", encoding="utf-8") as file:
        json.dump(
            best_result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nBest threshold by Micro F1")
    print(json.dumps(best_micro, indent=2))

    print("\nBest threshold by Macro F1")
    print(json.dumps(best_macro, indent=2))

    print("\nSaved:")
    print(threshold_csv)
    print(best_json)


if __name__ == "__main__":
    main()
