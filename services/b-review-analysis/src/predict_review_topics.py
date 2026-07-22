#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict signed topics from a Steam review."
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Steam review text",
    )
    parser.add_argument(
        "--model-dir",
        default=(
            "services/b-review-analysis/models/"
            "distilbert_topic_lora/merged_model"
        ),
    )
    parser.add_argument(
        "--label-mapping-path",
        default=(
            "services/b-review-analysis/data/"
            "lora_topic_dataset_verified/label_mapping.json"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.39)
    parser.add_argument("--max-length", type=int, default=256)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mapping_path = Path(args.label_mapping_path)

    with mapping_path.open(encoding="utf-8") as f:
        mapping = json.load(f)

    label_names = mapping["label_names"]

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

    encoded = tokenizer(
        args.text,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probabilities = torch.sigmoid(logits).cpu().tolist()

    selected = []

    for label, probability in zip(label_names, probabilities):
        if probability >= args.threshold:
            selected.append(
                {
                    "label": label,
                    "probability": round(probability, 6),
                }
            )

    selected.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    positive_topics = [
        item["label"].replace("pos__", "", 1)
        for item in selected
        if item["label"].startswith("pos__")
    ]

    negative_topics = [
        item["label"].replace("neg__", "", 1)
        for item in selected
        if item["label"].startswith("neg__")
    ]

    result = {
        "review_text": args.text,
        "threshold": args.threshold,
        "positive_topics": positive_topics,
        "negative_topics": negative_topics,
        "predictions": selected,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
