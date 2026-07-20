#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_INPUT_PATH = (
    "data/processed/english_reviews_clean.csv"
)

DEFAULT_MODEL_DIR = (
    "services/b-review-analysis/models/"
    "distilbert_topic_lora/merged_model"
)

DEFAULT_OUTPUT_DIR = (
    "services/b-review-analysis/data/predictions/"
    "review_topic_predictions_parts"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batch topic inference over cleaned Steam reviews."
        )
    )

    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Cleaned Steam review CSV path",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Merged Hugging Face model directory",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for compressed prediction part files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.39,
        help="Common multi-label prediction threshold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="CSV rows processed per output part",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum tokenizer sequence length",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum rows for smoke testing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parts",
    )

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available.")
        return torch.device("mps")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_label_names(model: Any) -> list[str]:
    label_names = []

    for label_id in range(model.config.num_labels):
        label = model.config.id2label.get(label_id)

        if label is None:
            label = model.config.id2label.get(str(label_id))

        if label is None:
            raise ValueError(
                f"Missing id2label value for label ID {label_id}"
            )

        label_names.append(str(label))

    invalid_labels = [
        label
        for label in label_names
        if not (
            label.startswith("pos__")
            or label.startswith("neg__")
        )
    ]

    if invalid_labels:
        raise ValueError(
            "Unexpected signed label names: "
            f"{invalid_labels}"
        )

    return label_names


def predict_texts(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    label_names: list[str],
    device: torch.device,
    threshold: float,
    batch_size: int,
    max_length: int,
) -> tuple[list[str], list[str], list[str]]:
    all_positive_topics = []
    all_negative_topics = []
    all_selected_scores = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[
            batch_start:batch_start + batch_size
        ]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.sigmoid(logits).cpu()

        for row_probabilities in probabilities:
            positive_topics = []
            negative_topics = []
            selected_scores = {}

            for label, probability_tensor in zip(
                label_names,
                row_probabilities,
            ):
                probability = float(probability_tensor.item())

                if probability < threshold:
                    continue

                selected_scores[label] = round(
                    probability,
                    6,
                )

                if label.startswith("pos__"):
                    positive_topics.append(
                        label.replace("pos__", "", 1)
                    )

                elif label.startswith("neg__"):
                    negative_topics.append(
                        label.replace("neg__", "", 1)
                    )

            selected_scores = dict(
                sorted(
                    selected_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            all_positive_topics.append(
                json.dumps(
                    positive_topics,
                    ensure_ascii=False,
                )
            )
            all_negative_topics.append(
                json.dumps(
                    negative_topics,
                    ensure_ascii=False,
                )
            )
            all_selected_scores.append(
                json.dumps(
                    selected_scores,
                    ensure_ascii=False,
                )
            )

        del encoded
        del logits
        del probabilities

    return (
        all_positive_topics,
        all_negative_topics,
        all_selected_scores,
    )


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    label_names: list[str],
) -> None:
    config = {
        "input_path": str(Path(args.input_path)),
        "model_dir": str(Path(args.model_dir)),
        "output_format": "csv.gz parts",
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "max_length": args.max_length,
        "device": str(device),
        "limit": args.limit,
        "num_labels": len(label_names),
        "label_names": label_names,
        "model_version": "distilbert_topic_lora",
    }

    with (
        output_dir / "run_config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(args.device)

    print(f"Input: {input_path}")
    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Limit: {args.limit}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(model_dir)
    )

    model.to(device)
    model.eval()

    label_names = get_label_names(model)

    print(f"Labels: {len(label_names)}")

    write_run_config(
        output_dir=output_dir,
        args=args,
        device=device,
        label_names=label_names,
    )

    required_columns = {
        "recommendationid",
        "appid",
        "review_text_clean",
    }

    optional_columns = [
        "voted_up",
        "label",
        "weighted_vote_score",
        "votes_up",
    ]

    read_kwargs = {
        "chunksize": args.chunk_size,
        "low_memory": False,
    }

    if args.limit is not None:
        read_kwargs["nrows"] = args.limit

    total_processed = 0
    total_predicted_labels = 0
    started_at = time.time()

    reader = pd.read_csv(
        input_path,
        **read_kwargs,
    )

    for chunk_index, chunk in enumerate(reader):
        output_path = (
            output_dir
            / f"part-{chunk_index:06d}.csv.gz"
        )

        if output_path.exists() and not args.overwrite:
            print(
                f"[SKIP] chunk={chunk_index} "
                f"path={output_path}"
            )
            continue

        missing_columns = (
            required_columns - set(chunk.columns)
        )

        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        chunk = chunk.copy()

        chunk["review_text_clean"] = (
            chunk["review_text_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        valid_mask = (
            chunk["review_text_clean"].str.len() > 0
        )

        valid_chunk = chunk.loc[valid_mask].copy()

        if valid_chunk.empty:
            print(
                f"[EMPTY] chunk={chunk_index}"
            )
            continue

        texts = (
            valid_chunk["review_text_clean"]
            .tolist()
        )

        chunk_started_at = time.time()

        (
            positive_topics,
            negative_topics,
            selected_scores,
        ) = predict_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            label_names=label_names,
            device=device,
            threshold=args.threshold,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        result_columns = [
            "recommendationid",
            "appid",
            "review_text_clean",
        ]

        for column in optional_columns:
            if column in valid_chunk.columns:
                result_columns.append(column)

        result = valid_chunk[
            result_columns
        ].copy()

        result["positive_topics"] = positive_topics
        result["negative_topics"] = negative_topics
        result["selected_scores"] = selected_scores
        result["threshold"] = args.threshold
        result["model_version"] = (
            "distilbert_topic_lora"
        )

        selected_count = sum(
            len(json.loads(value))
            for value in selected_scores
        )

        temporary_path = Path(
            str(output_path) + ".tmp"
        )

        result.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8",
            compression="gzip",
        )

        temporary_path.replace(output_path)

        chunk_elapsed = (
            time.time() - chunk_started_at
        )

        total_processed += len(result)
        total_predicted_labels += selected_count

        average_labels = (
            selected_count / len(result)
        )

        print(
            f"[DONE] chunk={chunk_index} "
            f"rows={len(result):,} "
            f"labels={selected_count:,} "
            f"avg_labels={average_labels:.3f} "
            f"seconds={chunk_elapsed:.2f} "
            f"path={output_path.name}"
        )

        if device.type == "mps":
            torch.mps.empty_cache()

    elapsed = time.time() - started_at

    print("\nBatch inference completed")
    print(f"Processed rows: {total_processed:,}")
    print(
        "Predicted labels: "
        f"{total_predicted_labels:,}"
    )

    if total_processed:
        print(
            "Average labels per review: "
            f"{total_predicted_labels / total_processed:.4f}"
        )

    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
