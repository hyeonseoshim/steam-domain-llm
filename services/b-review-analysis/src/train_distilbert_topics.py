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
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


class SteamTopicDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
        num_labels: int,
    ) -> None:
        self.texts = dataframe["input_text"].fillna("").astype(str).tolist()

        self.labels = []
        for value in dataframe["labels"]:
            parsed = ast.literal_eval(value) if isinstance(value, str) else value
            label_vector = np.asarray(parsed, dtype=np.float32)

            if len(label_vector) != num_labels:
                raise ValueError(
                    f"Expected {num_labels} labels, got {len(label_vector)}"
                )

            self.labels.append(label_vector)

        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )

        return item


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = sigmoid(logits)
    predictions = (probabilities >= 0.5).astype(int)
    labels = labels.astype(int)

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

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "exact_match": float(exact_match),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        default=(
            "services/b-review-analysis/data/"
            "lora_topic_dataset_verified/train.csv"
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
        "--model-name",
        default="distilbert-base-uncased",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "services/b-review-analysis/models/"
            "distilbert_topic_baseline"
        ),
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    valid_df = pd.read_csv(args.valid_path)

    if args.max_train_samples is not None:
        train_df = train_df.head(args.max_train_samples).copy()

    if args.max_valid_samples is not None:
        valid_df = valid_df.head(args.max_valid_samples).copy()

    with open(args.label_mapping_path, encoding="utf-8") as file:
        label_mapping = json.load(file)

    label_names = label_mapping["label_names"]
    num_labels = label_mapping["num_labels"]

    if len(label_names) != num_labels:
        raise ValueError(
            "label_names length and num_labels do not match"
        )

    id2label = {
        index: label
        for index, label in enumerate(label_names)
    }
    label2id = {
        label: index
        for index, label in enumerate(label_names)
    }

    if torch.backends.mps.is_available():
        device_name = "MPS"
    elif torch.cuda.is_available():
        device_name = "CUDA"
    else:
        device_name = "CPU"

    print("=" * 80)
    print("Model:", args.model_name)
    print("Device:", device_name)
    print("Train rows:", len(train_df))
    print("Validation rows:", len(valid_df))
    print("Number of labels:", num_labels)
    print("Max length:", args.max_length)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    train_dataset = SteamTopicDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_labels=num_labels,
    )

    valid_dataset = SteamTopicDataset(
        dataframe=valid_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_labels=num_labels,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    evaluation_metrics = trainer.evaluate()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    train_metrics = {
        key: float(value)
        for key, value in train_result.metrics.items()
    }

    eval_metrics = {
        key: float(value)
        for key, value in evaluation_metrics.items()
    }

    result = {
        "model_name": args.model_name,
        "device": device_name,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "num_labels": num_labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "threshold": args.threshold,
        "train_metrics": train_metrics,
        "evaluation_metrics": eval_metrics,
    }

    result_path = output_dir / "training_metrics.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTraining completed")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nMetrics saved to: {result_path}")


if __name__ == "__main__":
    main()
