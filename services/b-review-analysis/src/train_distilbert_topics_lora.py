#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from train_distilbert_topics_weighted import (
    SteamTopicDataset,
    WeightedMultiLabelTrainer,
    build_compute_metrics,
    calculate_pos_weights,
)


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
            "distilbert_topic_lora"
        ),
    )
    parser.add_argument(
        "--report-dir",
        default=(
            "services/b-review-analysis/reports/"
            "distilbert_topic_lora"
        ),
    )

    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--max-pos-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_dir)
    checkpoint_dir = output_root / "checkpoints"
    adapter_dir = output_root / "adapter"
    merged_dir = output_root / "merged_model"
    report_dir = Path(args.report_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    valid_df = pd.read_csv(args.valid_path)

    with open(args.label_mapping_path, encoding="utf-8") as file:
        label_mapping = json.load(file)

    label_names = label_mapping["label_names"]
    num_labels = label_mapping["num_labels"]

    train_label_matrix = np.vstack(
        train_df["labels"].apply(
            lambda value: np.asarray(
                ast.literal_eval(value),
                dtype=np.float32,
            )
        )
    )

    pos_weights = calculate_pos_weights(
        label_matrix=train_label_matrix,
        power=args.pos_weight_power,
        max_weight=args.max_pos_weight,
    )

    positive_counts = train_label_matrix.sum(axis=0).astype(int)

    weight_df = pd.DataFrame(
        {
            "label": label_names,
            "positive_count": positive_counts,
            "negative_count": len(train_df) - positive_counts,
            "pos_weight": pos_weights,
        }
    )

    weight_df.to_csv(
        report_dir / "pos_weights.csv",
        index=False,
        encoding="utf-8",
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_lin", "v_lin"],
        modules_to_save=["pre_classifier", "classifier"],
    )

    model = get_peft_model(
        base_model,
        lora_config,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_ratio = (
        trainable_parameters / total_parameters * 100
    )

    print("=" * 80)
    print("Experiment: Weighted DistilBERT LoRA")
    print("Device:", device_name)
    print("Train rows:", len(train_df))
    print("Validation rows:", len(valid_df))
    print("Number of labels:", num_labels)
    print("LoRA rank:", args.lora_r)
    print("LoRA alpha:", args.lora_alpha)
    print("LoRA dropout:", args.lora_dropout)
    print("Trainable parameters:", trainable_parameters)
    print("Total parameters:", total_parameters)
    print(f"Trainable ratio: {trainable_ratio:.4f}%")
    print("=" * 80)

    model.print_trainable_parameters()

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
        output_dir=str(checkpoint_dir),
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = WeightedMultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(
            args.threshold
        ),
        pos_weight=torch.tensor(
            pos_weights,
            dtype=torch.float32,
        ),
    )

    train_result = trainer.train()
    evaluation_metrics = trainer.evaluate()

    # LoRA adapter만 별도로 저장
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # 기존 평가 스크립트에서 바로 불러올 수 있도록
    # base model과 LoRA adapter를 합친 standalone 모델 저장
    merged_model = trainer.model.merge_and_unload(
        safe_merge=True
    )

    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    result = {
        "model_name": args.model_name,
        "experiment": "weighted_lora",
        "device": device_name,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "num_labels": num_labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "evaluation_threshold": args.threshold,
        "pos_weight_power": args.pos_weight_power,
        "max_pos_weight": args.max_pos_weight,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_ratio_percent": trainable_ratio,
        "train_metrics": {
            key: float(value)
            for key, value in train_result.metrics.items()
        },
        "evaluation_metrics": {
            key: float(value)
            for key, value in evaluation_metrics.items()
        },
    }

    metrics_path = output_root / "training_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTraining completed")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nAdapter saved to:", adapter_dir)
    print("Merged model saved to:", merged_dir)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
