"""Part A · Step 3 — LoRA(fp16/bf16) 파인튜닝 (Qwen2.5-3B-Instruct).

베이스라인(zero/few-shot)의 '후속'. gold_train 으로 3필드 요약을 지도학습해
LoRA 어댑터를 저장한다. 학습 프롬프트는 eval/baseline_predict.py 와 **완전히 동일**한
포맷(SYSTEM + user_prompt + assistant=summary JSON)을 재사용 → 파인튜닝 전/후를
동일 하브니스로 깨끗하게 비교하기 위함.

전략 메모(왜 QLoRA 아님): 3B fp16-LoRA 는 16GB 급 GPU에 들어가므로 4bit-frozen
베이스에 학습할 이유가 없다. 깨끗한 bf16 베이스에 LoRA → **서빙 단계에서** PTQ(AWQ/GPTQ)
로 양자화하는 게 품질상 유리(steam-part-a-step3 참고).

completion-only loss: 프롬프트 토큰은 -100 으로 마스킹하고 assistant(요약 JSON +
종료 토큰)에만 loss 를 건다 → 형식/스타일을 학습하되 입력 복창은 학습하지 않음.

⚠️ torch/transformers/peft/accelerate 필요 → **Lightning AI Studio(GPU)에서 실행.**
  uv sync --extra gpu

usage (Lightning):
    python part_a/train/finetune_lora.py                 # 기본 하이퍼파라미터
    python part_a/train/finetune_lora.py --epochs 3 --lr 1e-4 --limit 64   # 디버그

학습 후:
    python part_a/eval/baseline_predict.py --shots 0 \
        --adapter part_a/train/qwen2.5-3b-lora --out part_a/eval/preds_lora.jsonl
    python part_a/eval/run_eval.py --pred part_a/eval/preds_lora.jsonl \
        --gold part_a/data/references/gold_test.jsonl     # ⚠️ 반드시 kiwi 로 채점
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# eval/baseline_predict.py 의 프롬프트 포맷을 재사용(예측기와 짝 맞춤).
_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(_EVAL_DIR))
from baseline_predict import (  # noqa: E402  (경로 주입 후 import)
    MODEL_ID,
    build_messages,
    load_jsonl,
    user_prompt,  # noqa: F401  (build_messages 가 내부적으로 사용)
)

DATA_DIR = Path("part_a/data/references")
DEFAULT_OUT = "part_a/train/qwen2.5-3b-lora"


def encode(rec: dict, tok, max_len: int) -> dict:
    """한 레코드를 (input_ids, labels) 로 인코딩. 프롬프트는 마스킹, 요약만 학습."""
    msgs = build_messages(rec, shots=[])  # [system, user]
    target = json.dumps(rec["summary"], ensure_ascii=False)

    prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
    full_text = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": target}],
        tokenize=False, add_generation_prompt=False)

    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # 안전장치: 초과 시 좌측(입력 앞부분) 절단 → 꼬리의 정답(labels)은 보존.
    if len(full_ids) > max_len:
        cut = len(full_ids) - max_len
        full_ids, labels = full_ids[cut:], labels[cut:]
    return {"input_ids": full_ids, "labels": labels}


class JsonlDataset:
    """gold_*.jsonl → 토크나이즈된 예제 리스트(간단 torch Dataset)."""

    def __init__(self, path: Path, tok, max_len: int):
        self.rows = [encode(r, tok, max_len) for r in load_jsonl(path)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def make_collator(pad_id: int):
    import torch  # noqa: PLC0415

    def collate(batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:  # 우측 패딩
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--train", default=str(DATA_DIR / "gold_train.jsonl"))
    ap.add_argument("--val", default=str(DATA_DIR / "gold_val.jsonl"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)  # 유효 배치 16
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="디버그: 앞 N건만 학습")
    ap.add_argument("--gpu-hourly-usd", type=float, default=0.0,
                    help="학습 GPU 시간당 요금($). 넣으면 GPU-hour·비용 실측 로깅 "
                         "(예: Lightning L4≈0.80, A10G≈1.10, A100≈2.00; 클라우드 "
                         "GPU 활용을 증류 경제학 증거로 계측 — 7/21 수치 재료).")
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"[data] 토크나이즈 (max_len={args.max_len}) …")
    train_ds = JsonlDataset(Path(args.train), tok, args.max_len)
    val_ds = JsonlDataset(Path(args.val), tok, args.max_len)
    if args.limit:
        train_ds.rows = train_ds.rows[:args.limit]
        val_ds.rows = val_ds.rows[:max(8, args.limit // 8)]
    print(f"[data] train {len(train_ds)}  val {len(val_ds)}")

    # dtype 자동선택: T4(Turing)는 bf16 텐서코어 없음 → fp16 이 훨씬 빠름.
    # Ampere+(A10G/A100/L4)는 bf16 지원 → 안정성상 bf16.
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[gpu] {gpu_name}  bf16지원={use_bf16} → dtype={dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto")
    model.config.use_cache = False  # gradient checkpointing 과 충돌 방지
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=make_collator(tok.pad_token_id))

    n_gpu = max(1, torch.cuda.device_count())
    t0 = time.time()
    trainer.train()
    wall_s = time.time() - t0

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # --- 클라우드 GPU 계측 (증류 경제학 증거 / 7/21 수치 재료) ---
    gpu_hours = wall_s / 3600 * n_gpu
    print(f"[time] 학습 wall-clock {wall_s / 60:.1f} 분  "
          f"({n_gpu} GPU → {gpu_hours:.2f} GPU-hour)")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        name = torch.cuda.get_device_name(0)
        print(f"[gpu] {name}  peak VRAM {peak:.1f} GB")
    if args.gpu_hourly_usd > 0:
        cost = gpu_hours * args.gpu_hourly_usd
        print(f"[cost] 학습 1회 ≈ ${cost:.2f}  "
              f"(GPU {gpu_hours:.2f}h × ${args.gpu_hourly_usd:.2f}/h) "
              f"— 이후 서빙은 T4($0.19/h)로 24만 게임 반복 처리")
    else:
        print("[cost] --gpu-hourly-usd 를 주면 학습 비용($)까지 실측 로깅됨")
    print(f"[done] 어댑터 저장 → {args.out}")
    print("  다음: baseline_predict.py --adapter 로 예측 → run_eval.py (kiwi) 로 채점")


if __name__ == "__main__":
    main()
