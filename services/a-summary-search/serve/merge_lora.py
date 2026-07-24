"""Part A · Step 3(서빙) — LoRA 어댑터를 베이스에 병합해 독립 fp16 모델 저장.

서빙 양자화 스터디의 0단계. 학습 산출물은 '베이스 + LoRA 어댑터' 두 조각이라
서빙 엔진(vLLM)에 그대로 올리면 어댑터 오버헤드가 낀다. `merge_and_unload()` 로
LoRA 가중치를 베이스에 더해 **단일 fp16 체크포인트**로 만든다. 이 병합본이
  ① fp16 서빙의 대상이자
  ② AWQ 4bit 양자화(quantize_awq.py)의 입력
둘 다로 쓰인다 → fp16 vs 4bit 를 '같은 가중치'에서 출발시켜 공정 비교.

전략 메모: 학습은 fp16 베이스에 LoRA(QLoRA 아님) → 서빙에서 PTQ. 그래서 여기서
병합은 fp16 로 하고, 4bit 변환은 다음 단계(캘리브레이션 기반 AWQ)에서만 일어난다
(steam-part-a-step3 참고).

⚠️ torch/transformers/peft 필요 → **Lightning AI Studio 서빙 환경에서 실행.**
  uv sync --extra serve   # serve extra 에 peft 포함(병합용), 이후 quantize/bench 동일 환경

usage (Lightning):
    python serve/merge_lora.py \
        --adapter train/qwen2.5-3b-lora \
        --out serve/qwen2.5-3b-merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# eval/baseline_predict.py 의 MODEL_ID(베이스 id) 재사용 — 학습/예측과 짝 맞춤.
_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(_EVAL_DIR))
from baseline_predict import MODEL_ID  # noqa: E402

DEFAULT_ADAPTER = "train/qwen2.5-3b-lora"
DEFAULT_OUT = "serve/qwen2.5-3b-merged"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=MODEL_ID, help="베이스 모델 id")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA 어댑터 경로")
    ap.add_argument("--out", default=DEFAULT_OUT, help="병합본 저장 경로")
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # fp16 로 병합(AWQ 입력·T4 서빙 모두 fp16 이 자연스러움). bf16 로 학습했어도
    # 가중치 값은 동일하게 더해지므로 병합 dtype 은 서빙 타깃(fp16)에 맞춘다.
    print(f"[merge] 베이스 {args.base} (fp16) 로드 …")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float16, device_map="cpu")
    print(f"[merge] 어댑터 {args.adapter} 결합 …")
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()  # LoRA ΔW 를 베이스에 흡수 → 순수 fp16

    out = Path(args.out)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)

    # 병합본 가중치 용량(= fp16 서빙 footprint) 로깅 — 4bit 대비의 기준선.
    wbytes = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    print(f"[done] 병합본 저장 → {out}  (safetensors {wbytes / 1e9:.2f} GB, fp16)")
    print("  다음: quantize_awq.py 로 4bit 양자화 → bench_vllm.py 로 fp16 vs 4bit 실측")


if __name__ == "__main__":
    main()
