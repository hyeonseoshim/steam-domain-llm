"""Part A · Step 3(서빙) — 병합 fp16 모델을 AWQ 4bit 로 양자화(PTQ).

서빙 양자화 스터디의 핵심 변환. AWQ(Activation-aware Weight Quantization)는
캘리브레이션 데이터로 각 채널의 활성치 규모를 측정해 '중요한' 가중치 채널을
덜 깎는 4bit weight-only 양자화 → bnb-NF4 보다 품질 손실이 작다.

왜 서빙에서(학습 아님) 양자화하나: QLoRA 의 4bit 는 '큰 모델 학습 메모리'용 이득인데
3B fp16-LoRA 는 16GB 급에 들어가므로 학습은 fp16 이 낫고, 4bit 의 실익(VRAM·처리량)은
**추론 시점**에 온다. 그래서 깨끗한 fp16 병합본을 만든 뒤(merge_lora.py) 여기서 PTQ.

캘리브레이션 = gold_train 의 실제 태스크 프롬프트(우리 도메인의 '긴 Steam 설명').
범용 위키텍스트가 아니라 **서빙 분포와 같은 입력**으로 캘리브레이션해야 우리 태스크에서
양자화 품질 손실이 최소화된다.

⚠️ autoawq + torch/transformers 필요 → **Lightning AI Studio(GPU)에서 실행.**
  uv sync --extra serve   (또는 pip install autoawq)
  ⚠️ T4(Turing, sm_75)는 AWQ 추론 지원 O, FP8 은 미지원 → 4bit 는 AWQ/GPTQ 로.

usage (Lightning):
    python serve/quantize_awq.py \
        --model serve/qwen2.5-3b-merged \
        --out serve/qwen2.5-3b-awq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# eval/baseline_predict.py 의 프롬프트 포맷 재사용 → 캘리브레이션을 서빙 분포와 일치.
_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(_EVAL_DIR))
from baseline_predict import build_messages, load_jsonl  # noqa: E402

import json  # noqa: E402

DEFAULT_MODEL = "serve/qwen2.5-3b-merged"
DEFAULT_OUT = "serve/qwen2.5-3b-awq"
CALIB = "data/references/gold_train.jsonl"


def build_calib(tok, path: Path, n: int) -> list[str]:
    """gold_train → 채팅템플릿 적용한 (프롬프트+정답) 문자열 n개.

    서빙 때 실제로 들어오는 입력 분포(긴 Steam 설명 + 3필드 정답)로 캘리브레이션.
    """
    rows = load_jsonl(path)[:n]
    texts = []
    for rec in rows:
        msgs = build_messages(rec, shots=[])
        target = json.dumps(rec["summary"], ensure_ascii=False)
        texts.append(tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": target}],
            tokenize=False, add_generation_prompt=False))
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="병합 fp16 모델 경로")
    ap.add_argument("--out", default=DEFAULT_OUT, help="AWQ 4bit 저장 경로")
    ap.add_argument("--calib", default=CALIB, help="캘리브레이션 데이터(gold_train)")
    ap.add_argument("--n-calib", type=int, default=128, help="캘리브레이션 샘플 수")
    ap.add_argument("--group-size", type=int, default=128, help="AWQ q_group_size")
    args = ap.parse_args()

    from awq import AutoAWQForCausalLM  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    quant_config = {"zero_point": True, "q_group_size": args.group_size,
                    "w_bit": 4, "version": "GEMM"}

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    calib = build_calib(tok, Path(args.calib), args.n_calib)
    print(f"[awq] 캘리브레이션 {len(calib)}건 (gold_train, 서빙 분포 일치)")

    print(f"[awq] 모델 로드 {args.model} → 4bit 양자화 (group={args.group_size}) …")
    model = AutoAWQForCausalLM.from_pretrained(args.model, safetensors=True)
    model.quantize(tok, quant_config=quant_config, calib_data=calib)

    out = Path(args.out)
    model.save_quantized(str(out))  # autoawq 가 save_dir[-1] 슬라이싱 → str 필요
    tok.save_pretrained(out)

    wbytes = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    print(f"[done] AWQ 4bit 저장 → {out}  (safetensors {wbytes / 1e9:.2f} GB)")
    print("  다음: bench_vllm.py --quant awq 로 fp16 대비 지연/VRAM/품질 실측")


if __name__ == "__main__":
    main()
