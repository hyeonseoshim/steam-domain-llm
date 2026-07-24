"""Part A · Step 3(서빙) — vLLM 로 fp16 vs AWQ 4bit 지연/VRAM/처리량/품질 실측.

서빙 양자화 스터디의 측정 도구(진짜 focus). 같은 병합 가중치에서 출발한 fp16 과
AWQ 4bit 를 **동일 하브니스**로 재서 4가지를 뽑는다:
  ① 지연  — prefill(TTFT 근사, max_tokens=1) / 전체생성 p50·p95, decode tok/s
  ② VRAM — 가중치 footprint(디스크) + KV 캐시 용량(num_gpu_blocks×block_size 토큰)
  ③ 처리량 — 배치 생성의 총 tok/s
  ④ 품질  — 예측 jsonl 저장 → run_eval.py(kiwi) + bertscore.py 로 재채점(양자화 손실)

가설(steam-part-a-presentation): 우리 태스크는 입력이 길어 **prefill 지배**
→ 4bit weight-only 의 '지연' 이득은 작고, 실익은 **VRAM**(가중치 절반↓ → 같은
util 에서 KV 블록↑ → 동시요청·긴컨텍스트 여유↑). ①②로 이 가설을 실측 확인/반박.

⚠️ vllm + torch 필요 → **Lightning AI Studio(GPU)에서 실행.** T4 는 AWQ 지원(FP8 X).
  uv sync --extra serve

usage (Lightning) — fp16 과 awq 를 각각:
    python serve/bench_vllm.py --model serve/qwen2.5-3b-merged \
        --quant none --pred serve/preds_fp16.jsonl --out serve/bench_fp16.json
    python serve/bench_vllm.py --model serve/qwen2.5-3b-awq \
        --quant awq  --pred serve/preds_awq.jsonl  --out serve/bench_awq.json
    # 이후 로컬에서 품질 재채점(양자화 delta):
    #   python eval/run_eval.py --pred serve/preds_awq.jsonl
    #   uv run eval/bertscore.py --pred serve/preds_awq.jsonl --out ...
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

# eval/baseline_predict.py 재사용 — 예측 프롬프트·파싱을 베이스라인/LoRA 와 동일하게.
_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(_EVAL_DIR))
from baseline_predict import (  # noqa: E402
    build_messages,
    load_jsonl,
    parse_summary,
)


def pct(xs: list[float], p: float) -> float:
    """정렬 리스트의 백분위(선형보간 없이 근사) — p 는 0~100."""
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return round(s[i], 4)


def gpu_used_mib() -> float:
    """현재 디바이스 0 의 사용 중 VRAM(MiB). pynvml 없으면 0."""
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
    except Exception:
        return 0.0


def kv_capacity(llm) -> dict:
    """vLLM 엔진에서 KV 캐시 용량(토큰) 추출 — 4bit 의 VRAM 이득이 드러나는 지표.

    같은 gpu_memory_utilization 에서 가중치가 작을수록(AWQ) KV 블록이 많아진다
    → '동시에 담을 수 있는 토큰 수'가 커짐 = prefill 지배 태스크의 실질 이득.
    내부 API 라 버전에 따라 위치가 달라 방어적으로 접근.
    """
    try:
        cc = llm.llm_engine.cache_config
        blocks = getattr(cc, "num_gpu_blocks", None)
        bs = getattr(cc, "block_size", None)
        if blocks and bs:
            return {"num_gpu_blocks": blocks, "block_size": bs,
                    "kv_tokens": blocks * bs}
    except Exception:
        pass
    return {"num_gpu_blocks": None, "block_size": None, "kv_tokens": None}


def weight_gb(model_dir: str) -> float:
    """모델 디렉터리의 safetensors 총 용량(GB) = 가중치 footprint."""
    p = Path(model_dir)
    if not p.is_dir():
        return 0.0
    return round(sum(f.stat().st_size for f in p.glob("*.safetensors")) / 1e9, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="병합 fp16 또는 AWQ 모델 경로")
    ap.add_argument("--quant", choices=["none", "awq", "gptq"], default="none")
    ap.add_argument("--gold", default="data/references/gold_test.jsonl")
    ap.add_argument("--pred", default="", help="품질 재채점용 예측 jsonl 저장 경로")
    ap.add_argument("--out", default="", help="벤치 리포트 json 저장 경로")
    ap.add_argument("--limit", type=int, default=0, help="품질/처리량 대상 건수(0=전체)")
    ap.add_argument("--latency-n", type=int, default=32,
                    help="지연 측정용 단일요청 표본 수(배치1로 순차 측정)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--gpu-hourly-usd", type=float, default=0.0,
                    help="서빙 GPU 시간당 요금($). 넣으면 처리량 기준 게임당 원가 추정 "
                         "(예: T4≈0.19 — 증류 경제학 증거, 7/21 수치 재료).")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    quant = None if args.quant == "none" else args.quant
    mem_before = gpu_used_mib()
    t_load = time.time()
    llm = LLM(model=args.model, quantization=quant, dtype="float16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True)
    load_s = time.time() - t_load
    mem_after = gpu_used_mib()
    kv = kv_capacity(llm)
    print(f"[load] {args.model} quant={args.quant}  {load_s:.1f}s  "
          f"VRAM {mem_after:.0f}MiB (+{mem_after - mem_before:.0f})  "
          f"KV토큰 {kv['kv_tokens']}")

    test = load_jsonl(Path(args.gold))
    if args.limit:
        test = test[:args.limit]
    convs = [build_messages(rec, shots=[]) for rec in test]

    # ── ① 지연: 단일요청 순차. prefill(max_tokens=1)=TTFT 근사, 전체생성=총지연 ──
    greedy1 = SamplingParams(temperature=0, max_tokens=1)
    greedyN = SamplingParams(temperature=0, max_tokens=args.max_new)
    prefill_ms, total_ms, dec_tok_s = [], [], []
    for conv in convs[:args.latency_n]:
        t = time.time()
        llm.chat([conv], greedy1, use_tqdm=False)
        prefill_ms.append((time.time() - t) * 1000)

        t = time.time()
        out = llm.chat([conv], greedyN, use_tqdm=False)[0]
        dt = time.time() - t
        total_ms.append(dt * 1000)
        n_out = len(out.outputs[0].token_ids)
        if dt > 0 and n_out > 1:
            dec_tok_s.append(n_out / dt)

    # ── ③ 처리량 + ④ 품질: 전체를 배치로 한 번에 생성 ──
    t = time.time()
    outs = llm.chat(convs, greedyN, use_tqdm=True)
    batch_s = time.time() - t
    total_out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    throughput = round(total_out_tok / batch_s, 1) if batch_s > 0 else 0.0

    n_ok = n_bad = 0
    if args.pred:
        pf = Path(args.pred).open("w", encoding="utf-8")
    for rec, o in zip(test, outs):
        raw = o.outputs[0].text
        summary = parse_summary(raw)
        n_ok += summary is not None
        n_bad += summary is None
        if args.pred:
            pf.write(json.dumps({"appid": rec["appid"], "summary": summary,
                                 "raw": raw}, ensure_ascii=False) + "\n")
    if args.pred:
        pf.close()

    report = {
        "model": args.model,
        "quant": args.quant,
        "weight_gb": weight_gb(args.model),
        "vram_used_mib": round(mem_after, 0),
        "kv_cache": kv,
        "load_s": round(load_s, 1),
        "latency_n": min(args.latency_n, len(convs)),
        "prefill_ms_p50": pct(prefill_ms, 50),   # ≈ TTFT (prefill 지배 여부의 핵심)
        "prefill_ms_p95": pct(prefill_ms, 95),
        "total_ms_p50": pct(total_ms, 50),
        "total_ms_p95": pct(total_ms, 95),
        "decode_tok_s_p50": pct(dec_tok_s, 50),
        "batch_throughput_tok_s": throughput,
        "batch_n": len(convs),
        "format_ok": n_ok,
        "format_bad": n_bad,
    }
    if args.gpu_hourly_usd > 0 and throughput > 0:
        # 게임당 원가 ≈ (출력토큰/게임) / (tok/s) × ($/시간 ÷ 3600)
        per_game_tok = total_out_tok / max(1, len(convs))
        cost_per_game = per_game_tok / throughput * (args.gpu_hourly_usd / 3600)
        report["est_cost_per_game_usd"] = round(cost_per_game, 6)
        report["est_cost_per_240k_usd"] = round(cost_per_game * 240_000, 2)

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[bench] {args.quant}  가중치 {report['weight_gb']}GB  "
          f"VRAM {report['vram_used_mib']:.0f}MiB  KV토큰 {kv['kv_tokens']}")
    print(f"  prefill(TTFT근사) p50 {report['prefill_ms_p50']:.0f}ms "
          f"p95 {report['prefill_ms_p95']:.0f}ms")
    print(f"  전체생성 p50 {report['total_ms_p50']:.0f}ms  "
          f"decode {report['decode_tok_s_p50']:.0f}tok/s  "
          f"배치처리량 {throughput:.0f}tok/s")
    print(f"  형식파싱 성공 {n_ok}/{len(test)}")
    if "est_cost_per_240k_usd" in report:
        print(f"  추정원가 게임당 ${report['est_cost_per_game_usd']:.6f} → "
              f"24만건 ${report['est_cost_per_240k_usd']:.2f} "
              f"(T4 ${args.gpu_hourly_usd}/h 기준)")
    if args.pred:
        print(f"[pred] {args.pred} → run_eval.py(kiwi)+bertscore.py 로 품질 재채점")
    if args.out:
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
