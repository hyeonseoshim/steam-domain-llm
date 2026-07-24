"""A 요약 단일요청 지연 — speculative decoding 격리 실험.

운영 앱은 변경하지 않는다. 운영과 같은 merged Qwen2.5-3B + online FP8을
기준선으로 두고, Qwen2.5-0.5B draft 모델이 실제 warm latency를 줄이는지만
측정한다. 운영의 vLLM 0.10.2는 draft-model 방식을 구현하지 않았으므로,
실험 이미지만 0.17.1로 올려 기준선과 speculative K=1/2/3/5를 비교한다.

실행:
    modal run modal_speculative.py

결과:
    콘솔 비교표 + /data/report_speculative_*.json
"""

from __future__ import annotations

import json
import pathlib
import statistics

import modal

REPO = pathlib.Path(__file__).parent
HF_CACHE = "/root/.cache/huggingface"
TARGET_MODEL = "/data/merged-qwen2.5-3b"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.17.1")
    .add_local_dir(
        str(REPO / "eval"),
        remote_path="/root/eval",
        ignore=["*.jsonl", "__pycache__", "*.json"],
    )
    .add_local_file(
        str(REPO / "data" / "references" / "gold_test.jsonl"),
        "/root/gold_test.jsonl",
    )
)

data_vol = modal.Volume.from_name("steam-part-a-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("steam-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("steam-vllm-cache", create_if_missing=True)
app = modal.App("steam-part-a-speculative-bench")


def _pct(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percentile / 100) * (len(ordered) - 1))
    return round(ordered[index], 1)


def _representative_sample(rows: list[dict], count: int) -> list[dict]:
    """입력 길이 전 구간을 같은 비율로 포함하는 결정적 표본."""
    ordered = sorted(rows, key=lambda row: len(str(row.get("input", ""))))
    if count >= len(ordered):
        return ordered
    return [
        ordered[round(i * (len(ordered) - 1) / (count - 1))]
        for i in range(count)
    ]


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    volumes={"/data": data_vol, HF_CACHE: hf_vol, "/root/.cache/vllm": vllm_cache_vol},
    timeout=1200,
)
def benchmark_variant(tag: str, speculative_tokens: int) -> dict:
    import sys
    import time

    sys.path.insert(0, "/root/eval")
    from baseline_predict import build_messages, parse_summary
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [
        json.loads(line)
        for line in open("/root/gold_test.jsonl", encoding="utf-8")
        if line.strip()
    ]
    sample = _representative_sample(rows, 24)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    prompts = [
        tok.apply_chat_template(
            build_messages(
                {
                    "name": row.get("name", ""),
                    "genres": row.get("genres", []),
                    "input": row.get("input", ""),
                },
                [],
            ),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in sample
    ]

    enforce_eager = tag.endswith("_eager")
    limited_cudagraphs = tag.endswith("_cg4")
    llm_args = {
        "model": TARGET_MODEL,
        "quantization": "fp8",
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.5,
        "max_model_len": 8192,
        "enforce_eager": enforce_eager,
        "disable_log_stats": False,
    }
    if speculative_tokens:
        llm_args["speculative_config"] = {
            "model": DRAFT_MODEL,
            "method": "draft_model",
            "num_speculative_tokens": speculative_tokens,
            "draft_tensor_parallel_size": 1,
            "max_model_len": 8192,
        }
    if limited_cudagraphs:
        # 운영은 생성 락으로 batch=1만 허용한다. K=1 draft 검증 여유까지 작은 shape만 캡처한다.
        llm_args["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4]}

    print(
        f"[spec:{tag}] vllm=0.17.1 K={speculative_tokens} eager={enforce_eager} "
        f"cg={llm_args.get('compilation_config', 'default')} "
        f"target={TARGET_MODEL} draft={DRAFT_MODEL if speculative_tokens else '-'}"
    )
    load_started = time.perf_counter()
    llm = LLM(**llm_args)
    load_s = time.perf_counter() - load_started
    params = SamplingParams(temperature=0.0, max_tokens=320, stop=["}"])

    # 최초 shape별 CUDA graph/커널 준비 시간은 warm latency에서 제외한다.
    for prompt in (prompts[0], prompts[len(prompts) // 2], prompts[-1]):
        llm.generate([prompt], params, use_tqdm=False)

    latencies_ms: list[float] = []
    output_tokens: list[int] = []
    raw_outputs: list[str] = []
    format_ok = 0
    for index, prompt in enumerate(prompts):
        started = time.perf_counter()
        result = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)
        output_tokens.append(len(result.token_ids))
        raw_outputs.append(result.text)
        format_ok += parse_summary(result.text + "}") is not None
        print(
            f"[spec:{tag}] {index + 1:02d}/{len(prompts)} "
            f"{elapsed_ms:.0f}ms out={len(result.token_ids)}tok"
        )

    total_tokens = sum(output_tokens)
    total_s = sum(latencies_ms) / 1000
    report = {
        "tag": tag,
        "vllm": "0.17.1",
        "target": TARGET_MODEL,
        "target_quantization": "fp8",
        "draft": DRAFT_MODEL if speculative_tokens else None,
        "speculative_tokens": speculative_tokens,
        "enforce_eager": enforce_eager,
        "limited_cudagraphs": limited_cudagraphs,
        "sample_n": len(prompts),
        "warmup_n": 3,
        "load_s": round(load_s, 1),
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 1),
            "p50": _pct(latencies_ms, 50),
            "p95": _pct(latencies_ms, 95),
            "min": round(min(latencies_ms), 1),
            "max": round(max(latencies_ms), 1),
        },
        "output_tokens": {
            "total": total_tokens,
            "mean": round(statistics.mean(output_tokens), 1),
            "p50": _pct([float(n) for n in output_tokens], 50),
            "p95": _pct([float(n) for n in output_tokens], 95),
        },
        "effective_output_tok_s": round(total_tokens / total_s, 1),
        "format_ok": format_ok,
        "format_bad": len(prompts) - format_ok,
        # 엔진 실행 경로에 따른 수치적 출력 변화도 관찰할 수 있도록 원문을 보존한다.
        "outputs": raw_outputs,
    }
    pathlib.Path(f"/data/report_speculative_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    data_vol.commit()
    print(
        f"[spec:{tag}] p50={report['latency_ms']['p50']:.0f}ms "
        f"p95={report['latency_ms']['p95']:.0f}ms "
        f"tok/s={report['effective_output_tok_s']:.1f} "
        f"format={format_ok}/{len(prompts)}"
    )
    return report


@app.local_entrypoint()
def main() -> None:
    variants = [
        ("v017_base", 0),
        ("draft_k1", 1),
        ("draft_k2", 2),
        ("draft_k3", 3),
        ("draft_k5", 5),
    ]
    calls = [benchmark_variant.spawn(*variant) for variant in variants]
    reports = []
    for variant, call in zip(variants, calls):
        try:
            reports.append(call.get())
        except Exception as exc:
            print(f"[spec:{variant[0]}] ERROR: {exc!r}")

    base = next(report for report in reports if report["tag"] == "v017_base")
    base_p50 = base["latency_ms"]["p50"]
    base_outputs = base["outputs"]
    print("\n===== speculative decoding · warm 단일요청 비교 =====")
    print("variant       K   p50(ms)  p95(ms)  vs-base  tok/s  exact  format")
    for report in reports:
        p50 = report["latency_ms"]["p50"]
        delta = (base_p50 - p50) / base_p50 * 100 if base_p50 else 0.0
        exact = sum(
            actual == expected
            for actual, expected in zip(report["outputs"], base_outputs)
        )
        print(
            f"{report['tag']:11s}  {report['speculative_tokens']:>2d}  {p50:8.0f}  "
            f"{report['latency_ms']['p95']:8.0f}  {delta:+6.1f}%  "
            f"{report['effective_output_tok_s']:5.1f}  "
            f"{exact:>2d}/{report['sample_n']:<2d}  "
            f"{report['format_ok']:>2d}/{report['sample_n']:<2d}"
        )

    candidates = [
        report
        for report in reports
        if report["speculative_tokens"] and report["format_bad"] == 0
    ]
    best = min(candidates, key=lambda report: report["latency_ms"]["p50"])
    improvement = (base_p50 - best["latency_ms"]["p50"]) / base_p50 * 100
    verdict = "채택 후보" if improvement >= 15 else "미채택"
    print(
        f"\n판정: {best['tag']} p50 개선 {improvement:+.1f}% → {verdict} "
        "(채택선: p50 15% 이상 개선 + 형식 오류 0건)"
    )
