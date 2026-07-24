"""FP8 양자화 품질 검증 — 병합모델을 bf16 vs FP8 로 gold_test(416) 전체 생성 후
ROUGE-1/2/L + BERTScore-F 채점. 목적: FP8(서빙단계 양자화)이 요약 품질을 얼마나 떨어뜨리나.

기준선: HF-LoRA(report_lora) ROUGE r1_f=.595 · BERTScore-F=.788.
비교: vLLM-merged-bf16(엔진/병합 효과) vs vLLM-merged-fp8(양자화 효과) — 같은 gold_test.

실행:  modal run modal_eval.py           # bf16·fp8 둘 다
채점은 GPU 컨테이너서(생성 vLLM + BERTScore cuda) 한번에. 결과는 콘솔+볼륨 리포트.
"""

from __future__ import annotations

import json
import pathlib

import modal

REPO = pathlib.Path(__file__).parent
HF_CACHE = "/root/.cache/huggingface"
MERGED = "/data/merged-qwen2.5-3b"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.10.2", "transformers<5", "kiwipiepy", "bert-score",
    )
    .add_local_dir(str(REPO / "eval"), remote_path="/root/eval",
                   ignore=["*.jsonl", "__pycache__", "*.json"])
    .add_local_file(str(REPO / "data" / "references" / "gold_test.jsonl"),
                    "/root/gold_test.jsonl")
)

data_vol = modal.Volume.from_name("steam-part-a-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("steam-hf-cache", create_if_missing=True)
app = modal.App("steam-part-a-eval")


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, HF_CACHE: hf_vol},
              timeout=1800)
def eval_variant(quant: str) -> dict:
    """quant='' (bf16) 또는 'fp8'. gold_test 전체 생성 → ROUGE+BERTScore."""
    import sys
    import time

    sys.path.insert(0, "/root/eval")
    from baseline_predict import build_messages, parse_summary
    from metrics import FIELDS, concat, rouge_l, rouge_n
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gold = [json.loads(l) for l in open("/root/gold_test.jsonl", encoding="utf-8") if l.strip()]
    tag = quant or "bf16"
    print(f"[eval:{tag}] gold {len(gold)}건 로드")

    tok = AutoTokenizer.from_pretrained(MERGED)
    kw = dict(gpu_memory_utilization=0.5, max_model_len=8192, dtype="bfloat16",
              enforce_eager=False)              # 0.5 로 제한 → 나머지 GPU 는 BERTScore(cuda) 자리
    if quant:
        kw["quantization"] = quant
    llm = LLM(model=MERGED, **kw)

    prompts = [tok.apply_chat_template(
        build_messages({"name": g.get("name", ""), "genres": g.get("genres", []),
                        "input": g.get("input", "")}, []),
        tokenize=False, add_generation_prompt=True) for g in gold]
    params = SamplingParams(temperature=0.0, max_tokens=320, stop=["}"])
    t0 = time.perf_counter()
    outs = llm.generate(prompts, params)
    gen_s = time.perf_counter() - t0
    print(f"[eval:{tag}] 생성 {len(outs)}건 {gen_s:.1f}s ({len(outs)/gen_s:.1f} req/s 배치)")

    # 파싱 + ROUGE (전체 concat)
    r1, r2, rl = [], [], []
    hyps, refs = [], []
    per_field = {k: {"h": [], "r": []} for k in FIELDS}
    missing = 0
    for g, o in zip(gold, outs):
        p_sum = parse_summary(o.outputs[0].text + "}")
        g_sum = g.get("summary")
        if not isinstance(p_sum, dict):
            missing += 1
            p_sum = {}
        pc, gc = concat(p_sum), concat(g_sum)
        r1.append(rouge_n(pc, gc, 1)["f"])
        r2.append(rouge_n(pc, gc, 2)["f"])
        rl.append(rouge_l(pc, gc)["f"])
        hyps.append(pc or " ")
        refs.append(gc or " ")
        if isinstance(g_sum, dict):
            for k in FIELDS:
                per_field[k]["h"].append(str(p_sum.get(k, "")) or " ")
                per_field[k]["r"].append(str(g_sum.get(k, "")) or " ")

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    # BERTScore (전체 + 필드별), 한국어 다국어 BERT
    from bert_score import score as bscore
    _, _, F = bscore(hyps, refs, lang="ko", device="cuda", batch_size=64)
    bert_overall = round(float(F.mean()), 4)
    bert_field = {}
    for k in FIELDS:
        _, _, Fk = bscore(per_field[k]["h"], per_field[k]["r"], lang="ko",
                          device="cuda", batch_size=64)
        bert_field[k] = round(float(Fk.mean()), 4)

    rep = {
        "variant": tag, "quant": quant or "none", "scored": len(gold),
        "missing_pred": missing, "gen_seconds": round(gen_s, 1),
        "rouge_overall": {"r1_f": mean(r1), "r2_f": mean(r2), "rL_f": mean(rl)},
        "bertscore_f_overall": bert_overall,
        "bertscore_f_per_field": bert_field,
    }
    pathlib.Path(f"/data/report_vllm_{tag}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    data_vol.commit()
    print(f"[eval:{tag}] {json.dumps(rep, ensure_ascii=False)}")
    return rep


@app.local_entrypoint()
def main() -> None:
    print("=== bf16 (병합, 무양자화) ===")
    bf16 = eval_variant.remote("")
    print("=== fp8 (온라인 양자화) ===")
    fp8 = eval_variant.remote("fp8")
    print("\n===== FP8 품질 열화 비교 (기준 HF-LoRA: r1_f=.595 BERT-F=.788) =====")
    for r in (bf16, fp8):
        ro = r["rouge_overall"]
        print(f"  {r['variant']:5s}  R1={ro['r1_f']:.4f}  RL={ro['rL_f']:.4f}  "
              f"BERT-F={r['bertscore_f_overall']:.4f}  (누락 {r['missing_pred']})")
    dr1 = fp8["rouge_overall"]["r1_f"] - bf16["rouge_overall"]["r1_f"]
    dbf = fp8["bertscore_f_overall"] - bf16["bertscore_f_overall"]
    print(f"  FP8 열화:  ΔR1={dr1:+.4f}  ΔBERT-F={dbf:+.4f}")
