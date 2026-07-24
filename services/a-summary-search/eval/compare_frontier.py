"""Part A · Step 3 — 프론티어 baseline arm 채점·비교.

frontier_predict 로 만든 프론티어 예측(gold_test 앞 N건)과 우리 LoRA/0-shot 예측을
**같은 appid 집합**으로 좁혀 동일 하브니스(kiwi ROUGE + 형식준수 + BERTScore)로 채점,
한 표로 비교한다. 목적 = "품질 동급이면 self-host가 N배 싸다"의 '동급' 전제를 실측.

동일 비교 규칙: 프론티어는 앞 N건만 예측했으므로 그 **교집합 appid**로 전 소스를 채점
(LoRA/0-shot은 416건 있지만 같은 N건으로 잘라서 비교) → 사과-사과.

usage:
    uv run --extra eval --extra bertscore part_a/eval/compare_frontier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from metrics import FIELDS, TOKENIZER, concat, is_compliant, rouge_l, rouge_n  # noqa: E402

GOLD = _HERE.parent / "data/references/gold_test.jsonl"

# 이름 → 예측 jsonl. 위 4개=프론티어, 아래 2개=우리 모델(대조 기준).
SOURCES = {
    "Gemini Flash-Lite": _HERE / "preds_frontier_flashlite.jsonl",
    "Gemini Flash": _HERE / "preds_frontier_flash.jsonl",
    "GPT-5 mini": _HERE / "preds_frontier_gpt5mini.jsonl",
    "Claude Haiku 4.5": _HERE / "preds_frontier_haiku.jsonl",
    "LoRA (3B FT, 우리)": _HERE / "preds_lora.jsonl",
    "0-shot (3B base)": _HERE / "preds_0shot.jsonl",
}


def load(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["appid"]] = r
    return out


def main() -> None:
    gold = load(GOLD)
    loaded = {name: load(p) for name, p in SOURCES.items()}

    # 공통 appid = 프론티어 파일들(존재하는 것)의 교집합 ∩ gold
    frontier_files = [loaded[n] for n in
                      ("Gemini Flash-Lite", "Gemini Flash", "GPT-5 mini", "Claude Haiku 4.5")
                      if loaded.get(n)]
    if not frontier_files:
        print("프론티어 예측 파일이 아직 없음."); return
    common = set(gold)
    for d in frontier_files:
        common &= set(d)
    common = sorted(common)
    print(f"[compare] tokenizer={TOKENIZER} · 공통 appid {len(common)}건으로 채점\n")

    # BERTScore 준비(선택). concat 문자열 배치 채점.
    bert = None
    try:
        from bert_score import score as bert_score_fn  # noqa: PLC0415
        bert = bert_score_fn
    except Exception:
        print("  (bert-score 없음 → 의미점수 생략. `--extra bertscore`로 켜기)\n")

    rows = []
    for name, preds in loaded.items():
        if not preds:
            continue
        recs = [(preds.get(a), gold[a]) for a in common]
        n_have = sum(1 for p, _ in recs if p is not None)
        comp = r1 = rl = 0.0
        cand, ref = [], []
        for p, g in recs:
            summ = (p or {}).get("summary")
            comp += is_compliant(summ)
            cs, gs = concat(summ), concat(g["summary"])
            r1 += rouge_n(cs, gs, 1)["f"]
            rl += rouge_l(cs, gs)["f"]
            cand.append(cs or " "); ref.append(gs)
        n = len(common)
        bf = None
        if bert is not None:
            _, _, F = bert(cand, ref, lang="ko", verbose=False)
            bf = float(F.mean())
        rows.append((name, n_have / n, comp / n, r1 / n, rl / n, bf))

    hdr = f"{'모델':<20}{'파싱':>7}{'형식준수':>9}{'ROUGE-1':>9}{'ROUGE-L':>9}"
    hdr += f"{'BERT-F':>9}" if bert else ""
    print(hdr); print("-" * len(hdr))
    for name, parse, comp, r1, rl, bf in rows:
        line = f"{name:<20}{parse*100:>6.0f}%{comp*100:>8.0f}%{r1:>9.3f}{rl:>9.3f}"
        line += f"{bf:>9.3f}" if bf is not None else ""
        print(line)

    print("\n[읽는 법] 프론티어(위 4) vs 우리 LoRA/0-shot(아래 2)을 같은 게임으로 비교.")
    print("  가설: 의미(BERT-F)는 프론티어 대등, 형식준수·ROUGE(고정 스키마·중립 스타일)는")
    print("        LoRA가 이김 = DB 적재용 '결정성'이 파인튜닝의 값어치.")


if __name__ == "__main__":
    main()
