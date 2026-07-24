"""Part A · Step 3 — 베이스라인/파인튜닝 예측을 gold 대비 채점.

입력:
  --pred  예측 jsonl: {"appid", "summary": {장르,핵심플레이,특징}}  (파싱 실패 시 summary=null)
  --gold  정답 jsonl: gold_test.jsonl (build_gold.py 산출)
출력:
  형식 준수율(전체 + 항목별), ROUGE-1/2/L(전체 concat + 필드별) 평균 → report json + 콘솔.

BERTScore 는 무거워 분리(bertscore.py, GPU 권장). 여기선 형식+ROUGE 만.

usage:
    python3 part_a/eval/run_eval.py --pred preds.jsonl --gold part_a/data/references/gold_test.jsonl
    # 자기검증: --pred 에 gold 를 넣으면 ROUGE f≈1.0, 형식준수≈100% 이어야 함
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from metrics import (  # 같은 폴더 모듈
    FIELDS,
    TOKENIZER,
    concat,
    format_checks,
    rouge_l,
    rouge_n,
)


def load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["appid"]] = r
    return out


def mean(xs: list[float]) -> float:
    return round(st.mean(xs), 4) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default="part_a/data/references/gold_test.jsonl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    preds = load_jsonl(Path(args.pred))
    gold = load_jsonl(Path(args.gold))

    ids = [aid for aid in gold if aid in preds]  # gold 기준, 예측 있는 것만 채점
    missing = len(gold) - len(ids)

    # ROUGE 누적 (전체 concat + 필드별)
    r1, r2, rl = [], [], []
    per_field = {k: {"r1": [], "r2": [], "rL": []} for k in FIELDS}
    # 형식 준수 누적
    fmt_keys = ("schema", "genre_noun", "no_bullet", "korean")
    fmt_pass = {k: 0 for k in fmt_keys}
    fmt_all = 0

    for aid in ids:
        p_sum = preds[aid].get("summary")
        g_sum = gold[aid].get("summary")
        # 전체 concat ROUGE
        pc, gc = concat(p_sum), concat(g_sum)
        r1.append(rouge_n(pc, gc, 1)["f"])
        r2.append(rouge_n(pc, gc, 2)["f"])
        rl.append(rouge_l(pc, gc)["f"])
        # 필드별 ROUGE (둘 다 dict 일 때만)
        if isinstance(p_sum, dict) and isinstance(g_sum, dict):
            for k in FIELDS:
                pv, gv = str(p_sum.get(k, "")), str(g_sum.get(k, ""))
                per_field[k]["r1"].append(rouge_n(pv, gv, 1)["f"])
                per_field[k]["r2"].append(rouge_n(pv, gv, 2)["f"])
                per_field[k]["rL"].append(rouge_l(pv, gv)["f"])
        # 형식 준수 (예측 기준)
        checks = format_checks(p_sum)
        for k in fmt_keys:
            fmt_pass[k] += int(checks[k])
        fmt_all += int(all(checks.values()))

    n = len(ids)
    report = {
        "tokenizer": TOKENIZER,
        "gold_total": len(gold),
        "scored": n,
        "missing_pred": missing,
        "format_compliance": {
            "overall": round(fmt_all / n, 4) if n else 0.0,
            **{k: round(fmt_pass[k] / n, 4) if n else 0.0 for k in fmt_keys},
        },
        "rouge_overall": {"r1_f": mean(r1), "r2_f": mean(r2), "rL_f": mean(rl)},
        "rouge_per_field": {
            k: {"r1_f": mean(v["r1"]), "r2_f": mean(v["r2"]), "rL_f": mean(v["rL"])}
            for k, v in per_field.items()
        },
    }

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fc = report["format_compliance"]
    ro = report["rouge_overall"]
    print(f"[eval] tokenizer={TOKENIZER}  채점 {n}/{len(gold)}건 (예측누락 {missing})")
    print(f"  형식준수 전체 {fc['overall']*100:.1f}%  "
          f"(schema {fc['schema']*100:.0f} / genre {fc['genre_noun']*100:.0f} / "
          f"bullet {fc['no_bullet']*100:.0f} / korean {fc['korean']*100:.0f})")
    print(f"  ROUGE(전체)  R1 {ro['r1_f']:.3f}  R2 {ro['r2_f']:.3f}  RL {ro['rL_f']:.3f}")
    for k, v in report["rouge_per_field"].items():
        print(f"  ROUGE[{k}] R1 {v['r1_f']:.3f}  R2 {v['r2_f']:.3f}  RL {v['rL_f']:.3f}")
    if args.out:
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
