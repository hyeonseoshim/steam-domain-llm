"""Part A · Step 3 — BERTScore(의미 유사도) 채점. GPU 권장, CPU 가능.

run_eval.py(형식+ROUGE)의 짝. 같은 preds/gold 를 받아 BERTScore(P/R/F)를
전체 concat + 필드별로 계산한다. torch·transformers 필요해서 별도 파일로 분리.
파싱 실패(summary=null)·빈 예측은 F=0 으로 집계(형식 위반을 의미점수에도 반영).

BERTScore = pred/gold 를 BERT 임베딩해 토큰 코사인유사도로 매칭 → n-gram 겹침(ROUGE)이
못 잡는 '같은 뜻 다른 표현'을 잡는다. 한국어라 다국어 BERT(기본, lang=ko) 사용.

usage:
    uv sync --extra bertscore            # torch/transformers/bert-score 설치
    uv run part_a/eval/bertscore.py --pred part_a/eval/preds_lora.jsonl \
        --gold part_a/data/references/gold_test.jsonl --out part_a/eval/report_bert_lora.json
    # 0shot/2shot/lora 각각 돌려 before/after 완성 (⚠️ 셋 다 같은 model 로)
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from metrics import FIELDS, concat  # 같은 폴더 모듈


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
    ap.add_argument("--lang", default="ko", help="bert_score 기본 모델 선택용(ko=다국어 BERT)")
    ap.add_argument("--model", default="", help="model_type 직접 지정(예: klue/bert-base); "
                    "지정 시 --num-layers 도 필요")
    ap.add_argument("--num-layers", type=int, default=0, help="--model 지정 시 사용할 레이어")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from bert_score import score as bert_score  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(hyps: list[str], refs: list[str]) -> tuple[list, list, list]:
        kw = ({"model_type": args.model, "num_layers": args.num_layers or None}
              if args.model else {"lang": args.lang})
        P, R, F = bert_score(hyps, refs, device=device, batch_size=args.batch_size,
                             verbose=False, rescale_with_baseline=False, **kw)
        return [x.item() for x in P], [x.item() for x in R], [x.item() for x in F]

    preds = load_jsonl(Path(args.pred))
    gold = load_jsonl(Path(args.gold))
    ids = [a for a in gold if a in preds]  # gold 기준, 예측 있는 것만
    missing = len(gold) - len(ids)
    n = len(ids)

    # 전체 concat: 빈 예측(null/미준수)은 0 점, 나머지만 배치 계산
    oP, oR, oF = [0.0] * n, [0.0] * n, [0.0] * n
    hy, rf, pos = [], [], []
    for i, a in enumerate(ids):
        hc, gc = concat(preds[a].get("summary")), concat(gold[a].get("summary"))
        if hc.strip() and gc.strip():
            hy.append(hc); rf.append(gc); pos.append(i)
    if hy:
        P, R, F = run(hy, rf)
        for j, i in enumerate(pos):
            oP[i], oR[i], oF[i] = P[j], R[j], F[j]

    # 필드별: 둘 다 dict + 비어있지 않은 쌍만 (run_eval 필드별과 동일 취급)
    per_out = {}
    for k in FIELDS:
        hys, rfs = [], []
        for a in ids:
            ps, gs = preds[a].get("summary"), gold[a].get("summary")
            if isinstance(ps, dict) and isinstance(gs, dict):
                pv, gv = str(ps.get(k, "")).strip(), str(gs.get(k, "")).strip()
                if pv and gv:
                    hys.append(pv); rfs.append(gv)
        if hys:
            P, R, F = run(hys, rfs)
            per_out[k] = {"p": mean(P), "r": mean(R), "f": mean(F)}
        else:
            per_out[k] = {"p": 0.0, "r": 0.0, "f": 0.0}

    report = {
        "metric": "bertscore",
        "model": args.model or f"(lang={args.lang})",
        "device": device,
        "gold_total": len(gold),
        "scored": n,
        "missing_pred": missing,
        "bertscore_overall": {"p": mean(oP), "r": mean(oR), "f": mean(oF)},
        "bertscore_per_field": per_out,
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    bo = report["bertscore_overall"]
    print(f"[bertscore] device={device}  model={report['model']}  "
          f"채점 {n}/{len(gold)} (예측누락 {missing})")
    print(f"  BERTScore(전체)  P {bo['p']:.3f}  R {bo['r']:.3f}  F {bo['f']:.3f}")
    for k, v in per_out.items():
        print(f"  BERTScore[{k}] F {v['f']:.3f}")
    if args.out:
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
