"""RAG 파일럿 평가 — Gemini(싼) 요약 vs gold(Opus) 요약, 검색 품질 대결.

같은 4158 코퍼스·같은 60질의에서, 60개 평가 대상 문서만 (a)gold 요약 (b)Gemini 요약
으로 바꿔 dense 검색 Recall@K·MRR 을 비교한다. 나머지 distractor 문서는 양쪽 다 gold
요약으로 동일 → 순수하게 "타깃 요약의 품질 차이"만 측정.

usage:
    uv run --with sentence-transformers --with numpy backend/pilot_eval.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
EVAL = Path(__file__).resolve().parent / "eval_queries.jsonl"
PILOT = Path(__file__).resolve().parent / "pilot_cheap_summ.jsonl"
CACHE = Path(__file__).resolve().parent / ".emb_cache"


def doc(name: str, s: dict) -> str:
    return f"{name} 장르 {s.get('장르','')} {s.get('핵심플레이','')} {s.get('특징','')}".strip()


def main() -> None:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    # gold: appid 순서 + 요약 문서
    appids, gdocs, names = [], [], []
    for line in GOLD.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        appids.append(r["appid"]); names.append(r.get("name", ""))
        gdocs.append(doc(r.get("name", ""), r.get("summary") or {}))
    pos = {a: i for i, a in enumerate(appids)}

    evals = [json.loads(l) for l in EVAL.open(encoding="utf-8") if l.strip()]
    evals = [e for e in evals if e["appid"] in pos]
    gold_pos = [pos[e["appid"]] for e in evals]

    sources = {"Gemini·detailed(원문롱)": PILOT,
               "Gemini·short_desc(CSV)": Path(__file__).resolve().parent / "pilot_short_summ.jsonl",
               "GPT-4.1-nano(싼)": Path(__file__).resolve().parent / "pilot_gpt_summ.jsonl",
               "gpt-5-nano(최저·추론min)": Path(__file__).resolve().parent / "pilot_gpt5_summ.jsonl",
               "Haiku 4.5(중간)": Path(__file__).resolve().parent / "pilot_haiku_summ.jsonl"}
    sources = {k: v for k, v in sources.items() if v.exists()}
    print(f"[corpus] {len(appids)} · [질의] {len(evals)}")

    st = SentenceTransformer("BAAI/bge-m3")
    qembs = st.encode([e["query"] for e in evals], normalize_embeddings=True).astype(np.float32)

    # A: gold 요약 임베딩 (기존 캐시 재사용)
    key = hashlib.md5(("BAAI/bge-m3" + str(len(gdocs)) + gdocs[0] + gdocs[-1]).encode()).hexdigest()[:12]
    cf = CACHE / f"{key}.npy"
    matA = np.load(cf) if cf.exists() else st.encode(gdocs, normalize_embeddings=True,
                                                     batch_size=32).astype(np.float32)
    print(f"[A gold] {'캐시' if cf.exists() else '인코딩'} {matA.shape}")

    # 싼 모델별: 타깃 60행만 해당 요약 임베딩으로 교체한 코퍼스
    mats = {"gold(Opus, 정답지)": matA}
    for label, path in sources.items():
        pilot = {json.loads(l)["appid"]: json.loads(l) for l in path.open(encoding="utf-8") if l.strip()}
        mat = matA.copy()
        idx, docs = [], []
        for a, rec in pilot.items():
            if a in pos:
                idx.append(pos[a]); docs.append(doc(rec.get("name", ""), rec["summary"]))
        emb = st.encode(docs, normalize_embeddings=True, batch_size=16).astype(np.float32)
        for j, i in enumerate(idx):
            mat[i] = emb[j]
        mats[label] = mat
        print(f"[{label}] 타깃 {len(idx)}행 교체")

    def score(mat):
        sims = qembs @ mat.T
        order = np.argsort(-sims, axis=1)
        agg = [0, 0, 0, 0.0]
        for i, gp in enumerate(gold_pos):
            hit = np.where(order[i] == gp)[0]
            if len(hit):
                rank = int(hit[0]) + 1
                agg[0] += rank <= 1; agg[1] += rank <= 5; agg[2] += rank <= 10; agg[3] += 1/rank
        n = len(gold_pos)
        return agg[0]/n, agg[1]/n, agg[2]/n, agg[3]/n

    print(f"\n{'요약 출처':<24}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>8}")
    print("-" * 53)
    for name, mat in mats.items():
        r = score(mat)
        print(f"{name:<24}{r[0]*100:>6.0f}%{r[1]*100:>6.0f}%{r[2]*100:>6.0f}%{r[3]:>8.3f}")
    print("\n[판정] 싼 모델들이 gold 에 얼마나 근접하나 = 검색문서용 모델 선택 근거.")


if __name__ == "__main__":
    main()
