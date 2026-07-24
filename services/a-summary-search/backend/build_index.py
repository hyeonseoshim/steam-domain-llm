"""RAG 파이프라인 Step4 — 150K 하이브리드 인덱스(BM25+dense+RRF) 구축·평가.

Step3 임베딩(corpus_emb.npy)과 동일 순서로 문서를 만들어 BM25(kiwi)와 dense(사전계산
벡터)를 RRF로 융합, 60질의로 150K 스케일 Recall@K·MRR 실측. 4158 대비 훨씬 어려운
현실 수치. hybrid_search 의 rrf/metrics 재사용.

usage:
    uv run --with sentence-transformers --with rank-bm25 --with kiwipiepy --with numpy \
        backend/build_index.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from embed_corpus import build_docs
from hybrid_search import metrics, rrf

RP = Path(__file__).resolve().parent
EMB = RP / "corpus_emb.npy"
EVAL = RP / "eval_queries.jsonl"
TOKCACHE = RP / ".bm25_tokens.pkl"


def main() -> None:
    from kiwipiepy import Kiwi  # noqa: PLC0415
    from rank_bm25 import BM25Okapi  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    appids, docs = build_docs()  # Step3와 동일 순서(임베딩 정렬 일치)
    pos = {a: i for i, a in enumerate(appids)}
    demb = np.load(EMB)
    assert demb.shape[0] == len(docs), f"임베딩({demb.shape[0]})과 문서({len(docs)}) 불일치"
    print(f"[index] 코퍼스 {len(docs):,} · 임베딩 {demb.shape}")

    evals = [json.loads(l) for l in EVAL.open(encoding="utf-8") if l.strip()]
    evals = [e for e in evals if e["appid"] in pos]
    print(f"[eval] 질의 {len(evals)}개")

    # BM25 (kiwi 토큰 캐시)
    kiwi = Kiwi()
    def tok(s: str) -> list[str]:
        return [t.form for t in kiwi.tokenize(s)]
    if TOKCACHE.exists():
        toks = pickle.loads(TOKCACHE.read_bytes()); print(f"[bm25] 토큰 캐시 로드")
    else:
        print("[bm25] kiwi 토큰화 중(150K, 수분)…"); t0 = time.time()
        toks = [tok(d) for d in docs]
        TOKCACHE.write_bytes(pickle.dumps(toks)); print(f"  {time.time()-t0:.0f}s")
    bm25 = BM25Okapi(toks)

    st = SentenceTransformer("BAAI/bge-m3")
    qembs = st.encode([e["query"] for e in evals], normalize_embeddings=True).astype(np.float32)

    agg = {m: [0, 0, 0, 0.0] for m in ("BM25", "dense", "hybrid")}
    for e, qe in zip(evals, qembs):
        gp = pos[e["appid"]]
        bm_rank = list(np.argsort(bm25.get_scores(tok(e["query"])))[::-1][:1000])
        dn_rank = list(np.argsort(qe @ demb.T)[::-1][:1000])
        hy_rank = rrf([bm_rank[:200], dn_rank[:200]])
        for name, ranks in (("BM25", bm_rank), ("dense", dn_rank), ("hybrid", hy_rank)):
            r1, r5, r10, rr = metrics(list(ranks), gp)
            a = agg[name]; a[0] += r1; a[1] += r5; a[2] += r10; a[3] += rr

    n = len(evals)
    print(f"\n{'검색기':<10}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}  (코퍼스 {len(docs):,})")
    print("-" * 50)
    for m in ("BM25", "dense", "hybrid"):
        r1, r5, r10, rr = agg[m]
        print(f"{m:<10}{r1/n*100:>7.0f}%{r5/n*100:>7.0f}%{r10/n*100:>7.0f}%{rr/n:>8.3f}")
    print("\n[성공기준] hybrid ≥ max(BM25,dense) 유지 = RRF 값어치가 150K 스케일에서도 성립.")


if __name__ == "__main__":
    main()
