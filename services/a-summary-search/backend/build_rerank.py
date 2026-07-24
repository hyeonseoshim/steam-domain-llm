"""RAG 파이프라인 Step5 — 하이브리드 top-K에 bge-reranker-v2-m3 재정렬.

Step4 실측: hybrid 가 top-10 엔 정답 잘 넣지만(R@10 43%) 그 안 순서가 나쁨(R@1 15%,
MRR .251 < BM25 .288). 크로스인코더 리랭커로 hybrid 후보 top-K 를 (질의,문서) 쌍
재점수→재정렬해 R@1/MRR 회복폭을 실측. BM25 토큰 캐시(.bm25_tokens.pkl)·dense
벡터(corpus_emb.npy)·문서순서(embed_corpus.build_docs) 재사용 → Step4 와 정렬 일치.

usage (로컬 CPU 또는 Lightning GPU):
    uv run --with sentence-transformers --with rank-bm25 --with kiwipiepy --with numpy \
        backend/build_rerank.py --topk 100
    # Lightning: uv pip install rank-bm25 kiwipiepy && \
    #            uv run --no-sync backend/build_rerank.py --topk 100   (torchvision 회피)
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=100, help="리랭크할 hybrid 후보 수")
    ap.add_argument("--pool", type=int, default=200, help="RRF 입력 후보 풀(BM25/dense 각)")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from kiwipiepy import Kiwi  # noqa: PLC0415
    from rank_bm25 import BM25Okapi  # noqa: PLC0415
    from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: PLC0415

    appids, docs = build_docs()  # Step3/4 와 동일 순서(임베딩 정렬 일치)
    pos = {a: i for i, a in enumerate(appids)}
    demb = np.load(EMB)
    assert demb.shape[0] == len(docs), f"임베딩({demb.shape[0]})과 문서({len(docs)}) 불일치"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rerank] 코퍼스 {len(docs):,} · 임베딩 {demb.shape} · topk={args.topk} · device={dev}")

    evals = [json.loads(l) for l in EVAL.open(encoding="utf-8") if l.strip()]
    evals = [e for e in evals if e["appid"] in pos]
    print(f"[eval] 질의 {len(evals)}개")

    kiwi = Kiwi()
    def tok(s: str) -> list[str]:
        return [t.form for t in kiwi.tokenize(s)]
    if TOKCACHE.exists():
        toks = pickle.loads(TOKCACHE.read_bytes()); print("[bm25] 토큰 캐시 로드")
    else:
        print("[bm25] kiwi 토큰화(150K, 수분)…"); t0 = time.time()
        toks = [tok(d) for d in docs]
        TOKCACHE.write_bytes(pickle.dumps(toks)); print(f"  {time.time()-t0:.0f}s")
    bm25 = BM25Okapi(toks)

    st = SentenceTransformer("BAAI/bge-m3", device=dev)
    qembs = st.encode([e["query"] for e in evals], normalize_embeddings=True).astype(np.float32)
    ce = CrossEncoder(args.reranker, device=dev)

    agg = {m: [0, 0, 0, 0.0] for m in ("hybrid", "hybrid+rerank")}
    t0 = time.time()
    for e, qe in zip(evals, qembs):
        gp = pos[e["appid"]]
        bm = list(np.argsort(bm25.get_scores(tok(e["query"])))[::-1][:args.pool])
        dn = list(np.argsort(qe @ demb.T)[::-1][:args.pool])
        hy = rrf([bm, dn], k=args.rrf_k)
        cand = list(hy)[:args.topk]
        scores = ce.predict([(e["query"], docs[i]) for i in cand], batch_size=args.batch)
        reranked = [cand[j] for j in np.argsort(scores)[::-1]]
        for name, ranks in (("hybrid", hy), ("hybrid+rerank", reranked)):
            r1, r5, r10, rr = metrics(list(ranks), gp)
            a = agg[name]; a[0] += r1; a[1] += r5; a[2] += r10; a[3] += rr
    dt = time.time() - t0

    n = len(evals)
    print(f"\n{'검색기':<16}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}"
          f"  (코퍼스 {len(docs):,}, topk={args.topk})")
    print("-" * 56)
    for m in ("hybrid", "hybrid+rerank"):
        r1, r5, r10, rr = agg[m]
        print(f"{m:<16}{r1/n*100:>7.0f}%{r5/n*100:>7.0f}%{r10/n*100:>7.0f}%{rr/n:>8.3f}")
    pairs = n * args.topk
    print(f"\n[리랭크계측] device={dev} · {dt:.0f}s · {pairs} pairs · {pairs/dt:.0f} pair/s")
    print("[성공기준] hybrid+rerank 의 R@1·MRR 이 hybrid 대비 상승 = 리랭커가 순서문제 해결.")


if __name__ == "__main__":
    main()
