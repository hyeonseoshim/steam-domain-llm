"""RAG 연습 ① — 하이브리드 검색(BM25 + dense + RRF) & Recall@K 평가.

실무통합([[silmu-integration-project]]) 코어인 하이브리드 검색을 스팀 카탈로그로 연습.
코퍼스 = 게임별 한국어 문서(우리가 정규화한 3필드 요약). 세 검색기를 같은 평가셋
(build_eval.py 의 {query, appid})으로 재서 Recall@1/5/10·MRR 비교.

- **BM25**(어휘 정확 매칭, kiwi 형태소 토큰): 희귀 고유명사·장르어에 강, 패러프레이즈에 약.
- **dense**(bge-m3 임베딩, 코사인): 의미·패러프레이즈에 강, 희귀어/오탈자에 약.
- **hybrid = RRF**(Reciprocal Rank Fusion): 두 랭킹을 순위 기반으로 합쳐 서로의 약점 보완.
  RRF_score(d) = Σ_r 1/(k + rank_r(d)),  k=60.
목표(정석 결과): **hybrid ≥ max(BM25, dense)**.

usage:
    uv run --with rank-bm25 --with sentence-transformers --with kiwipiepy --with numpy \
        backend/hybrid_search.py --corpus-n 1500
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
EVAL = Path(__file__).resolve().parent / "eval_queries.jsonl"
CACHE = Path(__file__).resolve().parent / ".emb_cache"


def build_docs(n: int) -> tuple[list[int], list[str]]:
    """gold → (appid 리스트, 한국어 문서 리스트). 문서=정규화 요약 기반."""
    appids, docs = [], []
    for line in GOLD.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("summary") or {}
        doc = f"{r.get('name','')} 장르 {s.get('장르','')} {s.get('핵심플레이','')} {s.get('특징','')}"
        appids.append(r["appid"]); docs.append(doc.strip())
        if n and len(docs) >= n:
            break
    return appids, docs


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    """여러 랭킹(문서 인덱스 순서)을 RRF 로 융합해 최종 순위 반환."""
    score: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, 1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: score[i], reverse=True)


def metrics(ranked_idx: list[int], gold_pos: int) -> tuple[int, int, int, float]:
    """gold 문서 인덱스의 순위로 R@1/5/10, RR 계산."""
    try:
        rank = ranked_idx.index(gold_pos) + 1
    except ValueError:
        return 0, 0, 0, 0.0
    return int(rank <= 1), int(rank <= 5), int(rank <= 10), 1.0 / rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-n", type=int, default=1500, help="코퍼스 문서 수(0=전체)")
    ap.add_argument("--embed-model", default="BAAI/bge-m3",
                    help="임베딩 모델(실무통합 타깃=bge-m3; 빠르게 e5-small 등으로 교체 가능)")
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    from kiwipiepy import Kiwi  # noqa: PLC0415
    from rank_bm25 import BM25Okapi  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    appids, docs = build_docs(args.corpus_n)
    pos = {a: i for i, a in enumerate(appids)}  # appid → 코퍼스 인덱스
    print(f"[corpus] {len(docs)} 문서 · 임베딩={args.embed_model}")

    # 평가셋: 코퍼스 안에 있는 정답만
    evals = [json.loads(l) for l in EVAL.open(encoding="utf-8") if l.strip()]
    evals = [e for e in evals if e["appid"] in pos]
    print(f"[eval] 질의 {len(evals)}개 (코퍼스 내 정답만)\n")

    kiwi = Kiwi()
    def tok(s: str) -> list[str]:
        return [t.form for t in kiwi.tokenize(s)]

    # BM25
    bm25 = BM25Okapi([tok(d) for d in docs])

    # dense (임베딩 캐시)
    st = SentenceTransformer(args.embed_model)
    CACHE.mkdir(exist_ok=True)
    key = hashlib.md5((args.embed_model + str(len(docs)) + docs[0] + docs[-1]).encode()).hexdigest()[:12]
    cache_f = CACHE / f"{key}.npy"
    if cache_f.exists():
        demb = np.load(cache_f)
        print(f"[dense] 캐시 로드 {cache_f.name}")
    else:
        print("[dense] 코퍼스 임베딩 중…")
        demb = st.encode(docs, normalize_embeddings=True, show_progress_bar=True,
                         batch_size=32).astype(np.float32)
        np.save(cache_f, demb)

    qembs = st.encode([e["query"] for e in evals], normalize_embeddings=True).astype(np.float32)

    agg = {m: [0, 0, 0, 0.0] for m in ("BM25", "dense", "hybrid")}
    for e, qe in zip(evals, qembs):
        gp = pos[e["appid"]]
        bm_rank = list(np.argsort(bm25.get_scores(tok(e["query"])))[::-1])
        dn_rank = list(np.argsort(qe @ demb.T)[::-1])
        hy_rank = rrf([bm_rank[:200], dn_rank[:200]], k=args.rrf_k)
        for name, ranks in (("BM25", bm_rank), ("dense", dn_rank), ("hybrid", hy_rank)):
            r1, r5, r10, rr = metrics(list(ranks), gp)
            a = agg[name]; a[0] += r1; a[1] += r5; a[2] += r10; a[3] += rr

    n = len(evals)
    print(f"{'검색기':<10}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}")
    print("-" * 42)
    for m in ("BM25", "dense", "hybrid"):
        r1, r5, r10, rr = agg[m]
        print(f"{m:<10}{r1/n*100:>7.0f}%{r5/n*100:>7.0f}%{r10/n*100:>7.0f}%{rr/n:>8.3f}")
    print("\n[읽는 법] hybrid 가 BM25·dense 단독보다 높으면 RRF 융합이 값한다는 실증"
          " = 실무통합 검색 코어의 핵심 결과.")


if __name__ == "__main__":
    main()
