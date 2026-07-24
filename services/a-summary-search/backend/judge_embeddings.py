"""RAG 연습 ①-b — Zenodo pre-computed 임베딩 vs 우리-자체 임베딩 판정.

Zenodo 임베딩 패키지(applications_embeddings.npy = 전체 앱 1024-dim BGE-M3, 원문
short+detailed 인코딩)를 우리 60질의 평가셋으로 검색 실측해, "직접 인코딩 대신
받아 쓸 값이 있나"를 숫자로 판정한다.

두 문서표현을 같은 60질의·같은 gold appid 코퍼스에서 dense 검색 비교:
- **ours**: 우리가 만든 한국어 요약(name+장르+핵심플레이+특징)을 bge-m3로 인코딩(질의와 동일 언어).
- **zenodo**: 원문(영/중 short+detailed)을 인코딩한 pre-computed 벡터(질의와 교차언어).
질의는 bge-m3로 인코딩(양쪽 공용). 지표=Recall@1/5/10·MRR.

usage:
    uv run --with sentence-transformers --with numpy backend/judge_embeddings.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
EVAL = Path(__file__).resolve().parent / "eval_queries.jsonl"
CACHE = Path(__file__).resolve().parent / ".emb_cache"
EMB_DIR = Path("zenodo_dl/steam_dataset_2025_embeddings")
EMB_NPY = EMB_DIR / "applications_embeddings.npy"   # raw float32 (239664,1024), 헤더 없음
EMB_MAP = EMB_DIR / "applications_embedding_map.csv"  # vector_index,appid
N_APP, DIM = 239664, 1024


CSV_APP = Path("steam_dataset_2025_csv/applications.csv")


def build_docs() -> tuple[list[int], list[str], list[str]]:
    """gold → (appid, 한국어 요약, detailed 원문=input). hybrid_search 와 동일 요약 규칙."""
    appids, summ, detailed = [], [], []
    for line in GOLD.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("summary") or {}
        doc = f"{r.get('name','')} 장르 {s.get('장르','')} {s.get('핵심플레이','')} {s.get('특징','')}"
        appids.append(r["appid"]); summ.append(doc.strip())
        detailed.append((r.get("input") or "").strip())
    return appids, summ, detailed


def load_short(appids: list[int]) -> list[str]:
    """CSV 에서 appid 별 short_description(원문). 없으면 빈 문자열."""
    import csv as _csv  # noqa: PLC0415
    _csv.field_size_limit(10**7)
    want = set(appids); sd = {}
    with CSV_APP.open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                aid = int(row["appid"])
            except (ValueError, TypeError):
                continue
            if aid in want and (row.get("short_description") or "").strip():
                sd[aid] = row["short_description"].strip()
    return [sd.get(a, "") for a in appids]


def encode_cached(st, texts: list[str], tag: str) -> np.ndarray:
    """텍스트 리스트를 bge-m3 로 인코딩(캐시). tag 로 캐시 키 구분."""
    key = hashlib.md5((tag + str(len(texts)) + texts[0][:80] + texts[-1][:80]).encode()).hexdigest()[:12]
    cache_f = CACHE / f"{tag}_{key}.npy"
    if cache_f.exists():
        print(f"[{tag}] 캐시 로드 {cache_f.name}"); return np.load(cache_f)
    print(f"[{tag}] 인코딩 중 {len(texts)}개…")
    emb = st.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                    batch_size=16).astype(np.float32)
    CACHE.mkdir(exist_ok=True); np.save(cache_f, emb)
    return emb


def metrics(ranked_idx: np.ndarray, gold_pos: int) -> tuple[int, int, int, float]:
    hit = np.where(ranked_idx == gold_pos)[0]
    if len(hit) == 0:
        return 0, 0, 0, 0.0
    rank = int(hit[0]) + 1
    return int(rank <= 1), int(rank <= 5), int(rank <= 10), 1.0 / rank


def score(qembs: np.ndarray, docmat: np.ndarray, gold_pos: list[int]) -> tuple:
    """질의×문서 내적으로 랭킹 → R@1/5/10·MRR 집계. gold_pos[i]=질의 i의 정답 문서행."""
    agg = [0, 0, 0, 0.0]
    sims = qembs @ docmat.T                      # (Q, Ndoc)
    order = np.argsort(-sims, axis=1)            # 내림차순 랭킹
    for i, gp in enumerate(gold_pos):
        r1, r5, r10, rr = metrics(order[i], gp)
        agg[0] += r1; agg[1] += r5; agg[2] += r10; agg[3] += rr
    n = len(gold_pos)
    return agg[0]/n, agg[1]/n, agg[2]/n, agg[3]/n


def main() -> None:
    import random  # noqa: PLC0415

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    appids, summ, detailed = build_docs()
    short_all = load_short(appids)

    evals = [json.loads(l) for l in EVAL.open(encoding="utf-8") if l.strip()]
    tgt = {e["appid"] for e in evals if e["appid"] in set(appids)}

    # --- 코퍼스 1500 서브셋: 60 정답 전부 + 랜덤 채움(시드 고정). CPU 인코딩 시간 절약 ---
    SUB = 1500
    rng = random.Random(0)
    fill = [a for a in appids if a not in tgt]
    rng.shuffle(fill)
    keep = list(tgt) + fill[:SUB - len(tgt)]
    keep_set = set(keep)
    idx = [i for i, a in enumerate(appids) if a in keep_set]  # gold 원순서 유지
    sub_appids = [appids[i] for i in idx]
    sub_summ = [summ[i] for i in idx]
    sub_detail = [detailed[i] for i in idx]
    sub_short = [short_all[i] for i in idx]
    pos = {a: j for j, a in enumerate(sub_appids)}
    evals = [e for e in evals if e["appid"] in pos]
    gold_pos = [pos[e["appid"]] for e in evals]
    print(f"[corpus] {len(sub_appids)} 서브셋(정답 {len(tgt)} 포함)  [eval] 질의 {len(evals)}개")

    st = SentenceTransformer("BAAI/bge-m3")
    st.max_seq_length = 512  # detailed 롱텍스트 인코딩 시간 제한(앞 512토큰=요지)
    qembs = st.encode([e["query"] for e in evals], normalize_embeddings=True).astype(np.float32)

    # ours: 옛 전체(4158) 캐시에서 서브셋 행만 인덱싱(재인코딩 회피)
    ours_key = hashlib.md5(("BAAI/bge-m3" + str(len(appids)) + summ[0] + summ[-1]).encode()).hexdigest()[:12]
    ours_full = CACHE / f"{ours_key}.npy"
    if ours_full.exists():
        ours_sub = np.load(ours_full)[idx]; print(f"[ours] 전체캐시 {ours_full.name} → 서브셋")
    else:
        ours_sub = encode_cached(st, sub_summ, "summ_ko_sub")

    reps = {
        "ours(한국어요약·동일언어)": ours_sub,
        "self·detailed(원문롱)": encode_cached(st, sub_detail, "detail_sub"),
        "self·short_desc(원문짧)": encode_cached(st, sub_short, "short_sub"),
    }

    # zenodo pre-computed → 서브셋 정렬
    a2v = {}
    with EMB_MAP.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a2v[int(row["appid"])] = int(row["vector_index"])
    emb = np.fromfile(EMB_NPY, dtype=np.float32).reshape(N_APP, DIM)
    zmat = np.zeros((len(sub_appids), DIM), dtype=np.float32)
    for j, a in enumerate(sub_appids):
        if a in a2v:
            zmat[j] = emb[a2v[a]]
    reps["zenodo(원문 short+detail)"] = zmat

    print(f"\n{'문서 표현':<26}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>8}")
    print("-" * 55)
    for name, mat in reps.items():
        r = score(qembs, mat, gold_pos)
        print(f"{name:<26}{r[0]*100:>6.0f}%{r[1]*100:>6.0f}%{r[2]*100:>6.0f}%{r[3]:>8.3f}")
    print(f"\n[주의] 코퍼스={len(sub_appids)} 서브셋이라 절대수치는 4158과 다름(더 쉬움). 표현 간 상대비교용.")
    print("[읽는 법] self·detailed ≈ zenodo → 격차 원인=언어(교차언어), 텍스트선택 아님.")
    print("          short_desc < detailed → 얇은 텍스트 손해까지 확인.")


if __name__ == "__main__":
    main()
