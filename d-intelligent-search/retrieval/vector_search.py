"""전체 게임 벡터 검색 모듈.

DB에서 description_embedding을 메모리에 올려 두고,
쿼리 벡터와의 코사인 유사도로 상위 K개 appid를 반환.

특징:
- 최초 호출 시 전체 벡터 로드 후 싱글톤 캐싱 (프로세스 생존 동안 유지)
- 설명 없는 게임(임베딩 NULL)은 자동 제외
- numpy 배치 행렬 곱으로 150k x 1024 코사인 계산 ~50-200ms

사용 예:
    appids = vector_search(free_text="공포 게임", top_k=500)
    # -> [238210, 550, 883710, ...] (유사도 높은 appid 순)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

from steam_part_d.utils.logging import get_logger

logger = get_logger(__name__)

_INDEX: "VectorIndex | None" = None
_INDEX_LOCK = threading.Lock()


@dataclass
class VectorIndex:
    """메모리 내 벡터 인덱스."""

    appids: np.ndarray   # shape (N,), dtype int64
    matrix: np.ndarray   # shape (N, D), dtype float32, L2-normalized
    loaded_at: float
    count: int


def _load_index(db_path: str) -> VectorIndex:
    """SQLite에서 전체 임베딩 로드 -> numpy 행렬 구성."""
    logger.info("vector_index_loading", db=db_path)
    t = time.perf_counter()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT appid, description_embedding FROM applications "
        "WHERE type = 'game' AND description_embedding IS NOT NULL AND description_embedding != ''"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        logger.warning(
            "vector_index_empty",
            message="description_embedding 없음 -- scripts/build_embeddings.py 먼저 실행",
        )
        return VectorIndex(
            appids=np.array([], dtype=np.int64),
            matrix=np.empty((0, 1024), dtype=np.float32),
            loaded_at=time.time(),
            count=0,
        )

    appids: list[int] = []
    vecs: list[list[float]] = []
    for appid, emb_raw in rows:
        try:
            vec = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
            appids.append(int(appid))
            vecs.append(vec)
        except Exception:
            continue

    matrix = np.array(vecs, dtype=np.float32)

    # L2 정규화 (안전하게 재정규화)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix /= norms

    elapsed = time.perf_counter() - t
    logger.info(
        "vector_index_loaded",
        count=len(appids),
        shape=list(matrix.shape),
        elapsed_sec=round(elapsed, 2),
        size_mb=round(matrix.nbytes / 1024 / 1024, 1),
    )

    return VectorIndex(
        appids=np.array(appids, dtype=np.int64),
        matrix=matrix,
        loaded_at=time.time(),
        count=len(appids),
    )


def get_vector_index(db_path: str | None = None) -> VectorIndex:
    """전역 싱글톤 인덱스 반환. 최초 호출 시 DB에서 로드."""
    global _INDEX
    if _INDEX is not None and _INDEX.count > 0:
        return _INDEX

    with _INDEX_LOCK:
        if _INDEX is not None and _INDEX.count > 0:
            return _INDEX

        if db_path is None:
            db_path = _resolve_db_path()

        _INDEX = _load_index(db_path)
    return _INDEX


def _resolve_db_path() -> str:
    try:
        from steam_part_d.config import get_settings
        db_url = get_settings().database.url
        if db_url.startswith("sqlite:///"):
            return db_url[len("sqlite:///"):]
    except Exception:
        pass
    return "data/00-DB/steam.db"


def reset_vector_index() -> None:
    """테스트/재시작용 -- 인덱스 초기화."""
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = None


def is_index_ready() -> bool:
    """임베딩 인덱스가 로드됐고 비어있지 않으면 True."""
    return _INDEX is not None and _INDEX.count > 0


def vector_search(
    free_text: str,
    *,
    top_k: int = 500,
    db_path: str | None = None,
) -> list[int]:
    """free_text와 유사한 게임 appid 목록 반환 (유사도 내림차순).

    Args:
        free_text: 검색 텍스트 ("공포", "open world RPG", "좀비 생존" 등)
        top_k: 반환할 최대 게임 수 (이후 SQL hard filter에서 재축소됨)
        db_path: SQLite 경로 (None이면 config에서 읽음)

    Returns:
        list[int]: appid 목록 (유사도 높은 순). 인덱스 비어있으면 빈 리스트.
    """
    index = get_vector_index(db_path)

    if index.count == 0:
        logger.warning("vector_search_skipped", reason="인덱스 비어있음 -- SQL fallback 사용")
        return []

    from steam_part_d.llm.embedding import encode_cached
    from steam_part_d.retrieval.query_translator import translate_query_for_embedding

    # Korean → English 쿼리 확장: 한국어 게이밍 용어를 영어로 병합하여
    # 영문 게임 설명과의 semantic 유사도를 높임.
    expanded = translate_query_for_embedding(free_text)
    if expanded != free_text:
        logger.debug(
            "query_translated_for_embedding",
            original=free_text[:60],
            expanded=expanded[:120],
        )

    query_vec = encode_cached(expanded)  # (1, D)

    t = time.perf_counter()
    scores = (index.matrix @ query_vec.T).squeeze()  # (N,)
    elapsed_ms = (time.perf_counter() - t) * 1000

    actual_k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    appids = [int(x) for x in index.appids[top_indices]]
    top_scores = [float(scores[i]) for i in top_indices]

    logger.info(
        "vector_search_done",
        query=free_text[:50],
        candidates=len(appids),
        search_ms=round(elapsed_ms, 1),
        top_score=round(top_scores[0], 3) if top_scores else 0.0,
    )
    return appids


def vector_search_with_scores(
    free_text: str,
    *,
    top_k: int = 500,
    db_path: str | None = None,
) -> list[tuple[int, float]]:
    """[2026-07-21] 통합 명세 score 채우기 — vector_search + score (0~1) 함께 반환.

    vector_search()와 동일하지만 (appid, score) 튜플 반환. score는 BGE-M3 cosine
    similarity (0~1, L2-normalized → 1 - cosine_distance와 동치).

    Args:
        free_text: 검색 텍스트
        top_k: 반환할 최대 게임 수
        db_path: SQLite 경로

    Returns:
        list[tuple[appid, score]]: appid와 cosine score 튜플 (score 내림차순)
    """
    appids = vector_search(free_text, top_k=top_k, db_path=db_path)
    if not appids:
        return []

    # 두 번째 호출 대신 vector_search 내부 로직 재사용 — 이미 메모리 인덱스 있음
    # 0~1로 클램프 (cosine similarity -1~1 → 0~1)
    index = get_vector_index(db_path)
    from steam_part_d.llm.embedding import encode_cached
    from steam_part_d.retrieval.query_translator import translate_query_for_embedding

    expanded = translate_query_for_embedding(free_text)
    query_vec = encode_cached(expanded)
    scores_all = (index.matrix @ query_vec.T).squeeze()

    appid_to_idx = {int(a): i for i, a in enumerate(index.appids)}
    result: list[tuple[int, float]] = []
    for appid in appids:
        idx = appid_to_idx.get(appid)
        if idx is None:
            continue
        sim = float(scores_all[idx])
        sim_clamped = max(0.0, min(1.0, (sim + 1.0) / 2.0))  # -1~1 → 0~1
        result.append((appid, sim_clamped))
    return result
