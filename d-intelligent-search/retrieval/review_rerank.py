"""Review Signal Reranker — 게임 리뷰 임베딩 기반 재정렬.

게임별 top 5 영어 리뷰 임베딩 평균 (review_vector)과 query 임베딩의
cosine similarity로 Cross Encoder 결과를 재정렬.

사용 조건:
- free_text가 있을 때만 (SQL/카테고리 매칭이 안 되는 모호한 쿼리)
- enable_review_signal 옵션 ON일 때

Score blend:
- Cross Encoder 결과 (top_K)를 input으로 받음
- review_similarity = query_emb · review_vector[game] (L2 normalized)
- final_score = ce_rank_weight × ce_score + review_weight × review_sim

근데 ce_score 스케일 모름 → 단순 재정렬:
- ce_score 정규화 (min-max) 후 blend
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from steam_part_d.retrieval.sql_filter import Game

logger = structlog.get_logger(__name__)

# Lazy global cache
_review_vectors: np.ndarray | None = None
_appid_to_row: dict[int, int] | None = None


def _load_review_vectors(
    npy_path: str = "data/steam_review_vectors.npy",
    map_path: str = "data/steam_review_vectors_map.csv",
) -> tuple[np.ndarray, dict[int, int]]:
    """리뷰 벡터 lazy load (첫 호출에만)."""
    global _review_vectors, _appid_to_row
    if _review_vectors is not None:
        return _review_vectors, _appid_to_row

    import csv
    from pathlib import Path

    npy_path = Path(npy_path)
    map_path = Path(map_path)

    if not npy_path.exists() or not map_path.exists():
        logger.warning("review_vectors_not_found", npy=str(npy_path))
        return None, None

    _review_vectors = np.load(str(npy_path))

    appid_to_row = {}
    with map_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            appid_to_row[int(row[0])] = int(row[1])
    _appid_to_row = appid_to_row

    logger.info("review_vectors_loaded", shape=list(_review_vectors.shape))
    return _review_vectors, appid_to_row


def review_rerank(
    games: list["Game"],
    free_text: str,
    top_k: int = 10,
    review_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> list["Game"]:
    """리뷰 벡터 + 쿼리 임베딩 cosine sim으로 재정렬.

    Args:
        games: Cross Encoder 결과 또는 semantic_rerank 결과
        free_text: 사용자 쿼리 자유 텍스트 부분
        top_k: 반환할 게임 수
        review_weight: review_sim 가중치 (1.0이면 review만)
        semantic_weight: (1 - review_weight)로 자동 계산

    Returns:
        재정렬된 games (top_k)
    """
    if not free_text or not free_text.strip():
        return games[:top_k]
    if not games:
        return games

    rev_vecs, appid_to_row = _load_review_vectors()
    if rev_vecs is None:
        logger.warning("review_rerank_skipped_no_vectors")
        return games[:top_k]

    # Query 임베딩 (BGE-M3, normalize=True)
    from steam_part_d.llm.embedding import get_embedding_model
    model = get_embedding_model()
    q_emb = model.encode(free_text, normalize=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    # 각 게임의 review similarity 계산
    scores = []
    for g in games:
        row = appid_to_row.get(g.appid)
        if row is None:
            scores.append(0.0)
            continue
        rv = rev_vecs[row]
        # rv는 L2 normalized. .item()으로 scalar 추출 (0-dim 보장)
        sim = float(np.dot(q_emb, rv).item())
        scores.append(sim)

    # 재정렬: review_sim 기준
    paired = sorted(
        zip(games, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info(
        "review_rerank_done",
        input_count=len(games),
        output_count=min(top_k, len(games)),
        top_scores=[f"{s:.3f}" for _, s in paired[:3]],
    )

    return [g for g, _ in paired[:top_k]]