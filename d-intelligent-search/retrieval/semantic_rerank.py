"""Semantic Rerank — 디자인 doc 섹션 5.2.

Constraint.free_text가 있을 때만 동작.
BGE-M3 임베딩 코사인 유사도로 후보 재정렬.

DB의 applications.description_embedding (캐시됨) 사용.
캐시 미스 시 lazy하게 계산.

2026-07-13: encode_cached() 사용으로 같은 free_text 재인코딩 방지.
            pipeline의 MMR diversity도 같은 캐시 재사용 → latency 개선.
"""
from __future__ import annotations

import numpy as np

from steam_part_d.llm.embedding import encode_cached, get_embedding_model
from steam_part_d.models.game import Game
from steam_part_d.retrieval.query_translator import translate_query_for_embedding
from steam_part_d.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_embeddings(games: list[Game]) -> None:
    """description_embedding이 None인 게임은 in-place로 채움."""
    need_embed = [g for g in games if not g.description_embedding]
    if not need_embed:
        return

    embedder = get_embedding_model()
    texts = [
        (g.short_description or g.name or "")
        for g in need_embed
    ]
    embs = embedder.encode(texts, normalize=True)
    for game, emb in zip(need_embed, embs):
        game.description_embedding = emb.tolist()


def _cosine_scores(query_emb: np.ndarray, candidates_emb: np.ndarray) -> np.ndarray:
    """(1, D) @ (N, D)^T → (N,) — 임베딩은 이미 정규화됨."""
    if candidates_emb.size == 0:
        return np.zeros(0)
    return (candidates_emb @ query_emb.T).flatten()


def semantic_rerank(
    candidates: list[Game],
    free_text: str | None,
    top_k: int = 50,
) -> list[Game]:
    """free_text 의미가 비슷한 후보 위로 재정렬.

    Returns:
        top_k 개 후보. free_text가 비어있으면 입력 그대로 top_k 자른 것.
    """
    if not candidates:
        return []
    if not free_text or not free_text.strip():
        return candidates[:top_k]

    _ensure_embeddings(candidates)

    # Korean → English 확장 후 인코딩 (한국어 쿼리와 영문 설명 간 semantic 갭 완화)
    # encode_cached: 같은 free_text로 pipeline._query_embedding이 다시 호출해도
    # 재인코딩 없이 캐시 반환 → latency 절감
    expanded = translate_query_for_embedding(free_text)
    query_emb = encode_cached(expanded)

    candidate_embs = np.array(
        [g.description_embedding for g in candidates], dtype=np.float32
    )
    if candidate_embs.size == 0:
        return candidates[:top_k]

    scores = _cosine_scores(query_emb, candidate_embs)
    for game, score in zip(candidates, scores):
        game.semantic_score = float(score)

    ranked = sorted(candidates, key=lambda g: g.semantic_score or 0.0, reverse=True)
    return ranked[:top_k]
