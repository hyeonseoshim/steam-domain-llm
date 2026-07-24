"""Cross Encoder Reranker (2026-07-07 추가).

기존 Semantic Rerank (BGE-M3 cosine) 뒤에 추가 옵션으로 동작.
Cross Encoder는 query-document 쌍을 직접 점수 매김 — semantic cosine보다 정확.

구조:
    SQL Filter → [Semantic Rerank] → Cross Encoder → Top-K

Config로 On/Off:
    settings.retrieval.cross_encoder_enabled = True

기본 모델:
    BAAI/bge-reranker-v2-m3 (다국어, 568M params)

참고:
    - 입력 후보는 semantic_rerank 결과 또는 SQL filter 직접 결과
    - Top 30~50개 입력 → Top K 출력 (Cross Encoder는 느리므로)
    - 의미: dense semantic search를 coarse ranking, cross encoder를 fine reranking
      (이게 2024~2025 RAG 표준 패턴)
"""
from __future__ import annotations

import os
from threading import Lock
from typing import List

from steam_part_d.models.game import Game
from steam_part_d.utils.logging import get_logger

logger = get_logger(__name__)


# Lazy singleton (모델 로딩 시간 절약 — 첫 호출 시만)
_cross_encoder = None
_lock = Lock()


def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "mps"):
    """Cross Encoder 모델 lazy load (singleton)."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    with _lock:
        if _cross_encoder is not None:
            return _cross_encoder
        from sentence_transformers import CrossEncoder

        logger.info("cross_encoder_loading", model=model_name, device=device)
        _cross_encoder = CrossEncoder(model_name, device=device)
        logger.info("cross_encoder_loaded", model=model_name)
        return _cross_encoder


def cross_encoder_rerank(
    games: List[Game],
    query: str,
    *,
    top_k: int = 5,
    input_limit: int = 50,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "mps",
) -> List[Game]:
    """Cross Encoder rerank.

    Args:
        games: 후보 게임 리스트 (보통 semantic_rerank 결과 또는 SQL filter 직접 결과)
        query: 원본 사용자 query (또는 free_text)
        top_k: 최종 반환 개수
        input_limit: Cross Encoder 입력 최대 개수 (성능 보호)
        model_name: Cross Encoder 모델 이름
        device: 'mps' / 'cuda' / 'cpu'

    Returns:
        Cross Encoder 점수로 재정렬된 게임 리스트 (top_k개)
    """
    if not games:
        return games

    # 입력이 top_k 이하면 재정렬 의미 없음
    if len(games) <= top_k:
        return games

    # 입력 제한 (Cross Encoder는 O(n) 비용)
    candidates = games[:input_limit]
    model = get_cross_encoder(model_name=model_name, device=device)

    # (query, document) 쌍 만들기 — game name + description
    pairs = []
    for g in candidates:
        text = (g.short_description or g.name or "").strip()
        if len(text) > 200:
            text = text[:200]
        pairs.append((query, f"{g.name}: {text}"))

    # 점수 매김
    scores = model.predict(pairs)

    # 점수로 정렬
    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    reranked = [g for _, g in ranked[:top_k]]
    logger.info(
        "cross_encoder_rerank_done",
        input_size=len(candidates),
        output_size=len(reranked),
    )
    return reranked


def reset_cross_encoder() -> None:
    """테스트용 — singleton 초기화."""
    global _cross_encoder
    _cross_encoder = None