"""Diversity 모듈 — 디자인 doc 섹션 7.

두 단계:
1. MMR (Maximal Marginal Relevance) — 임베딩 유사도 기반 중복 제거
2. Bucket Sampling — 장르별 최대 쿼터로 편향 방지

원본 doc의 FAISS 예제는 embedding=None으로 잘못 호출되어 동작 안 함.
여기서는 numpy로 직접 구현 (FAISS 의존성 제거 + 명확한 동작).
"""
from __future__ import annotations

import numpy as np

from steam_part_d.llm.embedding import get_embedding_model
from steam_part_d.models.game import Game
from steam_part_d.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_embeddings(games: list[Game]) -> None:
    need_embed = [g for g in games if not g.description_embedding]
    if not need_embed:
        return
    embedder = get_embedding_model()
    texts = [(g.short_description or g.name or "") for g in need_embed]
    embs = embedder.encode(texts, normalize=True)
    for game, emb in zip(need_embed, embs):
        game.description_embedding = emb.tolist()


def _matrix(games: list[Game]) -> np.ndarray:
    """Game 리스트의 임베딩 행렬."""
    _ensure_embeddings(games)
    return np.array(
        [g.description_embedding or [0.0] * 1024 for g in games],
        dtype=np.float32,
    )


def mmr_select(
    candidates: list[Game],
    query_embedding: np.ndarray | None,
    k: int = 10,
    lambda_mult: float = 0.5,
) -> list[Game]:
    """MMR (Maximal Marginal Relevance).

    lambda_mult = 1 → 관련성 최대 (관련성만)
    lambda_mult = 0 → 다양성 최대
    0.5 → 균형

    Args:
        candidates: 후보 게임 (description_embedding 또는 채울 수 있어야 함).
        query_embedding: 쿼리 임베딩 (1, D). None이면 후보 간 유사도만으로 다양성 추구.
        k: 최종 선택 개수.
        lambda_mult: 관련성 vs 다양성 가중치.

    Returns:
        선택된 k개 게임 (관련성 + 다양성 균형).
    """
    if not candidates:
        return []
    if k >= len(candidates):
        return list(candidates)

    emb_matrix = _matrix(candidates)
    # 임베딩 정규화 확인
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb_matrix / norms

    # 쿼리 임베딩
    if query_embedding is None:
        # 쿼리 없으면 후보 간 다양성만 추구 — 첫 선택은 임의
        query_sim = np.zeros(len(candidates), dtype=np.float32)
    else:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            query_sim = np.zeros(len(candidates), dtype=np.float32)
        else:
            qn = query_embedding / q_norm
            query_sim = (emb_norm @ qn.T).flatten()

    selected_idx: list[int] = []
    remaining = set(range(len(candidates)))

    # 첫 번째: query_sim 최대
    first = int(np.argmax(query_sim))
    selected_idx.append(first)
    remaining.remove(first)

    for _ in range(k - 1):
        if not remaining:
            break
        # 후보 - 이미 선택된 항목 간 유사도 계산
        sel_emb = emb_norm[selected_idx]
        cand_emb = emb_norm[list(remaining)]

        # max similarity with any selected
        sim_to_selected = (cand_emb @ sel_emb.T).max(axis=1)

        rem_list = list(remaining)
        rem_scores = query_sim[rem_list]

        mmr_score = lambda_mult * rem_scores - (1 - lambda_mult) * sim_to_selected
        best_local = int(np.argmax(mmr_score))
        best_global = rem_list[best_local]

        selected_idx.append(best_global)
        remaining.remove(best_global)

    return [candidates[i] for i in selected_idx]


def bucket_sample(
    candidates: list[Game],
    k: int = 10,
    max_per_genre: int = 2,
) -> list[Game]:
    """장르 편향 방지.

    primary_genre = genres[0] (첫 번째 장르)을 기준으로 분산.
    게임이 여러 장르에 속할 때 첫 번째가 primary로 간주됨 (Steam 데이터셋은
    genre 등록 순서가 일정하지 않으므로 perfect하지 않음).
    """
    if not candidates:
        return []
    if k >= len(candidates):
        return list(candidates)

    genre_count: dict[str, int] = {}
    selected: list[Game] = []
    for game in candidates:
        primary = (game.genres[0] if game.genres else "Unknown") or "Unknown"
        if genre_count.get(primary, 0) >= max_per_genre:
            continue
        selected.append(game)
        genre_count[primary] = genre_count.get(primary, 0) + 1
        if len(selected) >= k:
            break
    return selected


def apply_diversity(
    candidates: list[Game],
    *,
    enable_mmr: bool = True,
    mmr_k: int = 20,
    mmr_lambda: float = 0.5,
    query_embedding: np.ndarray | None = None,
    bucket_k: int | None = None,
    bucket_max_per_genre: int = 2,
) -> list[Game]:
    """MMR → Bucket pipeline."""
    if not candidates:
        return []
    out = candidates
    if enable_mmr and mmr_k > 0 and len(out) > mmr_k:
        out = mmr_select(
            out,
            query_embedding=query_embedding,
            k=mmr_k,
            lambda_mult=mmr_lambda,
        )
    if bucket_k is not None and bucket_k > 0 and len(out) > bucket_k:
        out = bucket_sample(out, k=bucket_k, max_per_genre=bucket_max_per_genre)
    return out
