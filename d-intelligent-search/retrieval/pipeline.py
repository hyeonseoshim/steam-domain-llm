"""Recommendation Pipeline — End-to-End Orchestration.

SQL Filter → Semantic Rerank → MMR/Bucket → Explainer
"""
from __future__ import annotations

import time

import numpy as np
from sqlalchemy.orm import Session

from steam_part_d.config import RetrievalSettings, get_settings
from steam_part_d.extractor.constraint_extractor import (
    ConstraintExtractor,
    get_extractor,
)
from steam_part_d.models.game import (
    Game,
    RecommendationResponse,
    RecommendedGame,
    RetrievalStats,
)
from steam_part_d.retrieval.diversity import apply_diversity
from steam_part_d.retrieval.semantic_rerank import semantic_rerank
from steam_part_d.retrieval.review_rerank import review_rerank
from steam_part_d.retrieval.sql_filter import SQLFilterConfig, apply_exclude, apply_sql_filter, apply_sql_filter_by_appids
from steam_part_d.retrieval.cross_encoder_reranker import cross_encoder_rerank
from steam_part_d.retrieval.vector_search import is_index_ready, vector_search
from steam_part_d.agent.self_reflection import (
    detect_contradiction,
    relax_constraint,
    should_reflect,
)
from steam_part_d.utils.logging import get_logger

    # [2026-07-16 P0-5] Stage 4/5 (semantic_extractor + _enrich_with_features + _feature_boost) 제거.
    # 호출되지만 downstream (ranking/explainer) 어디에서도 사용 안 함 (dead).
    # src/steam_part_d/feature/ 디렉토리는 정리 대상입니다.

logger = get_logger(__name__)


def build_grounded_context(games: list[Game], max_desc_chars: int = 200) -> str:
    """하위 호환용 래퍼 — `steam_part_d.retrieval.grounded`로 위임."""
    from steam_part_d.retrieval.grounded import build_grounded_context as _impl
    return _impl(games, max_desc_chars)


def _query_embedding(free_text: str | None) -> np.ndarray | None:
    """free_text → (1, D) 임베딩. encode_cached 사용으로 재인코딩 방지.

    Korean → English 확장 후 인코딩하여 한국어 쿼리와 영문 설명 간 semantic 갭 완화.
    semantic_rerank()가 이미 encode_cached()로 캐싱했으면 즉시 반환.
    """
    if not free_text or not free_text.strip():
        return None
    from steam_part_d.llm.embedding import encode_cached
    from steam_part_d.retrieval.query_translator import translate_query_for_embedding
    expanded = translate_query_for_embedding(free_text)
    return encode_cached(expanded)


class RecommendationPipeline:
    def __init__(
        self,
        extractor: ConstraintExtractor | None = None,
        explainer=None,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.extractor = extractor or get_extractor()
        if explainer is None:
            from steam_part_d.generator.explainer import get_explainer
            explainer = get_explainer()
        self.explainer = explainer
        self.settings = settings or get_settings().retrieval

    def recommend(
        self,
        session: Session,
        user_query: str,
        *,
        top_k: int | None = None,
        enable_semantic_rerank: bool | None = None,
        enable_mmr: bool | None = None,
        language: str | None = None,
    ) -> RecommendationResponse:
        start = time.perf_counter()
        stats = RetrievalStats()

        # Reasoning Trace (2026-07-07 추가, SHOW_TRACE 옵션)
        trace: dict = {
            "query": user_query,
            "constraints": {},
            "candidate_count": 0,
            "semantic_topk": [],
            "cross_encoder_topk": [],
            "final_topk": [],
            "latency": {},
        }

        # P1-10: Basic sanitization (Jailbreak 방어)
        lower_query = user_query.lower()
        if "ignore previous" in lower_query or "system prompt" in lower_query:
            return RecommendationResponse(
                status="blocked",
                query=user_query,
                constraint={},
                recommended_games=[],
                retrieval_stats=RetrievalStats(),
                latency_ms=0,
                trace={"blocked": True, "reason": "malicious query pattern"}
            )

        # 0. Params initialization
        final_count = top_k or self.settings.final_top_k
        
        # 1. Constraint 추출
        t = time.perf_counter()
        constraint = self.extractor.extract(user_query)
        trace["latency"]["extract"] = int((time.perf_counter() - t) * 1000)
        trace["constraints"] = constraint.summary()
        logger.info(
            "constraint_extracted",
            query=user_query[:80],
            constraint=constraint.summary(),
        )

        # 2. 후보 검색 — Vector Search 또는 SQL Hard Filter
        #
        # [Vector Search 경로] free_text 있고 인덱스 준비된 경우:
        #   전체 150k 게임 벡터 검색 → SQL hard filter (가격/장르/플랫폼) 적용
        #   → metacritic 편중 없이 인디게임 포함 전체에서 관련 게임 발굴
        #
        # [SQL Fallback 경로] free_text 없거나 인덱스 미준비:
        #   기존 방식 — metacritic/recommendations 순 상위 200개
        t = time.perf_counter()
        sql_limit = max(final_count * 10, self.settings.sql_candidate_limit)
        sql_cfg = SQLFilterConfig(
            candidate_limit=sql_limit,
        )

        use_vector = (
            self.settings.enable_vector_search
            and constraint.free_text
            and is_index_ready()
        )

        if use_vector and constraint.free_text:
            from steam_part_d.retrieval.vector_search import vector_search_with_scores
            vs_with_scores = vector_search_with_scores(
                constraint.free_text,
                top_k=self.settings.vector_search_top_k,
            )
            vs_appids = [a for a, _ in vs_with_scores]
            # [2026-07-21] 통합 명세 score — Game.semantic_score에 BGE cosine (0~1) 채움
            vs_score_map = {a: s for a, s in vs_with_scores}
            sql_candidates = apply_sql_filter_by_appids(session, vs_appids, constraint, sql_cfg)
            for g in sql_candidates:
                if g.appid in vs_score_map:
                    g.semantic_score = vs_score_map[g.appid]
            trace["retrieval_mode"] = "vector_search"
        else:
            sql_candidates = apply_sql_filter(session, constraint, sql_cfg)
            trace["retrieval_mode"] = "sql_only"
        # [2026-07-21 배포 P0] must_not/exclude post-filter (Python)
        before_exclude = len(sql_candidates)
        sql_candidates = apply_exclude(sql_candidates, constraint)
        if len(sql_candidates) != before_exclude:
            trace["exclude_filtered"] = before_exclude - len(sql_candidates)
        trace["latency"]["sql"] = int((time.perf_counter() - t) * 1000)
        trace["candidate_count"] = len(sql_candidates)
        stats.sql_candidates = len(sql_candidates)
        logger.info("sql_filter_done", count=stats.sql_candidates)

        # 2-1. Self Reflection (2026-07-07 추가, 옵션)
        # 후보 0개 시 모순 constraint 검출 + 1회 relax 재검색
        if self.settings.enable_self_reflection and should_reflect(len(sql_candidates), attempt=0):
            field, reason = detect_contradiction(constraint)
            if field:
                logger.info(
                    "self_reflection_triggered",
                    field=field,
                    reason=reason,
                    original_constraints=constraint.model_dump(),
                )
                trace["self_reflection"] = {
                    "triggered": True,
                    "contradiction": reason,
                    "relaxed_field": field,
                }
                relaxed = relax_constraint(constraint, field)
                sql_candidates = apply_sql_filter(session, relaxed, sql_cfg)
                stats.sql_candidates = len(sql_candidates)
                trace["candidate_count"] = len(sql_candidates)
                trace["self_reflection"]["candidate_after_relax"] = len(sql_candidates)
                trace["constraints_after_relax"] = relaxed.model_dump()
                logger.info("self_reflection_done", candidates=len(sql_candidates))
            else:
                trace["self_reflection"] = {"triggered": False, "reason": "no contradiction pattern matched"}
        else:
            trace["self_reflection"] = {"triggered": False}

        # 3. Semantic Rerank
        # Vector search 경로: 이미 유사도 순 정렬됨 → 불필요
        # SQL fallback 경로: free_text 있으면 semantic rerank 실행
        use_rerank = (
            enable_semantic_rerank
            if enable_semantic_rerank is not None
            else self.settings.enable_semantic_rerank
        )
        reranked = sql_candidates
        t = time.perf_counter()
        if use_rerank and constraint.free_text and not use_vector:
            semantic_k = max(final_count, self.settings.semantic_rerank_top_k)
            reranked = semantic_rerank(
                sql_candidates,
                constraint.free_text,
                top_k=semantic_k,
            )
        else:
            # vector_search 경로: 이미 정렬됨
            reranked = sql_candidates
        # [2026-07-16 P0-5] Stage 4 _enrich_with_features 호출 제거
        trace["latency"]["semantic"] = int((time.perf_counter() - t) * 1000)
        trace["semantic_topk"] = [g.appid for g in reranked[:final_count]]
        stats.after_rerank = len(reranked)
        stats.cross_encoder_used = False
        logger.info("semantic_rerank_done", count=stats.after_rerank)

        # 3-1. Cross Encoder Reranker (2026-07-07 추가, 옵션)
        # semantic_rerank 결과 또는 sql_candidates 직접 입력으로 top-K 재정렬
        t = time.perf_counter()
        if self.settings.enable_cross_encoder:
            ce_input = reranked if use_rerank else sql_candidates
            ce_top_k = max(final_count, self.settings.cross_encoder_top_k)
            ce_input_limit = max(ce_top_k, self.settings.cross_encoder_input_limit)
            reranked = cross_encoder_rerank(
                ce_input,
                user_query,  # 원본 query 사용 (semantic은 free_text만 vs cross encoder는 전체 query)
                top_k=ce_top_k,
                input_limit=ce_input_limit,
                model_name=self.settings.cross_encoder_model,
            )
            stats.cross_encoder_used = True
            stats.after_cross_encoder = len(reranked)
            trace["cross_encoder_topk"] = [g.appid for g in reranked[:final_count]]
            logger.info("cross_encoder_done", count=stats.after_cross_encoder)
        trace["latency"]["cross_encoder"] = int((time.perf_counter() - t) * 1000)

        # 3-2. Review Signal Reranker (2026-07-09 추가, ② 옵션)
        # free_text 있을 때만 작동. Cross Encoder 결과를 review_similarity로 재정렬.
        t = time.perf_counter()
        if (
            self.settings.enable_review_signal
            and constraint.free_text
            and reranked
        ):
            reranked = review_rerank(
                reranked,
                constraint.free_text,
                top_k=self.settings.cross_encoder_top_k,
                review_weight=self.settings.review_weight,
            )
            trace["review_topk"] = [g.appid for g in reranked[:final_count]]
            stats.review_signal_used = True
            stats.after_review_signal = len(reranked)
            logger.info("review_signal_done", count=len(reranked))
        trace["latency"]["review_signal"] = int((time.perf_counter() - t) * 1000)

        # 4. Diversity (MMR + Bucket)
        use_mmr = (
            enable_mmr if enable_mmr is not None else self.settings.enable_mmr
        )
        query_emb = _query_embedding(constraint.free_text)
        t = time.perf_counter()
        
        # [FIX 2026-07-23] mmr_k가 final_count보다 작으면 결과가 잘림. 
        # final_count에 맞춰 동적으로 mmr_k 조정.
        current_mmr_k = max(final_count, self.settings.mmr_top_k)
        
        final = apply_diversity(
            reranked,
            enable_mmr=use_mmr,
            mmr_k=current_mmr_k,
            mmr_lambda=self.settings.mmr_lambda,
            query_embedding=query_emb,
            bucket_k=final_count,
            bucket_max_per_genre=self.settings.bucket_max_per_genre,
        )
        trace["latency"]["diversity"] = int((time.perf_counter() - t) * 1000)
        stats.after_mmr = len(final)
        stats.final_recommended = len(final)
        logger.info("diversity_done", count=stats.after_mmr)

        # 5. Explainer
        t = time.perf_counter()
        explanations = self.explainer.explain(
            user_query=user_query,
            constraint=constraint,
            candidates=final,
            top_k=final_count,
            language=language or "ko",
        )
        trace["latency"]["llm"] = int((time.perf_counter() - t) * 1000)

        # RecommendedGame 조립
        by_appid = {g.appid: g for g in final}
        # [2026-07-16 P1-1] Source attribution — retrieval mode + score 전달
        retrieval_mode = trace.get("retrieval_mode", "sql_only")
        recommended: list[RecommendedGame] = []
        for rank, item in enumerate(explanations, 1):
            app_id = item["app_id"]
            g = by_appid.get(app_id)
            if g is None:
                # hallucination 방어: LLM이 모르는 app_id 언급 시 스킵
                logger.warning(
                    "explainer_hallucinated_app_id",
                    app_id=app_id,
                    name=item.get("name"),
                )
                continue
            # [2026-07-16 P1-1] 각 추천의 retrieval source 기록
            source = {
                "retrieval_mode": retrieval_mode,
                "semantic_score": round(g.semantic_score, 4) if g.semantic_score is not None else None,
                "sql_candidate_count": stats.sql_candidates,
                "after_mmr_count": stats.after_mmr,
            }
            # ✅ 게임 객체에 explanation 저장 (Downstream routes.py에서 사용 가능하도록)
            g.explanation = item["explanation"]
            
            recommended.append(
                RecommendedGame(
                    rank=rank,
                    app_id=app_id,
                    name=g.name,
                    price_usd=g.price_usd,
                    is_free=g.is_free,
                    genres=g.genres,
                    categories=g.categories,
                    # [2026-07-21 배포 P0] platforms 전달 — reason/exclude 일관성
                    platforms=g.platforms,
                    explanation=item["explanation"],
                    source=source,
                    # [2026-07-20] 통합 명세 score — semantic_score 0~1, 없으면 None
                    semantic_score=round(g.semantic_score, 4) if g.semantic_score is not None else None,
                )
            )
            if len(recommended) >= final_count:
                break

        # 만약 LLM이 더 적게 추천하면 빈 슬롯을 code가 채움 (grounding 보강)
        if len(recommended) < final_count:
            used_ids = {r.app_id for r in recommended}
            for g in final:
                if g.appid in used_ids:
                    continue
                recommended.append(
                    RecommendedGame(
                        rank=len(recommended) + 1,
                        app_id=g.appid,
                        name=g.name,
                        price_usd=g.price_usd,
                        is_free=g.is_free,
                        genres=g.genres,
                        categories=g.categories,
                        platforms=g.platforms,
                        explanation="",  # LLM이 언급 안 함 → 빈 설명
                        semantic_score=round(g.semantic_score, 4) if g.semantic_score is not None else None,
                    )
                )
                if len(recommended) >= final_count:
                    break

        stats.final_recommended = len(recommended)
        latency_ms = int((time.perf_counter() - start) * 1000)
 
        # trace 완성 (2026-07-07 추가)
        trace["final_topk"] = [g.app_id for g in recommended[:final_count]]
        trace["total_latency"] = latency_ms

        return RecommendationResponse(
            status="ok",
            query=user_query,
            constraint=constraint.summary(),
            recommended_games=recommended,
            retrieval_stats=stats,
            latency_ms=latency_ms,
            trace=trace,
        )
