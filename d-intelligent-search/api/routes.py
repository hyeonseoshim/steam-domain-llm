"""API routes — 디자인 doc 섹션 10."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from steam_part_d import __version__
from steam_part_d.api.schemas import (
    ErrorResponse,
    HealthResponse,
    RecommendRequest,
)
from steam_part_d.db.session import get_db
from steam_part_d.llm.client import get_llm_client
from steam_part_d.config import get_settings
from steam_part_d.retrieval.pipeline import RecommendationPipeline
from steam_part_d.utils.cache import get_cache
from steam_part_d.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/parts/d", tags=["part-d"])

_pipeline: RecommendationPipeline | None = None
_agent: "RecommendationAgent | None" = None


def get_pipeline() -> RecommendationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RecommendationPipeline()
    return _pipeline


def get_agent():
    """Agent (2026-07-07 추가) — use_agent 옵션 시 RecommendationAgent 사용."""
    global _agent
    if _agent is None:
        from steam_part_d.agent.recommendation_agent import RecommendationAgent
        _agent = RecommendationAgent(pipeline=get_pipeline(), use_llm=False)
    return _agent


@router.post(
    "/recommend",
    response_model=None,  # 동적 응답 (성공/실패)
    responses={
        200: {"description": "추천 결과"},
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        404: {"model": ErrorResponse, "description": "후보 없음"},
        422: {"model": ErrorResponse, "description": "Constraint 추출 실패"},
        500: {"model": ErrorResponse, "description": "추론 오류"},
    },
)
def recommend(
    req: RecommendRequest,
    db: Session = Depends(get_db),
) -> dict:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty (한국어 자연어 조건을 입력하세요)")

    options = req.options or {}
    top_k = options.get("top_k")
    enable_rerank = options.get("enable_semantic_rerank")
    enable_mmr = options.get("enable_mmr")
    language = options.get("language")
    # 모델 선택: 요청에서 명시하면 그 모델 사용, 없으면 기본 (LoRA 활성화 시 .env의 LoRA 모델)
    model_name = options.get("model_name")
    # Minimal Agent (2026-07-07 추가) — use_agent=True 시 RecommendationAgent 사용
    use_agent = options.get("use_agent", False)

    # 캐시 키에 model_name 포함 (모델별로 캐시 분리)
    cache_options = {**options}
    if model_name:
        cache_options["_model_name"] = model_name

    # 캐시 확인 (use_cache 옵션으로 끄기 가능)
    use_cache = options.get("use_cache", True)
    cache = get_cache() if use_cache else None
    if cache is not None:
        cached = cache.get(req.query, cache_options)
        if cached is not None:
            cached["_cache_hit"] = True
            cached["_model_name"] = model_name or "(default)"
            return cached

    # LLM 클라이언트를 모델별로 가져오기 (LoRA 모델 선택 지원)
    if model_name:
        from steam_part_d.llm.client import get_llm_client as _get_client
        from steam_part_d.llm.client import reset_llm_client

        # 모델 변경 시 캐시된 클라이언트 무효화 (모델명 다른 경우만)
        try:
            current = _get_client()
            if getattr(current, "_cache_key", None) != model_name:
                reset_llm_client()
        except Exception:
            reset_llm_client()
        _get_client(model_name)

    try:
        if use_agent:
            # Minimal Agent 경로 (2026-07-07 추가)
            agent_result = get_agent().recommend(
                db,
                req.query,
                top_k=top_k,
                enable_semantic_rerank=enable_rerank,
                enable_mmr=enable_mmr,
                language=language,
            )
            result = agent_result["response"]
            agent_trace = agent_result["agent_trace"]
            agent_strategy = agent_result["strategy"]
        else:
            # 기존 pipeline 직접 호출 (기본)
            result = get_pipeline().recommend(
                db,
                req.query,
                top_k=top_k,
                enable_semantic_rerank=enable_rerank,
                enable_mmr=enable_mmr,
                language=language,
            )
            agent_trace = None
            agent_strategy = None
    except Exception as e:  # noqa: BLE001 — API boundary
        # [2026-07-21 배포 P0] SQL/예외 문자열을 client에 노출하지 않고, 일반화된 메시지만 반환
        # 상세는 로그에만 기록 (운영 디버깅용)
        logger.exception("recommend_failed", query=req.query, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "추천 생성 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            },
        ) from e

    if result.retrieval_stats.sql_candidates == 0:
        raise HTTPException(
            status_code=404,
            detail="조건과 일치하는 게임을 찾지 못했습니다 (Constraint 추출 또는 SQL filter 결과 0건 — Self Reflection으로 자동 relaxation 시도 가능)",
        )

    body = result.model_dump()
    body["_model_name"] = model_name or (
        "lora" if get_settings().llm.lora_enabled else "(default)"
    )
    body["_cache_hit"] = False  # 이 경로로 도달 = 캐시 미스
    # Agent trace (2026-07-07 추가)
    if agent_trace is not None:
        body["agent_trace"] = agent_trace
        body["agent_strategy"] = agent_strategy

    # Reasoning Trace (2026-07-07 추가, SHOW_TRACE=true일 때만 포함)
    if not get_settings().api.show_trace:
        body.pop("trace", None)

    # 캐시 저장
    if cache is not None:
        cache.set(req.query, cache_options, body)

    return body


@router.get("/cache/stats")
def cache_stats() -> dict:
    """캐시 통계 (디버그 / 모니터링용)."""
    return get_cache().stats()


@router.post("/cache/clear")
def cache_clear() -> dict:
    """캐시 비우기."""
    get_cache().clear()
    return {"status": "cleared"}


# ── [2026-07-20] 통합 명세 대응 — GET /search ────────────────────────────────

def format_note(constraint: dict) -> str:
    """LoRA가 추출한 조건을 요약하여 A 파트 note 형식으로 반환."""
    parts: list[str] = []
    # 가격
    if constraint.get("max_price_usd") is not None:
        v = constraint["max_price_usd"]
        if v == 0:
            parts.append("무료")
        else:
            # KRW 환산 (1300원 기준)
            krw = int(v * 1300)
            # 5천원 단위로 반올림하여 사용자 친화적 표시 (예: 19994 -> 20000)
            rounded_krw = round(krw / 5000) * 5000
            if rounded_krw >= 10000:
                man = rounded_krw // 10000
                cheon = (rounded_krw % 10000) // 1000
                if cheon > 0:
                    parts.append(f"가격 {man}만 {cheon}천원 이하")
                else:
                    parts.append(f"가격 {man}만원 이하")
            else:
                parts.append(f"가격 {rounded_krw}원 이하")
    elif constraint.get("free_only"):
        parts.append("무료")

    # 장르
    if constraint.get("genres"):
        parts.append(f"{', '.join(constraint['genres'])} 장르")
    # 카테고리
    if constraint.get("categories"):
        parts.append(f"{', '.join(constraint['categories'])}")
    # 인원수
    if constraint.get("min_players") is not None:
        parts.append(f"{constraint['min_players']}인 이상")
    # 플랫폼
    if constraint.get("platforms"):
        parts.append(f"{', '.join(constraint['platforms'])} 지원")
    # 자유 텍스트
    if constraint.get("free_text"):
        parts.append(f"'{constraint['free_text']}'")
    # must_not (제외)
    if constraint.get("exclude_genres"):
        parts.append(f"{', '.join(constraint['exclude_genres'])} 장르 제외")
    if constraint.get("exclude_categories"):
        parts.append(f"{', '.join(constraint['exclude_categories'])} 카테고리 제외")
    if constraint.get("exclude_platforms"):
        parts.append(f"{', '.join(constraint['exclude_platforms'])} 미지원 제외")
    if constraint.get("exclude_free_only"):
        parts.append("무료 게임 제외")

    if not parts:
        return "조건에 맞는 게임 없음"

    return f"추출 조건: {' / '.join(parts)}"


def _has_batchim(ch: str) -> bool:
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return False
    return code % 28 != 0


def _josa_eul_reul(word: str) -> str:
    return "을" if _has_batchim(word[-1]) else "를"


def _josa_wa_gwa(word: str) -> str:
    return "과" if _has_batchim(word[-1]) else "와"


_GENRE_MOOD_MAP = [
    (["horror"], "공포 분위기"),
    (["rpg"], "캐릭터 성장과 서사"),
    (["simulation"], "현실감 있는 시뮬레이션 경험"),
    (["strategy"], "치밀한 전략적 사고가 필요한 플레이"),
    (["puzzle"], "차근차근 풀어가는 퍼즐 경험"),
    (["action"], "긴장감 있는 액션 플레이"),
    (["adventure"], "탐험하며 이야기를 즐기는 어드벤처"),
    (["indie"], "개성 있는 인디 게임 특유의 매력"),
]


def build_reason(g) -> str:
    """[신규] DB metadata 기반 자연스러운 한국어 reason (나열 금지, 1문장)."""
    explanation = getattr(g, "explanation", None)
    if explanation:
        return explanation

    # 1) 가격/무료 조건절
    if getattr(g, "is_free", False):
        condition = "무료로 즐길 수 있으며"
    elif getattr(g, "price_usd", None) is not None:
        krw = int(g.price_usd * 1300)
        man = krw // 10000
        condition = f"약 {man}만원대 가격이며" if man > 0 else f"약 {krw}원대 가격이며"
    else:
        condition = None

    # 2) 멀티/솔로 + 장르 분위기를 하나의 경험절로 결합
    cats = getattr(g, "categories", None) or []
    genres_lower = [x.lower() for x in (getattr(g, "genres", None) or [])]

    experience_parts = []
    if any(c in cats for c in ("Multi-player", "Co-op", "Online Co-op", "Local Co-op", "Cross-Platform Multiplayer", "PvP")):
        experience_parts.append("다른 유저와 함께하는 멀티플레이 경험")
    elif "Single-player" in cats:
        experience_parts.append("혼자서 편하게 즐기는 플레이")

    for keywords, phrase in _GENRE_MOOD_MAP:
        if any(any(k in gl for k in keywords) for gl in genres_lower):
            experience_parts.append(phrase)
            break  # 대표 분위기 1개만

    experience = None
    if experience_parts:
        if len(experience_parts) == 2:
            joined = f"{experience_parts[0]}{_josa_wa_gwa(experience_parts[0])} {experience_parts[1]}"
        else:
            joined = " · ".join(experience_parts)
        experience = f"{joined}{_josa_eul_reul(joined)} 즐길 수 있는"

    plats = [p for p in (getattr(g, "platforms", None) or []) if p]
    platform_note = " 다양한 플랫폼을 지원합니다." if plats else ""

    # 3) 최종 조립
    if condition and experience:
        sentence = f"{condition} {experience} 게임입니다."
    elif condition:
        # 경험절이 없을 경우 자연스럽게 마무리
        sentence = condition.replace("있으며", "있는") + " 게임입니다."
    elif experience:
        sentence = f"{experience} 게임입니다."
    else:
        sentence = "추천 조건에 부합하는 게임입니다."

    return sentence + platform_note


def format_search_response(
    query: str,
    k: int,
    db: Session,
) -> dict:
    """[2026-07-20] 통합 명세 — 공통 search response builder.

    GET /parts/d/search 와 root /search 양쪽에서 사용.
    명세 응답 형식:
    {
      "results": [{"appid", "name", "score", "reason"}],
      "note": "추출 조건 요약"
    }

    Args:
        query: 자연어 query
        k: 최대 결과 수 (1~30)
        db: SQLAlchemy Session

    Returns:
        명세 형식 dict. 결과 없으면 200 + empty results.
    """
    if not query or not query.strip():
        return {"results": [], "note": "조건에 맞는 게임 없음"}

    try:
        result = get_pipeline().recommend(
            db, query, top_k=k, language="ko",
        )
    except Exception as e:  # noqa: BLE001 — API boundary
        # [2026-07-21 배포 P0] SQL/예외 문자열을 client에 노출하지 않고 일반화된 메시지만 반환
        logger.exception("search_failed", query=query, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "검색 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            },
        ) from e

    # 0 candidate → 명세: HTTP 200 + empty results
    if result.retrieval_stats.sql_candidates == 0:
        return {
            "results": [],
            "note": "조건에 맞는 게임 없음",
        }

    constraint = result.constraint or {}
    note = format_note(constraint)

    # [2026-07-21 배포 P0] score desc 정렬 + None은 마지막으로
    # (pipeline이 입력 순서 보존하지만, score 명시적 정렬로 일관성 보장)
    sorted_games = sorted(
        result.recommended_games,
        key=lambda g: (
            g.semantic_score is None,        # None은 마지막
            -(g.semantic_score or 0.0),      # score desc
        ),
    )

    results: list[dict] = []
    for g in sorted_games:
        # [2026-07-20] score = semantic_score (BGE-M3 cosine 0~1) 우선. 없으면 0.5 fallback
        # (vector_search path는 semantic_rerank skip → None. SQL fallback path는 항상 채워짐)
        score = g.semantic_score if g.semantic_score is not None else 0.5
        
        # [2026-07-24] A 파트 명세: 자연어 문장 기반 reason 우선 사용
        # explanation이 있으면(LLM 생성) 그것을 사용, 없으면 build_reason(DB 기반) 사용
        final_reason = getattr(g, "explanation", "") or build_reason(g)
        
        results.append(
            {
                "appid": g.app_id,
                "name": g.name,
                "score": round(score, 4),
                "reason": final_reason,
            }
        )

    return {"results": results, "note": note}


@router.get("/search")
def search(
    q: str = Query(..., min_length=1, description="자연어 query (필수)"),
    k: int = Query(30, ge=1, le=30, description="최대 결과 수 (1~30)"),
    uid: str | None = Query(None, description="익명 user ID (D는 무시 가능)"),
    db: Session = Depends(get_db),
) -> dict:
    """[2026-07-20] 통합 명세 — GET /parts/d/search

    A 게이트웨이가 호출하는 표준 endpoint. 명세 응답 형식.
    """
    return format_search_response(q, k, db)


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    from steam_part_d import __version__ as version

    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:  # noqa: BLE001
        db_connected = False

    model_loaded = True
    lora_adapter_loaded = False
    lora_adapter_name = None
    lora_config_enabled = False

    try:
        settings = get_settings()
        lora_config_enabled = settings.llm.lora_enabled
        client = get_llm_client()

        if lora_config_enabled:
            lora_adapter_name = settings.llm.model_name

            # Ollama인 경우 실제 모델 등록 여부 확인
            from steam_part_d.llm.client import OpenAICompatibleClient
            if isinstance(client, OpenAICompatibleClient):
                base = client.base_url
                # Ollama 감지: 포트 11434 또는 URL에 ollama 포함
                is_ollama = ":11434" in base or "ollama" in base.lower()

                if is_ollama:
                    try:
                        import httpx
                        # /v1/chat/completions -> /api/tags
                        tags_url = base.replace("/v1", "/api/tags")
                        resp = httpx.get(tags_url, timeout=2)
                        resp.raise_for_status()
                        models = [m["name"] for m in resp.json().get("models", [])]
                        # Ollama는 model:latest 또는 model 형식이므로 둘 다 체크하거나 정확히 매칭
                        target = settings.llm.model_name
                        lora_adapter_loaded = target in models or f"{target}:latest" in models
                    except Exception:
                        # Ollama 확인 실패(네트워크 등) → 설정값 기준 fallback (긍정)
                        lora_adapter_loaded = True
                else:
                    # Ollama 아님 (vLLM, TGI 등) → 설정값 기준
                    lora_adapter_loaded = True
            else:
                # MLXLoRAClient 등 기타 클라이언트
                lora_adapter_loaded = True

    except Exception:  # noqa: BLE001
        model_loaded = False

    # Vector index warmup 상태 확인
    from steam_part_d.retrieval.vector_search import is_index_ready
    vector_index_ready = is_index_ready()

    # status 우선순위: degraded > warming > ok
    settings = get_settings()
    if not db_connected or not model_loaded:
        status = "degraded"
    elif not vector_index_ready and settings.retrieval.enable_vector_search:
        status = "warming"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        db_connected=db_connected,
        vector_index_ready=vector_index_ready,
        lora_adapter_loaded=lora_adapter_loaded,
        lora_adapter_name=lora_adapter_name,
        lora_config_enabled=lora_config_enabled,
        version=version,
    )
