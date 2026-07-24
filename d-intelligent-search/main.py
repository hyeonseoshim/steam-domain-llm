"""FastAPI app entry point.

실행:
    uvicorn steam_part_d.main:app --host 0.0.0.0 --port 8004
또는:
    python -m steam_part_d.main
"""
from __future__ import annotations

import uvicorn

from steam_part_d.api.routes import router, format_search_response, health
from steam_part_d.config import get_settings
from steam_part_d.utils.logging import configure_logging, get_logger


def create_app():
    from fastapi import FastAPI, Query, Depends
    from sqlalchemy.orm import Session
    from steam_part_d.db.session import get_db

    settings = get_settings()
    configure_logging(settings.app.log_level)
    logger = get_logger(__name__)

    app = FastAPI(
        title="Steam Part D — 조건 기반 게임 추천",
        version=settings.app.version,
        description=(
            "자연어 조건 → SQL Hard Filter → Semantic Rerank → MMR/Bucket → LLM Explanation. "
            "LLM은 추천 이유만 생성, 게임 목록은 코드가 결정 (grounding 보장)."
        ),
    )
    app.include_router(router)

    # [2026-07-21] D 탭 CORS — A 게이트웨이의 브라우저 fetch 허용
    # 운영: 환경변수 CORS_ALLOW_ORIGINS에 명시적 origin list (comma-separated).
    # 개발: 환경변수 비어있으면 명시적 dev default (localhost only). "*"는 명시적 opt-in만 허용.
    # 기본값은 와일드카드("*)가 아니므로, 환경변수 미설정 시 외부 도메인 호출은 거부됨.
    import os as _os
    from fastapi.middleware.cors import CORSMiddleware
    _cors_origins_env = _os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
    if not _cors_origins_env:
        # env 미설정 — dev default (localhost only). 운영에서는 반드시 env로 명시.
        _cors_origins = ["http://localhost:3000", "http://localhost:8004"]
        logger.info("cors_origins_default_dev", origins=_cors_origins)
    elif _cors_origins_env == "*":
        # 명시적 * (dev/test only). 운영 부적합 — 보안 경고.
        logger.warning("cors_allow_all_insecure", message="CORS_ALLOW_ORIGINS=* 사용, 운영 부적합")
        _cors_origins = ["*"]
    else:
        _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
        logger.info("cors_origins_configured", count=len(_cors_origins), origins=_cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict:
        return {
            "service": settings.app.name,
            "version": settings.app.version,
            "endpoints": [
                "POST /parts/d/recommend",
                "GET /parts/d/search",
                "GET /search",
                "GET /parts/d/health",
                "GET /health",
            ],
        }

    # [2026-07-20] 통합 명세 — root path alias
    # A 게이트웨이가 GET /search (path prefix 없음) 호출 가능하도록.
    @app.get("/search")
    def root_search(
        q: str = Query(..., min_length=1, description="자연어 query (필수)"),
        k: int = Query(30, ge=1, le=30, description="최대 결과 수 (1~30)"),
        uid: str | None = Query(None, description="익명 user ID (D는 무시 가능)"),
        db: Session = Depends(get_db),
    ) -> dict:
        return format_search_response(q, k, db)

    @app.get("/health")
    def root_health(db: Session = Depends(get_db)) -> dict:
        """[2026-07-20] 통합 명세 — root path /health alias"""
        return health(db).model_dump()

    logger.info("app_created", env=settings.app.env, port=settings.api.port)

    # Vector index warmup (서버 부팅 시 1회, ~46초)
    # warmup 중에는 /health가 "warming" 반환 → k8s readiness가 트래픽 차단
    if settings.retrieval.enable_vector_search:
        from steam_part_d.retrieval.vector_search import get_vector_index
        logger.info("vector_index_warmup_starting")
        get_vector_index()
        logger.info("vector_index_warmed_up")

    return app


app = create_app()


def run() -> None:
    """python -m steam_part_d.main 또는 steam-part-d 스크립트로 실행."""
    settings = get_settings()
    uvicorn.run(
        "steam_part_d.main:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.app.log_level.lower(),
    )


if __name__ == "__main__":
    run()
