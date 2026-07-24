"""Empty Recommendation 폴백 로직 — 4단계 relaxed retrieval.

목적
----
2026-07-23 baseline 평가에서 전체 추천 쿼리의 **48.9%가 빈 결과**로 회귀.
운영에서 "결과 없음"은 신뢰를 크게 떨어뜨리므로, 단계적으로 조건을 완화해
가능한 한 사용자에게 게임을 돌려주는 폴백 체계를 제공.

4단계
----
1. **strict**  (level=1)
   - 입력 conditions 전체를 AND로 적용
   - 사용자가 명시한 모든 조건 만족
2. **relaxed**  (level=2)
   - 핵심 조건만 AND: ``max_price_usd`` + ``genres`` + ``categories``
   - ``platforms``/``min_players``/``free_text``/``exclude_*`` 제거
3. **minimal**  (level=3)
   - 가격/무료 여부만: ``max_price_usd`` + ``free_only``
4. **popular**  (level=4)
   - 조건 없이 인기 게임 Top K (metacritic + recommendations_total 정렬)
   - explanation 첨부: "정확한 조건에 맞는 게임이 없어 인기 게임을 추천합니다."

설계 결정
---------
- **순수 함수** (input dict → output dict). DB/Session을 직접 다루지 않음.
- ``sql_filter_fn`` 파라미터로 SQL 호출을 주입받아 **테스트는 mock으로 가능**.
- ``sql_filter_fn`` 이 ``None`` 이면 lazy import 로 ``apply_sql_filter`` + ``session_scope`` 사용.
- ``conditions`` dict 의 ``max_price`` alias 는 ``max_price_usd`` 로 자동 변환
  (사용자/외부 입력 호환용).
- GameConstraint 가 ``extra="forbid"`` 이므로 GameConstraint 로 매핑 가능한
  키만 통과 ( ``exclude_*``, ``min_players`` 등도 그대로 보존 — strict 단계).
- 결과는 ``.results[k]`` 슬라이스 — sql_filter_fn 이 limit 보다 많이
  반환해도 호출자 k 개만 노출.

사용 예
------
운영에서 호출::

    from steam_part_d.retrieval.fallback import get_recommendations_with_fallback

    result = get_recommendations_with_fallback(
        conditions={
            "max_price_usd": 30,
            "genres": ["RPG"],
            "platforms": ["windows", "mac"],
            "min_players": 4,
        },
        k=10,
    )
    # result["fallback_level"] ∈ {1, 2, 3, 4}
    # result["level_name"]    ∈ {"strict", "relaxed", "minimal", "popular"}
    # result["results"]       list of game dicts (최대 k 개)
    # result["explanation"]   None 또는 한국어 안내문

테스트에서 mock 사용::

    from unittest.mock import MagicMock
    from steam_part_d.retrieval.fallback import get_recommendations_with_fallback

    mock_sql = MagicMock()
    mock_sql.side_effect = lambda c, m: [{"appid": 1}] if m == "strict" else []
    result = get_recommendations_with_fallback(
        {"max_price_usd": 30}, k=10, sql_filter_fn=mock_sql
    )

Import 가이드
-------------
운영 (DB 자동)::

    from steam_part_d.retrieval.fallback import get_recommendations_with_fallback

테스트 (mock 주입)::

    from steam_part_d.retrieval.fallback import (
        get_recommendations_with_fallback,
        _normalize_conditions,         # internal helper
        _RELAXED_KEYS, _MINIMAL_KEYS,  # 정책 튜닝 시
    )

    from steam_part_d.models.constraint import GameConstraint
    from steam_part_d.retrieval.sql_filter import apply_sql_filter, SQLFilterConfig
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 4단계 정책
# ---------------------------------------------------------------------------

# 2단계 (relaxed) 에서 보존할 키 — 가격 + 장르 + 카테고리 (사용자 spec)
RELAXED_KEYS: tuple[str, ...] = ("max_price_usd", "genres", "categories")

# 3단계 (minimal) 에서 보존할 키 — 가격/무료 여부만
MINIMAL_KEYS: tuple[str, ...] = ("max_price_usd", "free_only")

# 4단계 (popular) 는 빈 conditions + mode="popular" 으로 sql_filter 호출

# dict 입력 → GameConstraint 필드명 alias (사용자 spec에 max_price 표기)
_ALIAS_MAP: dict[str, str] = {
    "max_price": "max_price_usd",
}

# sql_filter_fn 시그니처: (conditions_dict, mode_str) -> list[game_dict]
SqlFilterFn = Callable[[dict[str, Any], str], list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _normalize_conditions(conditions: dict[str, Any]) -> dict[str, Any]:
    """dict → GameConstraint 입력용 dict 로 정규화.

    - ``max_price`` → ``max_price_usd`` alias 적용
    - ``None`` / 빈 문자열 / 빈 리스트 값 제거 (조건 없는 키는 무시)
    - 모르는 키는 그대로 보존 (strict 단계에서 GameConstraint 가 reject;
      relaxed/minimal 단계는 key whitelist 로 거른 뒤 사용)
    """
    out: dict[str, Any] = {}
    for k, v in conditions.items():
        if v is None:
            continue
        if isinstance(v, (list, str)) and len(v) == 0:
            continue
        out[_ALIAS_MAP.get(k, k)] = v
    return out


def _pick_keys(conditions: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """whitelist 키만 골라 새 dict 생성."""
    return {k: conditions[k] for k in keys if k in conditions}


def _default_sql_filter(
    conditions: dict[str, Any], mode: str, *, limit: int
) -> list[dict[str, Any]]:
    """운영용 sql_filter — ``apply_sql_filter`` + ``GameConstraint`` + ``session_scope``.

    mode:
      - ``"strict"`` / ``"relaxed"`` / ``"minimal"`` : 입력 dict 그대로 GameConstraint 매핑
      - ``"popular"`` : 빈 GameConstraint + metacritic+recommendations 정렬
    """
    # lazy import — 이 모듈을 import 만 해도 DB 가 열리지 않도록
    from steam_part_d.db.session import session_scope
    from steam_part_d.models.constraint import GameConstraint
    from steam_part_d.retrieval.sql_filter import SQLFilterConfig, apply_sql_filter

    with session_scope() as session:
        if mode == "popular":
            constraint = GameConstraint()
            config = SQLFilterConfig(candidate_limit=limit)
        else:
            # strict 단계에서 GameConstraint 가 모르는 키를 만나면 fail.
            # 운영에서는 GameConstraint 가 받아들이는 키만 들어옴 (max_price_usd,
            # free_only, genres, categories, min_players, platforms, free_text,
            # exclude_*). 모르는 키가 있으면 logger.warning 으로 알리고 무시.
            try:
                constraint = GameConstraint(**conditions)
            except Exception as exc:  # pragma: no cover — 방어용
                logger.warning(
                    "fallback_strict_invalid_conditions",
                    conditions=conditions,
                    error=str(exc),
                )
                constraint = GameConstraint()
            config = SQLFilterConfig(candidate_limit=limit)

        games = apply_sql_filter(session, constraint, config)
        return [g.model_dump() for g in games]


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------

def get_recommendations_with_fallback(
    conditions: dict[str, Any],
    k: int = 10,
    *,
    sql_filter_fn: SqlFilterFn | None = None,
    candidate_limit: int = 200,
) -> dict[str, Any]:
    """4단계 폴백으로 추천 결과를 반환.

    Args:
        conditions: 사용자 조건 dict. 키 예시::
            {
                "max_price_usd": 30.0,
                "free_only": False,
                "genres": ["RPG", "Action"],
                "categories": ["Single-player"],
                "min_players": 1,
                "platforms": ["windows", "mac"],
                "free_text": "공포 분위기",
            }
            ``max_price`` alias 도 허용 ( ``max_price_usd`` 로 변환).
        k: 최종 반환할 최대 게임 수. default 10.
        sql_filter_fn: ``(conditions_dict, mode) -> list[game_dict]`` 시그니처.
            ``None`` 이면 운영용 ``_default_sql_filter`` 사용 (DB 자동).
        candidate_limit: sql_filter 가 한 번에 가져올 최대 후보 수. 4단계
            ``popular`` 의 인기 게임 정렬 폭을 결정. default 200.

    Returns:
        dict::

            {
                "results":        list[game_dict],      # 최대 k 개
                "explanation":    str | None,           # 4단계에서만 한국어 안내
                "fallback_level": 1 | 2 | 3 | 4,        # int (사용자 spec 최종 정정)
                "level_name":     "strict" | "relaxed" | "minimal" | "popular",
            }

    동작:
        1. ``sql_filter(conditions, "strict")`` → 결과 있으면 level=1 반환
        2. ``sql_filter({price+genres+categories}, "relaxed")`` → level=2
        3. ``sql_filter({price+free_only}, "minimal")`` → level=3
        4. ``sql_filter({}, "popular")`` → level=4 + explanation
    """
    # sql_filter dispatcher — mock/default 주입 + k 캡쳐
    def _dispatch(conds: dict[str, Any], mode: str) -> list[dict[str, Any]]:
        if sql_filter_fn is not None:
            return sql_filter_fn(conds, mode)
        return _default_sql_filter(conds, mode, limit=candidate_limit)

    normalized = _normalize_conditions(conditions)

    # 1단계: strict — 입력 conditions 전체 AND
    if normalized:
        results = _dispatch(normalized, "strict")
        if results:
            return {
                "results": results[:k],
                "explanation": None,
                "fallback_level": 1,
                "level_name": "strict",
            }

    # 2단계: relaxed — 가격 + 장르 + 카테고리
    core = _pick_keys(normalized, RELAXED_KEYS)
    if core:
        results = _dispatch(core, "relaxed")
        if results:
            logger.info(
                "fallback_relaxed",
                kept_keys=sorted(core.keys()),
                n=len(results),
            )
            return {
                "results": results[:k],
                "explanation": None,
                "fallback_level": 2,
                "level_name": "relaxed",
            }

    # 3단계: minimal — 가격 + 무료 여부만
    price_only = _pick_keys(normalized, MINIMAL_KEYS)
    if price_only:
        results = _dispatch(price_only, "minimal")
        if results:
            logger.info(
                "fallback_minimal",
                kept_keys=sorted(price_only.keys()),
                n=len(results),
            )
            return {
                "results": results[:k],
                "explanation": None,
                "fallback_level": 3,
                "level_name": "minimal",
            }

    # 4단계: popular — 인기 게임 Top K
    popular_results = _dispatch({}, "popular")
    logger.info(
        "fallback_popular",
        n=len(popular_results),
        requested_k=k,
    )
    return {
        "results": popular_results[:k],
        "explanation": "정확한 조건에 맞는 게임이 없어 인기 게임을 추천합니다.",
        "fallback_level": 4,
        "level_name": "popular",
    }


__all__ = [
    "get_recommendations_with_fallback",
    "RELAXED_KEYS",
    "MINIMAL_KEYS",
    "SqlFilterFn",
]
