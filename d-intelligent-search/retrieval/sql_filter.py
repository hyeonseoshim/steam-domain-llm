"""SQL Hard Filter — 디자인 doc 섹션 5.1.

원본 doc 대비 개선 사항 (리뷰 반영):
1. platforms 필터 추가 (Windows/Mac/Linux)
2. min_players 필터 추가 (Co-op 카테고리 매칭과 별도로 인원수 조건)
3. genres 필터를 교집합(AND) / 합집합(OR) 모드로 분리
   - 기본은 교집합 ("Action RPG" = 둘 다)
   - 옵션으로 OR 전환 가능
4. 가격 비교 시 NULL 처리 명확화
5. 메타크리틱 점수는 정렬에만 사용 (98% 결측)

SQL 인젝션 방어를 위해 모든 사용자 입력은 SQLAlchemy 파라미터 바인딩 사용.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import and_, func, or_, select, intersect
from sqlalchemy.orm import Session

from steam_part_d.db.models import (
    Application,
    Category,
    Genre,
    application_categories,
    application_genres,
)
from steam_part_d.models.constraint import GameConstraint
from steam_part_d.models.game import Game
from steam_part_d.utils.price import normalize_price, normalize_price_usd

GenreMode = Literal["intersection", "union"]


@dataclass
class SQLFilterConfig:
    candidate_limit: int = 200
    genre_mode: GenreMode = "intersection"  # 기본 AND
    include_cached_embedding: bool = False  # True면 description_embedding도 SELECT


# 카테고리 description → "최소 동시 인원" 추정 휴리스틱.
# Steam 카테고리는 'Multi-player', 'Co-op', 'Online Co-op' 같이 추상적이라
# 정확한 인원 매칭은 어려움. min_players가 명시된 경우에만 동작.
def _category_supports_min_players(cat_descriptions: set[str], min_players: int) -> bool:
    """min_players 조건을 카테고리로 표현 가능한지."""
    if min_players <= 1:
        return True
    if min_players >= 2:
        return bool(
            cat_descriptions
            & {
                "Multi-player",
                "Co-op",
                "Online Co-op",
                "Online Multi-Player",
                "Local Co-op",
                "Shared/Split Screen",
                "Cross-Platform Multiplayer",
            }
        )
    return True


def _category_intersection_subquery(category_names: list[str]):
    """모든 카테고리를 가진 appid 조회 (intersection).

    [2026-07-23 P0 fix] junction table (application_categories + Category.english_name) 대신
    Application.english_categories JSON column을 직접 LIKE 검색.
    이유: junction table 데이터가 불완전 (예: 'Multi-player AND Single-player' junction=0, JSON=40,132).
    """
    if not category_names:
        return None
    patterns = [f"%{n}%" for n in category_names]
    subqueries = []
    for pattern in patterns:
        subquery = (
            select(Application.appid)
            .where(Application.english_categories.ilike(pattern))
            .distinct()
        )
        subqueries.append(subquery)

    if not subqueries:
        return None
    return intersect(*subqueries)


def _category_union_subquery(category_names: list[str]):
    """하나 이상의 카테고리를 가진 appid 조회.

    [2026-07-23 P0 fix] junction table → Application.english_categories JSON column 직접 검색.
    """
    if not category_names:
        return None
    from sqlalchemy import or_

    patterns = [f"%{n}%" for n in category_names]
    ilike_clauses = [Application.english_categories.ilike(p) for p in patterns]
    return (
        select(Application.appid)
        .where(or_(*ilike_clauses))
        .distinct()
    )


def _genre_intersection_subquery(genre_names: list[str]):
    """모든 장르를 가진 appid.

    장르도 변형이 있으므로 LIKE 매칭. 단 intersection은 모든 canonical을 가진 게임만.
    english_name이 NULL인 행(예: id=32 Casual)은 name 컬럼으로 fallback (COALESCE).
    """
    if not genre_names:
        return None
    patterns = [f"%{n}%" for n in genre_names]
    subqueries = []
    for pattern in patterns:
        subquery = (
            select(application_genres.c.appid)
            .join(Genre, Genre.id == application_genres.c.genre_id)
            .where(func.coalesce(Genre.english_name, Genre.name).ilike(pattern))
            .distinct()
        )
        subqueries.append(subquery)
    
    if not subqueries:
        return None
    return intersect(*subqueries)


def _genre_union_subquery(genre_names: list[str]):
    """하나 이상의 장르를 가진 appid.
    english_name이 NULL인 행은 name 컬럼으로 fallback (COALESCE).
    """
    if not genre_names:
        return None
    patterns = [f"%{n}%" for n in genre_names]
    sub = None
    for pattern in patterns:
        subquery = (
            select(application_genres.c.appid)
            .join(Genre, Genre.id == application_genres.c.genre_id)
            .where(func.coalesce(Genre.english_name, Genre.name).ilike(pattern))
            .distinct()
        )
        if sub is None:
            sub = subquery
        else:
            sub = sub.union(subquery)
    return sub


def _platform_intersection_subquery(platform_names: list[str]):
    """모든 지정 플랫폼을 지원하는 appid.

    application_platforms + platforms 테이블 활용.
    """
    if not platform_names:
        return None
    from steam_part_d.db.models import Platform, application_platforms

    return (
        select(application_platforms.c.appid)
        .join(Platform, Platform.id == application_platforms.c.platform_id)
        .where(Platform.name.in_(platform_names))
        .group_by(application_platforms.c.appid)
        .having(func.count(func.distinct(Platform.name)) == len(platform_names))
    )


def _platform_subquery(platforms: list[str]):
    """platforms 배열에 모든 지정 플랫폼을 포함하는 appid."""
    if not platforms:
        return None
    return _platform_intersection_subquery(platforms)


def _build_platform_clause_sqlite(platforms: list[str]) -> str | None:
    """SQLite에서는 junction table 쿼리로 처리 — 위 _platform_subquery 사용."""
    return None  # SQLAlchemy ORM 경로로 처리됨


def _build_platform_clause_postgres(platforms: list[str]) -> str | None:
    """PostgreSQL도 junction table 쿼리 사용."""
    return None


def _parse_embedding(raw: Any) -> list[float] | None:
    """DB에서 읽은 embedding (JSON 문자열 또는 list) → list[float]."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return [float(x) for x in data]
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def apply_sql_filter(
    session: Session,
    constraint: GameConstraint,
    config: SQLFilterConfig | None = None,
) -> list[Game]:
    """Constraint → DB 쿼리 → Game 리스트 반환.

    Returns:
        후보 게임 리스트 (metacritic_score DESC 정렬, 최대 config.candidate_limit개).
    """
    if config is None:
        config = SQLFilterConfig()

    constraint = constraint.normalize()

    # dialect 감지 — SQLite는 array_agg가 없어서 GROUP_CONCAT 사용
    dialect = session.bind.dialect.name if session.bind else "sqlite"

    if dialect == "postgresql":
        genres_agg = func.array_agg(func.distinct(Genre.english_name)).label(
            "genres"
        )
        cats_agg = func.array_agg(func.distinct(Category.english_name)).label(
            "categories"
        )
    else:
        # SQLite (및 기타) — GROUP_CONCAT 후 Python에서 split
        genres_agg = func.group_concat(func.distinct(Genre.english_name)).label(
            "genres"
        )
        cats_agg = func.group_concat(func.distinct(Category.english_name)).label(
            "categories"
        )

    selected_cols: list[Any] = [
        Application.appid,
        Application.name,
        Application.short_description,
        Application.mat_final_price,
        Application.is_free,
        Application.metacritic_score,
        Application.mat_supports_windows,
        Application.mat_supports_mac,
        Application.mat_supports_linux,
        genres_agg,
        cats_agg,
    ]
    group_cols: list[Any] = [
        Application.appid,
        Application.name,
        Application.short_description,
        Application.mat_final_price,
        Application.is_free,
        Application.metacritic_score,
        Application.mat_supports_windows,
        Application.mat_supports_mac,
        Application.mat_supports_linux,
    ]
    if config.include_cached_embedding:
        selected_cols.append(Application.description_embedding)
        # embedding은 JSON 텍스트라 distinct하지 않음 → GROUP BY엔 빼고 MAX/MIN으로 묶음
        # SQLite에서 MAX(text) 가능. distinct할 필요 없으니 단일 값으로 묶음.
        # 여기선 subquery로 해결 — embedding은 appid당 unique하므로 MAX 사용.
        group_cols.append(Application.description_embedding)

    stmt = (
        select(*selected_cols)
        .select_from(Application)
        .outerjoin(application_genres, application_genres.c.appid == Application.appid)
        .outerjoin(Genre, Genre.id == application_genres.c.genre_id)
        .outerjoin(
            application_categories,
            application_categories.c.appid == Application.appid,
        )
        .outerjoin(
            Category, Category.id == application_categories.c.category_id
        )
        .where(Application.type == "game")
        .where((Application.required_age == 0) | (Application.required_age.is_(None)) | (Application.required_age < 18))  # Adult-only (required_age=18) 제외 (1단계 전처리, 204개)
        .group_by(*group_cols)
        # 정렬 (2026-07-07 fix): metacritic + popularity 혼합
        # - 1순위: metacritic_score (DESC, NULL 마지막)
        # - 2순위: recommendations_total (DESC, NULL 마지막) — 인기도 proxy
        # 이유: metacritic만으로는 niche 인기작이 잘림. recommendations로 검증된 인기작도 노출.
        .order_by(
            Application.metacritic_score.desc().nulls_last(),
            Application.recommendations_total.desc().nulls_last(),
        )
        .limit(config.candidate_limit)
    )

    # --- 가격 필터 ---
    if constraint.max_price_usd is not None:
        max_cents = int(constraint.max_price_usd * 100)
        # is_free=True 도 통과, NULL 가격 제외, 지정 가격 이하
        stmt = stmt.where(
            or_(
                Application.is_free.is_(True),
                and_(
                    Application.mat_final_price.is_not(None),
                    Application.mat_final_price <= max_cents,
                ),
            )
        )
    elif constraint.free_only is True:
        stmt = stmt.where(Application.is_free.is_(True))

    # --- 장르 필터 (교집합/합집합) ---
    if constraint.genres:
        sub = (
            _genre_intersection_subquery(constraint.genres)
            if config.genre_mode == "intersection"
            else _genre_union_subquery(constraint.genres)
        )
        if sub is not None:
            stmt = stmt.where(Application.appid.in_(sub))

    # --- 카테고리 필터 (기본 교집합) ---
    if constraint.categories:
        sub = _category_intersection_subquery(constraint.categories)
        if sub is not None:
            stmt = stmt.where(Application.appid.in_(sub))

    # --- min_players 휴리스틱 (카테고리 합집합으로 검증) ---
    if constraint.min_players is not None and constraint.min_players >= 2:
        # 명시적으로 협동/멀티 카테고리 중 하나 이상 가진 게임만 통과
        coop_categories = [
            "Multi-player",
            "Co-op",
            "Online Co-op",
            "Online Multi-Player",
            "Local Co-op",
            "Shared/Split Screen",
            "Cross-Platform Multiplayer",
        ]
        sub = _category_union_subquery(coop_categories)
        if sub is not None:
            stmt = stmt.where(Application.appid.in_(sub))

    # --- platforms 필터 (junction table 경로) ---
    if constraint.platforms:
        sub = _platform_intersection_subquery(constraint.platforms)
        if sub is not None:
            stmt = stmt.where(Application.appid.in_(sub))

    # --- 실행 ---
    rows = session.execute(stmt).all()

    games: list[Game] = []
    for row in rows:
        genres_raw = row.genres or []
        categories_raw = row.categories or []

        # SQLite의 GROUP_CONCAT은 콤마 구분 문자열로 반환
        if dialect != "postgresql":
            if isinstance(genres_raw, str):
                genres_raw = [g.strip() for g in genres_raw.split(",") if g.strip()]
            if isinstance(categories_raw, str):
                categories_raw = [c.strip() for c in categories_raw.split(",") if c.strip()]

        # None 항목 필터
        genres = [g for g in genres_raw if g]
        categories = [c for c in categories_raw if c]

        # mat_supports_* boolean → platforms 리스트
        platforms: list[str] = []
        if getattr(row, "mat_supports_windows", False):
            platforms.append("windows")
        if getattr(row, "mat_supports_mac", False):
            platforms.append("mac")
        if getattr(row, "mat_supports_linux", False):
            platforms.append("linux")

        games.append(
            Game(
                appid=row.appid,
                name=row.name or "",
                short_description=row.short_description,
                price_usd=normalize_price(
                    mat_final_price=row.mat_final_price,
                    currency=getattr(row, 'mat_currency', None),
                    is_free=row.is_free,
                ),
                is_free=row.is_free,
                metacritic_score=row.metacritic_score,
                genres=genres,
                categories=categories,
                # [2026-07-21 배포 P0] platforms 필드 전달 — apply_exclude에서 사용
                platforms=platforms,
                description_embedding=(
                    _parse_embedding(getattr(row, "description_embedding", None))
                    if config.include_cached_embedding
                    else None
                ),
            )
        )

    return games


def apply_sql_filter_by_appids(
    session: Session,
    appids: list[int],
    constraint: GameConstraint,
    config: SQLFilterConfig | None = None,
) -> list[Game]:
    """Vector search 결과 appid 목록에 SQL hard filter 적용.

    apply_sql_filter()와 달리:
    - candidate_limit 없음 (appids 전체 필터링)
    - ORDER BY 없음 (입력 appids 순서 = 벡터 유사도 순 보존)
    - metacritic/recommendations 정렬 없음 (벡터 유사도가 이미 관련성 기준)

    Args:
        appids: vector_search()가 반환한 appid 목록 (유사도 높은 순)
        constraint: 가격/장르/카테고리/플랫폼 조건
        config: 장르 모드 등 부가 설정

    Returns:
        constraint를 만족하는 게임 리스트, 입력 appids 순서 유지.
    """
    if not appids:
        return []
    if config is None:
        config = SQLFilterConfig()

    constraint = constraint.normalize()
    dialect = session.bind.dialect.name if session.bind else "sqlite"

    if dialect == "postgresql":
        genres_agg = func.array_agg(func.distinct(Genre.english_name)).label("genres")
        cats_agg = func.array_agg(func.distinct(Category.english_name)).label("categories")
    else:
        genres_agg = func.group_concat(func.distinct(Genre.english_name)).label("genres")
        cats_agg = func.group_concat(func.distinct(Category.english_name)).label("categories")

    selected_cols: list[Any] = [
        Application.appid,
        Application.name,
        Application.short_description,
        Application.mat_final_price,
        Application.is_free,
        Application.metacritic_score,
        Application.mat_supports_windows,
        Application.mat_supports_mac,
        Application.mat_supports_linux,
        genres_agg,
        cats_agg,
    ]
    group_cols: list[Any] = [
        Application.appid,
        Application.name,
        Application.short_description,
        Application.mat_final_price,
        Application.is_free,
        Application.metacritic_score,
        Application.mat_supports_windows,
        Application.mat_supports_mac,
        Application.mat_supports_linux,
    ]

    stmt = (
        select(*selected_cols)
        .select_from(Application)
        .outerjoin(application_genres, application_genres.c.appid == Application.appid)
        .outerjoin(Genre, Genre.id == application_genres.c.genre_id)
        .outerjoin(application_categories, application_categories.c.appid == Application.appid)
        .outerjoin(Category, Category.id == application_categories.c.category_id)
        .where(Application.type == "game")
        .where((Application.required_age == 0) | (Application.required_age.is_(None)) | (Application.required_age < 18))  # Adult-only (required_age=18) 제외 (1단계 전처리, 204개)
        .where(Application.appid.in_(appids))  # vector search 결과만
        .group_by(*group_cols)
        # ORDER BY 없음 — Python에서 입력 순서 복원
    )

    # --- 가격 필터 ---
    if constraint.max_price_usd is not None:
        max_cents = int(constraint.max_price_usd * 100)
        stmt = stmt.where(
            or_(
                Application.is_free.is_(True),
                and_(
                    Application.mat_final_price.is_not(None),
                    Application.mat_final_price <= max_cents,
                ),
            )
        )
    elif constraint.free_only is True:
        stmt = stmt.where(Application.is_free.is_(True))

    # --- 장르 필터 ---
    if constraint.genres:
        sub = (
            _genre_intersection_subquery(constraint.genres)
            if config.genre_mode == "intersection"
            else _genre_union_subquery(constraint.genres)
        )
        if sub is not None:
            stmt = stmt.where(Application.appid.in_(sub))

    # --- 카테고리 필터 ---
    if constraint.categories:
        sub = _category_intersection_subquery(constraint.categories)
        if sub is not None:
            stmt = stmt.where(Application.appid.in_(sub))

    # --- min_players ---
    if constraint.min_players is not None and constraint.min_players >= 2:
        coop_categories = [
            "Multi-player", "Co-op", "Online Co-op", "Online Multi-Player",
            "Local Co-op", "Shared/Split Screen", "Cross-Platform Multiplayer",
        ]
        sub = _category_union_subquery(coop_categories)
        if sub is not None:
            stmt = stmt.where(Application.appid.in_(sub))

    # --- platforms 필터 ---
    if constraint.platforms:
        sub = _platform_intersection_subquery(constraint.platforms)
        if sub is not None:
            stmt = stmt.where(Application.appid.in_(sub))

    rows = session.execute(stmt).all()

    # 결과를 dict로 변환
    game_map: dict[int, Game] = {}
    for row in rows:
        genres_raw = row.genres or []
        categories_raw = row.categories or []

        if dialect != "postgresql":
            if isinstance(genres_raw, str):
                genres_raw = [g.strip() for g in genres_raw.split(",") if g.strip()]
            if isinstance(categories_raw, str):
                categories_raw = [c.strip() for c in categories_raw.split(",") if c.strip()]

        genres = [g for g in genres_raw if g]
        categories = [c for c in categories_raw if c]

        platforms: list[str] = []
        if getattr(row, "mat_supports_windows", False):
            platforms.append("windows")
        if getattr(row, "mat_supports_mac", False):
            platforms.append("mac")
        if getattr(row, "mat_supports_linux", False):
            platforms.append("linux")

        game_map[row.appid] = Game(
            appid=row.appid,
            name=row.name or "",
            short_description=row.short_description,
            price_usd=normalize_price(
                mat_final_price=row.mat_final_price,
                currency=getattr(row, 'mat_currency', None),
                is_free=row.is_free,
            ),
            is_free=row.is_free,
            metacritic_score=row.metacritic_score,
            genres=genres,
            categories=categories,
            # [2026-07-21 배포 P0] platforms 필드 전달 — apply_exclude에서 사용
            platforms=platforms,
        )

    # 입력 appids 순서로 정렬 (벡터 유사도 순 복원)
    return [game_map[aid] for aid in appids if aid in game_map]


def apply_exclude(
    games: list[Game],
    constraint: GameConstraint,
) -> list[Game]:
    """[2026-07-21 배포 P0] must_not/exclude 조건 적용 (Python post-filter).

    SQL LIKE 매칭을 그대로 사용 (sql_filter._category_intersection_subquery와 동일).
    exclude_categories=["Co-op"] → "Online Co-op", "Local Co-op", "Co-op LAN" 등 매칭.

    Args:
        games: SQL filter 후 후보 게임 리스트
        constraint: GameConstraint (exclude_* 필드 사용)

    Returns:
        exclude 조건 통과한 게임 리스트. 입력 순서 유지.
    """
    if not any([
        constraint.exclude_genres,
        constraint.exclude_categories,
        constraint.exclude_platforms,
        constraint.exclude_free_only,
    ]):
        return games

    out: list[Game] = []
    for g in games:
        # exclude_genres: 장르 1개라도 매칭하면 제외
        if constraint.exclude_genres:
            if any(ex in (g.genres or []) for ex in constraint.exclude_genres):
                continue
        # exclude_categories: LIKE 매칭 (substring)
        if constraint.exclude_categories:
            cats_text = " ".join(g.categories or [])
            if any(ex in cats_text for ex in constraint.exclude_categories):
                continue
        # exclude_platforms: 모든 지정 플랫폼 미지원이면 OK, 1개라도 지원이면 제외
        # [2026-07-21 hotfix] Game 모델에 platforms 필드 없을 수 있어 getattr fallback
        if constraint.exclude_platforms:
            g_platforms = getattr(g, "platforms", None) or []
            if any(p in g_platforms for p in constraint.exclude_platforms):
                # 제외 플랫폼 중 1개라도 지원 = 제외
                continue
        # exclude_free_only: 무료 게임 제외 (유료만)
        if constraint.exclude_free_only and g.is_free:
            continue
        out.append(g)
    return out
