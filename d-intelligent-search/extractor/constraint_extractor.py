"""Constraint Extractor — 디자인 doc 섹션 4.

자연어 → JSON → GameConstraint 변환.

핵심: 3회 재시도 + JSON 스키마 검증 + fallback.
LLM 응답이 마크다운 펜스나 추가 텍스트를 포함해도 첫 번째 JSON 블록을 파싱 시도.
"""
from __future__ import annotations

import json
import re

import jsonschema
from jsonschema import ValidationError

from steam_part_d.config import ExtractorSettings, get_settings, load_prompts
from steam_part_d.llm.client import BaseLLMClient, LLMError, get_llm_client
from steam_part_d.models.constraint import (
    CONSTRAINT_JSON_SCHEMA,
    GameConstraint,
    fallback_constraint,
)
from steam_part_d.utils.logging import get_logger
from steam_part_d.utils.synonyms import (
    VALID_STEAM_GENRES,
    partition_genres,
    translate_constraint_categories,
    translate_constraint_genres,
    translate_genre,
)

logger = get_logger(__name__)


# [2026-07-23 P0 fix] Steam 카테고리 화이트리스트
# LLM이 "Story", "Horror", "Atmospheric" 등 Steam TAGS를 categories로 잘못 출력하는 버그 수정.
# 이 값들은 english_categories에 0건 매칭 → SQL hard filter에서 모든 결과를 죽임.
# game_evidence.json의 12개 features와 정렬 (multiplayer/coop/singleplayer 등만 categories에 허용).
VALID_STEAM_CATEGORIES: frozenset[str] = frozenset({
    # 게임 모드 (영어)
    "Single-player", "Multi-player", "Co-op", "Co-op LAN", "Co-op Online",
    "Online Co-op", "Online Multi-Player", "Local Co-op", "Local Multi-Player",
    "Shared/Split Screen", "Shared/Split Screen Co-op", "Shared/Split Screen PvP",
    "Cross-Platform Multiplayer", "Cross-Platform Co-op", "LAN PvP", "LAN Co-op",
    "Online PvP", "PvP",
    # 컨트롤
    "Full controller support", "Partial Controller Support",
    # 멀티/소셜
    "Steam Achievements", "Steam Cloud", "Steam Trading Cards", "Steam Workshop",
    "Steam Leaderboards", "Steam Timeline", "SteamVR Collectibles",
    "Family Sharing", "Remote Play Together", "Remote Play on TV",
    "Remote Play on Phone", "Remote Play on Tablet",
    # 모더/툴
    "Includes Source SDK", "Includes level editor", "Mods",
    # 기타
    "Steam Turn Notifications", "Commentary available", "Captions available",
    "Subtitle Options", "MMO", "VR Only", "VR Supported",
})

# categories에 들어오면 안 되는 (Steam TAGS) — free_text로 라우팅하거나 무시
NON_CATEGORY_TAGS: frozenset[str] = frozenset({
    "Story", "Story Rich", "Horror", "Atmospheric", "Survival",
    "Casual", "Competitive", "Difficult", "Hardcore", "Relaxing",
    "Funny", "Cute", "Dark", "Sci-fi", "Fantasy", "Anime",
    "FPS", "Action", "RPG", "Strategy", "Adventure", "Simulation",
    "Puzzle", "Sports", "Racing", "Indie",
})


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_extract_json_object(text: str) -> dict | None:
    """LLM 응답에서 JSON 객체 추출 시도.

    순서:
      1. ```json ... ``` 펜스 내부
      2. 첫 번째 { ... } 블록
      3. 전체 응답 그대로 json.loads
    """
    if not text:
        return None
    text = text.strip()

    fence = _JSON_FENCE_RE.search(text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    obj = _JSON_OBJECT_RE.search(text)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _build_prompt(user_query: str) -> tuple[str, str]:
    """system + user prompt 생성."""
    prompts = load_prompts()
    extractor = prompts.get("extractor", {})
    krw_per_usd = get_settings().currency.krw_per_usd

    system = (
        extractor.get("system", "")
        + "\n"
        + extractor.get("notes", "").format(krw_per_usd=krw_per_usd)
    ).strip()

    examples_filled = extractor.get("examples", "").format(
        krw_usd=round(20000 / krw_per_usd, 2),
        krw_30k=round(30000 / krw_per_usd, 2),
    )
    schema = extractor.get("schema", "")

    user = (
        f"스키마:\n{schema}\n\n"
        f"환율: 1 USD = {krw_per_usd} KRW (20000원 = {round(20000/krw_per_usd, 2)} USD, 30000원 = {round(30000/krw_per_usd, 2)} USD)\n\n"
        f"규칙 + 예시:\n{examples_filled}\n\n"
        f"사용자 요청: {user_query}\n\n"
        f"위 규칙/예시를 따라 **순수 JSON 객체 하나**만 출력하세요."
    )
    return system, user


def _validate_and_normalize(raw: dict, original_query: str = "") -> GameConstraint:
    """JSON schema 검증 + 동의어 정규화 + Pydantic 검증 + 쿼리 기반 보정.

    LLM이 자주 빠뜨리는 필드를 쿼리 텍스트로 보정:
    - "무료"/"공짜"/"free"/"프리" → free_only=true (LLM이 null로 둔 경우)
    - "N인칭"/"N명이서" → min_players
    """
    # 0. 빈 필드 채우기 (LLM이 키 자체를 누락한 경우 대비) — 이건 보정 단계 전에 적용
    for k in ["max_price_usd", "free_only", "genres", "categories",
              "min_players", "platforms", "free_text",
              "exclude_genres", "exclude_categories", "exclude_platforms",
              "exclude_free_only"]:
        raw.setdefault(k, None)

    # 1. JSON schema 검증 — 모든 필드가 명시된 상태에서 검증
    try:
        jsonschema.validate(instance=raw, schema=CONSTRAINT_JSON_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"schema validation failed: {e.message}") from e

    # 모든 필드가 None이면 LLM이 추출 실패한 것 → fallback 유도
    # [2026-07-21] exclude_* 포함 — exclude_만 채워진 경우도 valid
    if all(raw.get(k) is None for k in ["max_price_usd", "free_only", "genres",
                                          "categories", "min_players", "platforms", "free_text",
                                          "exclude_genres", "exclude_categories",
                                          "exclude_platforms", "exclude_free_only"]):
        raise ValueError("all fields null — extraction failed")

    # 2. 동의어 정규화 (한↔영) + 비공식 장르 → free_text 자동 라우팅
    if raw.get("genres"):
        raw["genres"] = translate_constraint_genres(raw["genres"])
        valid_genres, free_genres = partition_genres(raw["genres"])
        raw["genres"] = valid_genres if valid_genres else None
        if free_genres:
            # 비공식 장르(Horror, Open World 등)를 free_text에 병합
            existing_ft = (raw.get("free_text") or "").strip()
            appended = " ".join(free_genres)
            raw["free_text"] = f"{existing_ft} {appended}".strip() if existing_ft else appended
            logger.debug(
                "non_steam_genres_routed_to_free_text",
                genres=free_genres,
                free_text=raw["free_text"],
            )
    if raw.get("categories"):
        raw["categories"] = translate_constraint_categories(raw["categories"])

    # 2-1. 장르명이 categories에 잘못 들어간 경우 구출 (LLM 오분류 보정)
    # 예: categories=["Adventure","Racing"] → genres=["Adventure","Racing"], categories=[]
    # GENRE_SYNONYMS로 번역 시도 → 또는 VALID_STEAM_GENRES에 직접 있는 경우 이동.
    if raw.get("categories"):
        true_cats: list[str] = []
        rescued_genres: list[str] = []
        non_cat_demoted: list[str] = []  # [P0 fix] categories→free_text로 라우팅
        for cat in raw["categories"]:
            # 1순위: genre synonym으로 번역 (예: "FPS" → "Action", "fps" → "Action")
            genre_mapped = translate_genre(cat) or translate_genre(cat.lower())
            if genre_mapped and genre_mapped in VALID_STEAM_GENRES:
                rescued_genres.append(genre_mapped)
            # 2순위: 카테고리값 자체가 Steam 공식 장르명 (예: "Adventure", "Racing")
            elif cat in VALID_STEAM_GENRES:
                rescued_genres.append(cat)
            # [P0 fix] 3순위: Steam TAGS (Story, Horror 등)는 categories가 아님 → free_text로 라우팅
            elif cat in NON_CATEGORY_TAGS:
                non_cat_demoted.append(cat)
            else:
                true_cats.append(cat)
        if rescued_genres:
            existing_genres = list(raw.get("genres") or [])
            for g in rescued_genres:
                if g not in existing_genres:
                    existing_genres.append(g)
            raw["genres"] = existing_genres if existing_genres else None
            logger.debug(
                "genres_rescued_from_categories",
                rescued=rescued_genres,
                remaining_cats=true_cats,
            )
        # [P0 fix] non-category tags (Story, Horror 등) → free_text 병합
        if non_cat_demoted:
            existing_ft = (raw.get("free_text") or "").strip()
            appended = " ".join(non_cat_demoted)
            raw["free_text"] = f"{existing_ft} {appended}".strip() if existing_ft else appended
            logger.debug(
                "non_category_tags_demoted_to_free_text",
                tags=non_cat_demoted,
                free_text=raw["free_text"],
            )
        raw["categories"] = true_cats if true_cats else None

    # [P0 fix] categories가 너무 많으면 over-constrained → 상위 2개만 유지
    # 3개+ AND는 현실적으로 0 결과 가능성 ↑ (DB 매칭도 적음)
    if raw.get("categories") and len(raw["categories"]) > 2:
        logger.debug(
            "categories_truncated_to_top2",
            original=raw["categories"],
            kept=raw["categories"][:2],
        )
        raw["categories"] = raw["categories"][:2]

    # 3. 쿼리 기반 보정
    if original_query:
        q_lower = original_query.lower()
        free_keywords = ["무료", "공짜", "프리", "free to play", "f2p"]
        if raw.get("free_only") is None and any(kw in q_lower for kw in free_keywords):
            raw["free_only"] = True
        # "N인칭"/"N명이서" → min_players
        import re
        m = re.search(r"(\d+)\s*인\s*용|(\d+)\s*명이\s*서", original_query)
        if m and raw.get("min_players") is None:
            n = int(m.group(1) or m.group(2))
            raw["min_players"] = n

    # 4. Pydantic 검증
    try:
        c = GameConstraint(**raw)
    except Exception as e:
        raise ValueError(f"pydantic validation failed: {e}") from e

    return c.normalize()


class ConstraintExtractor:
    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        settings: ExtractorSettings | None = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.settings = settings or get_settings().extractor

    def extract(self, user_query: str) -> GameConstraint:
        """자연어 → Constraint.

        3회 재시도 후 fallback (전체 null).
        """
        if not user_query or not user_query.strip():
            return fallback_constraint()

        system, user = _build_prompt(user_query)

        last_error: str | None = None
        for attempt in range(self.settings.max_retries):
            try:
                raw_text = self.llm.generate(
                    user,
                    system=system,
                    temperature=0.0,
                    max_tokens=512,
                    json_mode=True,
                )
            except LLMError as e:
                logger.warning(
                    "llm_call_failed",
                    attempt=attempt,
                    error=str(e),
                )
                last_error = str(e)
                continue

            parsed = _try_extract_json_object(raw_text)
            if parsed is None:
                logger.warning(
                    "json_parse_failed",
                    attempt=attempt,
                    raw=raw_text[:200],
                )
                last_error = "json parse failed"
                continue

            try:
                return _validate_and_normalize(parsed, original_query=user_query)
            except ValueError as e:
                logger.warning(
                    "validation_failed",
                    attempt=attempt,
                    error=str(e),
                )
                last_error = str(e)
                continue

        logger.error(
            "constraint_extraction_failed_using_fallback",
            attempts=self.settings.max_retries,
            last_error=last_error,
        )
        if not self.settings.fallback_to_null:
            raise RuntimeError(
                f"constraint extraction failed after {self.settings.max_retries} attempts: {last_error}"
            )
        return fallback_constraint()


_default_extractor: ConstraintExtractor | None = None


def get_extractor() -> ConstraintExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = ConstraintExtractor()
    return _default_extractor


def reset_extractor() -> None:
    global _default_extractor
    _default_extractor = None
