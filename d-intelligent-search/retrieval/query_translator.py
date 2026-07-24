"""Korean → English query expansion for better vector search matching.

Steam game descriptions are in English. When users query in Korean, the semantic
gap between Korean queries and English descriptions reduces cosine similarity.

This module expands Korean gaming terms with their English equivalents BEFORE
encoding, so the embedding captures both the Korean and English semantics.

Strategy: for each known Korean term found in the query, append English synonyms
(space-separated). The original Korean text is kept so the embedding is bilingual.

Example:
    "힐링 게임" → "힐링 게임 relaxing cozy wholesome peaceful"
    "소울라이크 어려운 게임" → "소울라이크 어려운 게임 souls-like challenging difficult punishing"
"""
from __future__ import annotations

# Korean gaming term → English translation/expansion for better embedding match.
# Keys: Korean/abbreviated terms users commonly type.
# Values: English words that appear in Steam game descriptions.
KO_TO_EN_GAMING: dict[str, str] = {
    # Horror / atmosphere
    "공포": "horror fear scary frightening",
    "무서운": "horror scary frightening terror",
    "호러": "horror scary terror",
    "으스스": "horror spooky creepy atmosphere",
    # Zombies
    "좀비": "zombie undead apocalypse survival",
    # Healing / relaxing
    "힐링": "relaxing cozy casual peaceful wholesome",
    "힐링게임": "relaxing cozy peaceful wholesome casual",
    "힐링겜": "relaxing cozy peaceful wholesome casual",
    "편안한": "relaxing calm peaceful cozy",
    "스트레스": "relaxing stress relief casual",
    # Souls-like
    "소울라이크": "souls-like challenging difficult punishing dark",
    "소울류": "souls-like challenging difficult punishing",
    "어두운": "dark grim atmosphere Gothic",
    # Classic / retro
    "고전": "classic retro vintage old-school",
    "레트로": "retro classic pixel old-school",
    # Open world
    "오픈월드": "open world sandbox exploration vast",
    "오픈 월드": "open world sandbox exploration",
    # Sandbox
    "샌드박스": "sandbox open world building creativity",
    # Metroidvania
    "메트로배니아": "metroidvania side-scrolling exploration ability",
    "메트로이드바니아": "metroidvania side-scrolling exploration",
    # Roguelike
    "로그라이크": "roguelike roguelite permadeath random run",
    "로그라이트": "roguelite roguelike run-based random",
    "로그라식": "roguelike roguelite",
    # Hack and slash
    "핵앤슬래시": "hack and slash action combat loot",
    "핵슬": "hack and slash action combat",
    # Casual
    "캐주얼": "casual relaxing easy fun",
    # Cute / colorful
    "귀여운": "cute adorable colorful charming",
    "귀엽": "cute adorable charming",
    "아기자기": "cute adorable colorful whimsical",
    # Stealth
    "스텔스": "stealth infiltration sneaking espionage",
    # Survival
    "생존": "survival crafting base building resources",
    "서바이벌": "survival crafting wilderness resources",
    # Space / sci-fi
    "우주": "space sci-fi galaxy exploration stars",
    "스페이스": "space sci-fi exploration",
    "SF": "science fiction sci-fi futuristic",
    "sf": "science fiction sci-fi futuristic",
    # Fantasy
    "판타지": "fantasy magic medieval swords dragons",
    "마법": "magic wizard spells fantasy",
    "드래곤": "dragon fantasy creature mythical",
    # Medieval
    "중세": "medieval knights castle swords armor",
    # Pirates
    "해적": "pirate ship ocean sailing treasure",
    # Ninja
    "닌자": "ninja stealth action martial arts",
    # War / military
    "전쟁": "war military combat strategy battle",
    "전투": "combat battle fighting action",
    # Fighting
    "격투": "fighting combat martial arts",
    "대전": "fighting versus combat",
    # Racing
    "레이싱": "racing cars speed driving motorsport",
    "경주": "racing speed competition track",
    # Puzzle
    "퍼즐": "puzzle brain teaser logic problem",
    "두뇌": "puzzle brain teaser logic",
    # Emotional / story
    "감동": "emotional touching story narrative heartwarming",
    "스토리": "story narrative plot rich",
    "스토리풍": "story-rich narrative adventure",
    # Addictive
    "중독": "addictive engaging compelling satisfying",
    # Co-op
    "협동": "co-op cooperative multiplayer friends team",
    # Battle royale
    "배틀로얄": "battle royale survival last man standing",
    # FPS / shooters
    "FPS": "first person shooter fps shooting",
    "fps": "first person shooter fps shooting",
    "TPS": "third person shooter action",
    "tps": "third person shooter action",
    # MMO / RPG
    "MMORPG": "mmorpg massively multiplayer online rpg",
    "mmorpg": "mmorpg massively multiplayer online rpg",
    "MMO": "massively multiplayer online",
    "mmo": "massively multiplayer online",
    # Farming / life sim
    "농사": "farming simulation harvest crops life",
    "농장": "farming simulation harvest crops",
    "가드닝": "farming gardening simulation nature",
    # City building
    "건설": "city building construction management",
    "도시": "city building management urban",
    # Management
    "경영": "management simulation strategy resource",
    # Sports
    "야구": "baseball sports",
    "축구": "soccer football sports",
    "농구": "basketball sports",
    "골프": "golf sports",
    "테니스": "tennis sports",
    # Psychological
    "심리": "psychological thriller mystery",
    "심리스릴러": "psychological thriller horror mystery",
    # Family / kids
    "어린이": "family friendly children kids",
    "가족": "family friendly cooperative",
    # Deck building
    "덱빌딩": "deckbuilding card game roguelike strategy",
    "카드게임": "card game deckbuilding",
    # Turn-based
    "턴제": "turn-based strategy tactical",
    # Tower defense
    "타워디펜스": "tower defense strategy",
    # Platformer
    "플랫포머": "platformer side-scrolling action jump",
    "플랫폼게임": "platformer side-scrolling action",
    # Adventure
    "어드벤처": "adventure exploration story quest",
    "모험": "adventure exploration quest journey",
    # Indie
    "인디": "indie independent game",
    "인디게임": "indie independent game",
    # Hidden gems
    "숨겨진": "hidden gem underrated indie",
    "보석": "hidden gem underrated",
    # Masterpiece
    "명작": "masterpiece acclaimed classic must-play",
    # New releases
    "새로운": "new recent release modern",
    "신작": "new release recent modern",
    # Visual novel
    "비주얼노벨": "visual novel story romance",
    "비주얼 노벨": "visual novel story romance",
    # Action RPG
    "액션RPG": "action rpg real-time combat",
    "액션 RPG": "action rpg real-time combat",
    # Dark fantasy
    "다크판타지": "dark fantasy gothic grim",
    "다크 판타지": "dark fantasy gothic grim",
    # Anime style
    "애니": "anime Japanese art style",
    "애니메이션": "anime animation Japanese",
    # Pixel art
    "픽셀": "pixel art retro 2D",
    "도트": "pixel art retro 2D",
    # 3D
    "3D": "3D three-dimensional first-person third-person",
    # Multiplayer
    "멀티": "multiplayer online cooperative",
    "멀티플레이": "multiplayer online cooperative",
    # Single player
    "싱글": "single-player solo",
    "혼자": "single-player solo",
    # VR
    "VR": "virtual reality immersive",
    "가상현실": "virtual reality immersive",
    # Horror survival
    "공포 생존": "horror survival scary crafting",
    # Post-apocalyptic
    "포스트아포칼립스": "post-apocalyptic survival wasteland",
    # Cyberpunk
    "사이버펑크": "cyberpunk sci-fi futuristic neon",
    # Steampunk
    "스팀펑크": "steampunk Victorian industrial",
    # Dungeon
    "던전": "dungeon crawler hack and slash",
    # Western
    "서부": "western cowboy frontier",
    # Crime
    "범죄": "crime thriller detective mystery",
    "탐정": "detective mystery crime investigation",
    # Music / rhythm
    "음악": "music rhythm game",
    "리듬": "rhythm music game",
    # Peaceful / zen
    "힐링 게임": "relaxing cozy peaceful wholesome casual",
    "잔잔한": "peaceful calm relaxing cozy",
    "힐링적인": "relaxing cozy wholesome peaceful",
}


def translate_query_for_embedding(query: str) -> str:
    """Expand Korean gaming terms with English equivalents for better vector search.

    The strategy is to append English expansions without removing the Korean text.
    This makes the resulting embedding bilingual, bridging the semantic gap between
    Korean queries and English Steam game descriptions.

    Args:
        query: User's original query (may be Korean, English, or mixed).

    Returns:
        Expanded query string with English synonyms appended for known Korean terms.
        Falls back to original query if no Korean gaming terms are found.

    Examples:
        >>> translate_query_for_embedding("힐링 게임")
        "힐링 게임 relaxing cozy peaceful wholesome casual"
        >>> translate_query_for_embedding("소울라이크 어려운 게임")
        "소울라이크 어려운 게임 souls-like challenging difficult punishing dark"
    """
    if not query or not query.strip():
        return query

    added_words: set[str] = set()
    expansion_parts: list[str] = []

    # Sort by length descending so longer phrases are matched first
    # (e.g., "힐링 게임" before "힐링")
    for ko, en in sorted(KO_TO_EN_GAMING.items(), key=lambda x: len(x[0]), reverse=True):
        if ko in query:
            # Deduplicate at the individual English word level
            new_words = [w for w in en.split() if w.lower() not in added_words]
            if new_words:
                expansion_parts.append(" ".join(new_words))
                added_words.update(w.lower() for w in new_words)

    if not expansion_parts:
        return query

    expanded = query + " " + " ".join(expansion_parts)
    return expanded.strip()
