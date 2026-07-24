"""Grounded context builder — explainer가 사용할 후보 게임 컨텍스트 문자열.

app_id를 포함해서 LLM이 hallucination 못 하도록 함.
pipeline.py와 explainer.py 모두에서 import하지만 서로를 import하지 않도록
이 모듈은 순수 함수만 가짐.
"""
from __future__ import annotations

from steam_part_d.models.game import Game


def build_grounded_context(
    games: list[Game], max_desc_chars: int = 200
) -> str:
    """LLM에 전달할 grounding context. app_id 포함으로 hallucination 방지."""
    lines: list[str] = []
    for i, g in enumerate(games, 1):
        price = (
            "Free"
            if g.is_free
            else (f"${g.price_usd:.2f}" if g.price_usd is not None else "N/A")
        )
        genres = ", ".join(g.genres[:3]) or "N/A"
        cats = ", ".join(g.categories[:5]) or "N/A"
        desc = (g.summary_ko or g.short_description or "")[:max_desc_chars]
        lines.append(
            f"[{i}] app_id={g.appid} | {g.name} | {price} | "
            f"장르: {genres} | 카테고리: {cats}\n    {desc}"
        )
    return "\n".join(lines)
