"""Part A · Step 3 — 평가 지표 (모델 무관, 로컬 CPU 실행 가능).

게임 설명 요약(고정 3필드: 장르/핵심플레이/특징)을 gold 대비 채점한다.
  - 형식 준수율(format compliance): 모델이 3필드 스키마·명사구 장르·불릿없음·한국어를
    지켰는가 (규칙 기반, 표준 라이브러리만).
  - ROUGE-1/2/L: gold 요약과의 n-gram/LCS 겹침. 한국어이므로 형태소 토크나이즈
    (kiwipiepy 있으면 사용, 없으면 정규식 폴백) — 어느 환경에서도 실행되도록.
BERTScore(의미 유사도)는 torch 필요 → bertscore.py 에서 별도(GPU 권장).

ROUGE 는 외부 rouge 패키지의 영어 토크나이저 문제를 피하려 직접 구현.
"""

from __future__ import annotations

import re
from collections import Counter

FIELDS = ("장르", "핵심플레이", "특징")

# --- 토크나이저: 한국어 형태소(가능하면) / 정규식 폴백 -----------------------
try:
    from kiwipiepy import Kiwi  # type: ignore

    _kiwi = Kiwi()

    def tokenize(text: str) -> list[str]:
        return [t.form for t in _kiwi.tokenize(text or "")]

    TOKENIZER = "kiwi"
except Exception:  # noqa: BLE001 — kiwi 미설치 환경 폴백
    _WORD = re.compile(r"[가-힣]+|[a-zA-Z]+|[0-9]+")

    def tokenize(text: str) -> list[str]:
        return _WORD.findall(text or "")

    TOKENIZER = "regex"


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(match: int, pred_total: int, ref_total: int) -> dict[str, float]:
    p = match / pred_total if pred_total else 0.0
    r = match / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"p": p, "r": r, "f": f}


def rouge_n(pred: str, ref: str, n: int) -> dict[str, float]:
    pt, rt = tokenize(pred), tokenize(ref)
    pg, rg = _ngrams(pt, n), _ngrams(rt, n)
    match = sum((pg & rg).values())  # 겹치는 n-gram 수(중복 고려)
    return _prf(match, max(sum(pg.values()), 0), max(sum(rg.values()), 0))


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> dict[str, float]:
    pt, rt = tokenize(pred), tokenize(ref)
    return _prf(_lcs(pt, rt), len(pt), len(rt))


# --- 형식 준수(format compliance) -------------------------------------------
_BULLET = re.compile(r"(^|\n)\s*[-*•·▪◦]\s")
_PAREN = re.compile(r"[()（）]")
_LATIN = re.compile(r"[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*")
_GENRE_LATIN_OK = {
    "rpg", "mmo", "mmorpg", "fps", "tps", "moba", "trpg", "srpg", "arpg",
    "jrpg", "pvp", "pve", "pvpve", "avg", "slg", "rts", "tcg", "vr", "ar",
    "sf", "fmv", "midi", "roguelike", "roguelite", "2d", "3d",
}


def _hangul_ratio(s: str) -> float:
    h = c = l = 0
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF:
            h += 1
        elif 0x4E00 <= o <= 0x9FFF:
            c += 1
        elif ch.isascii() and ch.isalpha():
            l += 1
    d = h + c + l
    return h / d if d else 1.0


def format_checks(summary: dict | None) -> dict[str, bool]:
    """단일 요약(dict)의 형식 항목별 통과 여부. 파싱 실패 시 전부 False."""
    if not isinstance(summary, dict):
        return {"schema": False, "genre_noun": False,
                "no_bullet": False, "korean": False}
    fields_ok = all(isinstance(summary.get(k), str) and summary[k].strip()
                    for k in FIELDS)
    genre = summary.get("장르") or ""
    body = f"{summary.get('핵심플레이', '')}\n{summary.get('특징', '')}"
    latin = [m.group(0) for m in _LATIN.finditer(genre)]
    genre_noun = (not _PAREN.search(genre)
                  and all(w.lower() in _GENRE_LATIN_OK for w in latin))
    no_bullet = not _BULLET.search(f"{genre}\n{body}") and not any(
        ord(ch) >= 0x1F000 for ch in f"{genre}\n{body}")
    korean = _hangul_ratio(f"{genre}\n{body}") >= 0.40
    return {"schema": fields_ok, "genre_noun": genre_noun,
            "no_bullet": no_bullet, "korean": korean}


def is_compliant(summary: dict | None) -> bool:
    return all(format_checks(summary).values())


def concat(summary: dict | None) -> str:
    if not isinstance(summary, dict):
        return ""
    return " ".join(str(summary.get(k, "")) for k in FIELDS).strip()
