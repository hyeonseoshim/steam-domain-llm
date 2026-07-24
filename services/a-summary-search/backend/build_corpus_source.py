"""RAG 파이프라인 Step1 — PG덤프에서 150K 검색 소스 텍스트 추출.

`pg_restore --data-only -t applications` 의 COPY 스트림을 stdin 으로 받아,
type=game 게임의 detailed_description(평범한 text 컬럼)을 뽑아 jsonl 로 낸다.
postgres 서버·docker 불필요(스트림 파싱). detailed 없으면 about_the_game→short 폴백.

usage:
    D=zenodo_dl/steam_dataset_2025_power_users/steam_dataset_20250929.dump
    pg_restore --data-only -t applications -f - "$D" \
        | uv run backend/build_corpus_source.py > backend/corpus_source.jsonl
"""

from __future__ import annotations

import json
import sys

# COPY 컬럼 순서(확인됨): appid=0, name=3, type=4, detailed=12, short=13, about=14
I_APPID, I_NAME, I_TYPE, I_DETAIL, I_SHORT, I_ABOUT = 0, 3, 4, 12, 13, 14
_ESC = {"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}


def unesc(s: str) -> str | None:
    """COPY text 이스케이프 해제. \\N=NULL."""
    if s == r"\N":
        return None
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            out.append(_ESC.get(s[i + 1], s[i + 1])); i += 2
        else:
            out.append(c); i += 1
    return "".join(out)


def main() -> None:
    out = sys.stdout
    in_copy = False
    n_game = n_out = 0
    for line in sys.stdin:
        if not in_copy:
            if line.startswith("COPY public.applications "):
                in_copy = True
            continue
        if line.startswith("\\."):
            break
        cols = line.rstrip("\n").split("\t")
        if len(cols) <= I_ABOUT:
            continue
        if (unesc(cols[I_TYPE]) or "") != "game":
            continue
        n_game += 1
        src = unesc(cols[I_DETAIL]) or unesc(cols[I_ABOUT]) or unesc(cols[I_SHORT]) or ""
        src = src.strip()
        if not src:
            continue
        appid = cols[I_APPID].strip()
        name = (unesc(cols[I_NAME]) or "").strip()
        out.write(json.dumps({"appid": int(appid), "name": name, "source_text": src},
                             ensure_ascii=False) + "\n")
        n_out += 1
        if n_game % 20000 == 0:
            sys.stderr.write(f"  game {n_game} · 소스있음 {n_out}\n"); sys.stderr.flush()
    sys.stderr.write(f"[step1] type=game {n_game} · 소스텍스트 추출 {n_out}\n")


if __name__ == "__main__":
    main()
