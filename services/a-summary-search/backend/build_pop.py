"""RAG 인기 신호 — PG덤프에서 appid→recommendations_total 추출.

검색 순위에 '인기 보정'을 넣기 위한 재료. applications 테이블의 recommendations_total
(스팀 자체 추천 수) 컬럼만 뽑아 {appid: n} JSON 으로 낸다. 값 있는 게임(≈2만)만 저장 —
나머지는 조회 시 0 으로 취급되어 부스트를 안 받는다(무명 클론은 제자리, 유명게임만 상승).

build_corpus_source.py 와 동일하게 COPY 스트림을 파싱(postgres 서버 불필요).
COPY 컬럼: appid=0, type=4, recommendations_total=9 (헤더로 확인됨).

usage:
    D=zenodo_dl/steam_dataset_2025_power_users/steam_dataset_20250929.dump
    pg_restore --data-only -t applications -f - "$D" \
        | uv run backend/build_pop.py > backend/corpus_pop.json
"""

from __future__ import annotations

import json
import sys

I_APPID, I_TYPE, I_REC = 0, 4, 9


def main() -> None:
    in_copy = False
    pop: dict[int, int] = {}
    n_game = 0
    for line in sys.stdin:
        if not in_copy:
            if line.startswith("COPY public.applications "):
                in_copy = True
            continue
        if line.startswith("\\."):
            break
        cols = line.rstrip("\n").split("\t")
        if len(cols) <= I_REC or cols[I_TYPE] != "game":
            continue
        n_game += 1
        rec = cols[I_REC]
        if rec and rec != r"\N":
            try:
                v = int(rec)
            except ValueError:
                continue
            if v > 0:
                pop[int(cols[I_APPID])] = v
    json.dump(pop, sys.stdout)
    sys.stderr.write(f"[build_pop] type=game {n_game:,} · recommendations_total>0 {len(pop):,}\n")


if __name__ == "__main__":
    main()
