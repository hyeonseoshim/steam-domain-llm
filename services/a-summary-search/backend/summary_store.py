"""실시간 요약 전용 최소 데이터 스토어.

검색 Retriever의 bge-m3·리랭커·BM25를 올리지 않고, 생성에 필요한 게임명과
appid별 원문 위치만 준비한다. 검색/요약 GPU 분리 시 요약 컨테이너의 콜드스타트를
LLM 로딩 시간에 가깝게 유지하기 위한 모듈이다.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from demo_retriever import clean_html

RP = Path(__file__).resolve().parent
NAMES = RP / "corpus_names.jsonl"
SRC = RP / "corpus_source.jsonl"
_APPID = re.compile(rb'"appid":\s*(\d+)')


class SummaryStore:
    def __init__(self) -> None:
        started = time.perf_counter()
        self.names: dict[int, str] = {}
        for line in NAMES.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                self.names[int(row["appid"])] = str(row.get("name") or "")

        self._src_off: dict[int, int] = {}
        if SRC.exists():
            with SRC.open("rb") as source:
                offset = source.tell()
                line = source.readline()
                while line:
                    match = _APPID.search(line)
                    if match:
                        self._src_off[int(match.group(1))] = offset
                    offset = source.tell()
                    line = source.readline()
        print(
            f"[summary-store] 이름 {len(self.names):,}건 · 원문 인덱스 "
            f"{len(self._src_off):,}건 준비 {time.perf_counter() - started:.1f}s"
        )

    def raw_source(self, appid: int) -> str:
        offset = self._src_off.get(appid)
        if offset is None:
            return ""
        with SRC.open("rb") as source:
            source.seek(offset)
            line = source.readline()
        return clean_html(json.loads(line).get("source_text", ""))
