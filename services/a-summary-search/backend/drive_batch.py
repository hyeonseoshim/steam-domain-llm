"""RAG Step2 배치 자동 드라이버 — Tier1 쿼터 하에서 직렬 배출.

collect(완료 청크 파싱·append) → submit(쿼터 되는 만큼 다음 청크 제출) → 대기,
를 전 청크 수집 완료까지 반복. 완료 시 종료(1회 알림). resume 안전(state 기반).

usage:
    uv run --with google-genai --with python-dotenv backend/drive_batch.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import build_corpus_summaries as B

SLEEP = 180        # 사이클 간 대기(초)
MAX_CYCLES = 400   # 안전장치(~20시간)


def n_chunks() -> int:
    return len(list(B.REQ_DIR.glob("req_*.jsonl")))


def n_collected() -> int:
    s = json.loads(B.STATE.read_text())
    return sum(1 for j in s["jobs"] if j.get("collected"))


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> None:
    total = n_chunks()
    log(f"드라이버 시작 · 총 {total}청크")
    for cyc in range(1, MAX_CYCLES + 1):
        try:
            B.collect()
        except Exception as e:  # noqa: BLE001
            log(f"collect 예외(무시): {str(e)[:80]}")
        done = n_collected()
        if done >= total:
            log(f"✅ 전 청크 수집 완료 ({done}/{total}) — 종료")
            return
        try:
            B.submit()
        except Exception as e:  # noqa: BLE001
            log(f"submit 예외(무시): {str(e)[:80]}")
        log(f"cycle {cyc}: 수집 {done}/{total} · {SLEEP}s 대기")
        time.sleep(SLEEP)
    log("⚠️ MAX_CYCLES 도달 — 중단(재실행하면 이어서)")


if __name__ == "__main__":
    main()
