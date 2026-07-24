"""A 실시간 요약 전용 GPU 백엔드 — 검색 스택과 분리된 scale-to-zero 서비스."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import shared_counters as sc
from summary_store import SummaryStore
from vllm_summarizer import VllmSummarizer

app = FastAPI(title="Steam Part A — realtime summary GPU")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KST = timezone(timedelta(hours=9))
STARTED_AT = datetime.now(KST).isoformat(timespec="seconds")
STORE: SummaryStore | None = None
S = VllmSummarizer()
_WARM_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()
_warm_thread: threading.Thread | None = None


def gpu_stats() -> dict:
    """현재 요약 L4의 전역 VRAM 사용량(vLLM 엔진 자식 프로세스 포함)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"device": "cpu", "vram_total_mb": None}
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return {
            "device": torch.cuda.get_device_name(0),
            "vram_used_mb": round(used / 2**20),
            "vram_total_mb": round(total / 2**20),
            "vram_pct": round(used / total * 100, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {"device": "?", "vram_total_mb": None, "error": str(exc)}


def _ensure_warming() -> None:
    global _warm_thread
    if S.loaded:
        return
    with _WARM_LOCK:
        if S.loaded or (_warm_thread is not None and _warm_thread.is_alive()):
            return

        def load() -> None:
            try:
                S._ensure()
                print("[summary] vLLM 준비 완료")
            except Exception as exc:  # noqa: BLE001
                print(f"[summary] vLLM 준비 실패: {type(exc).__name__}: {exc}")

        _warm_thread = threading.Thread(target=load, daemon=True)
        _warm_thread.start()


@app.on_event("startup")
def startup() -> None:
    global STORE
    started = time.perf_counter()
    # 데이터 인덱싱은 CPU·I/O 위주라 vLLM GPU 로드와 동시에 시작한다.
    _ensure_warming()
    STORE = SummaryStore()
    print(f"[summary] HTTP 준비 {time.perf_counter() - started:.1f}s")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "store_ready": STORE is not None,
        "model_loaded": S.loaded,
        "since": STARTED_AT,
    }


@app.get("/stats")
def stats() -> dict:
    return {**gpu_stats(), "store_ready": STORE is not None,
            "model_loaded": S.loaded, "since": STARTED_AT}


@app.get("/panel")
def panel(appid: int, q: str | None = None) -> dict:
    del q

    def response(status: str, fields: list[dict], note: str, gen_ms: int | None = None) -> dict:
        # 모델 로딩 중에는 CUDA 상태 조회를 추가로 만들지 않는다. 로드 완료 응답부터 계측한다.
        gpu = gpu_stats() if S.loaded else None
        return {"status": status, "fields": fields, "note": note,
                "gen_ms": gen_ms, "gpu": gpu}

    if STORE is None:
        return response("unavailable", [], "원문 인덱스 준비 중")
    if not S.loaded:
        _ensure_warming()
        return response("unavailable", [], "모델 예열 중(콜드스타트)")
    if not _GEN_LOCK.acquire(timeout=25):
        sc.bump("rejected")
        return response("unavailable", [], "생성 혼잡(백프레셔)")
    try:
        raw = STORE.raw_source(appid)
        if not raw:
            return response("unavailable", [], "원문 없음")
        started = time.perf_counter()
        summary, _ = S.summarize(STORE.names.get(appid, ""), [], raw)
        gen_ms = round((time.perf_counter() - started) * 1000)
        if not summary:
            return response("unavailable", [], "형식 파싱 실패", gen_ms)
        sc.bump("gen")
        fields = [
            {"k": key, "v": str(summary[key])}
            for key in ("장르", "핵심플레이", "특징")
            if summary.get(key)
        ]
        return response("ok", fields, "파인튜닝 모델 실시간 생성 ⚡", gen_ms)
    finally:
        _GEN_LOCK.release()
