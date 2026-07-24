"""GPU 백엔드 — 검색(bge-m3+리랭커) + A 실시간 요약만. compute 전담.

CPU 게이트웨이(demo_app)가 이걸 원격 호출한다:
  · GET /search?q=&k=          하이브리드+리랭커 검색(Retriever)
  · GET /panel?appid=&q=       A 3필드 실시간 요약 → {status, fields:[{k,v}], note}
  · GET /health · /stats       준비상태·VRAM
검색·A만 GPU를 깨우고, 팀원 파트(B/C/D)·페이지는 CPU 게이트웨이가 처리하므로
이 앱이 scale-to-zero로 자도 데모의 나머지는 계속 동작한다.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import shared_counters as sc   # 누적(modal.Dict 공유·영속) — GPU가 쓴 gen/rejected를 게이트웨이도 읽음
from demo_gateway import PanelField
from demo_retriever import Retriever
from vllm_summarizer import VllmSummarizer as Summarizer   # vLLM graph 진단(enforce_eager=False)

app = FastAPI(title="Steam Part A — GPU backend")
_origins = os.environ.get("ALLOW_ORIGINS", "*").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)

R: Retriever | None = None
S = Summarizer()  # lazy — 첫 /panel 때 모델 로드
SUMMARY_ENABLED = os.environ.get("ENABLE_SUMMARY", "1") != "0"
KST = timezone(timedelta(hours=9))
STARTED_AT = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
COUNTS = {"search": 0, "gen": 0, "rejected": 0}

_GEN_LOCK = threading.Lock()
GEN_QUEUE_WAIT_S = 25
MIN_FREE_GEN_MB = 1200
MIN_FREE_LOAD_MB = 6500

# vLLM 엔진은 백그라운드로 예열 — 로딩 중엔 /panel 이 락 안 잡고 즉시 '예열 중' 반환(재시도 유도),
# 로딩 완료 후에만 실제 생성. 로딩 중 락 점유 시 재시도가 '혼잡'으로 오집계되는 것 방지. 멱등(중복 스레드 X).
_WARM_LOCK = threading.Lock()
_warm_thread = None


def _ensure_warming() -> None:
    global _warm_thread
    if not SUMMARY_ENABLED:
        return
    if S.loaded:
        return
    with _WARM_LOCK:
        if S.loaded or (_warm_thread is not None and _warm_thread.is_alive()):
            return

        def _w() -> None:
            try:
                S._ensure()
                print("[gpu] vLLM 예열 완료 ✓")
            except Exception as e:  # noqa: BLE001
                print(f"[gpu] vLLM 예열 실패(재시도 가능): {type(e).__name__}: {e}")
        _warm_thread = threading.Thread(target=_w, daemon=True)
        _warm_thread.start()
# vLLM 은 gpu_memory_utilization 로 VRAM 을 선점 예약(PagedAttention KV) → free-VRAM 이 항상 낮게
# 보여 free 게이트가 오판(라이브 생성 거부→폴백). vLLM 은 자체 페이징+락으로 백프레셔 → 게이트 끔.
VLLM_MODE = os.environ.get("SUMMARIZER", "hf").lower() == "vllm"


def _vram_free_mb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return free / 2**20
    except Exception:  # noqa: BLE001
        return None


def gpu_stats() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"device": "cpu", "vram_total_mb": None}
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return {"device": torch.cuda.get_device_name(0),
                "vram_used_mb": round(used / 2**20), "vram_total_mb": round(total / 2**20),
                "vram_pct": round(used / total * 100, 1),
                "vram_alloc_mb": round(torch.cuda.memory_allocated() / 2**20)}
    except Exception as e:  # noqa: BLE001
        return {"device": "?", "error": str(e)}


@app.on_event("startup")
def _startup() -> None:
    global R
    started = time.perf_counter()
    R = Retriever()
    print(f"[gpu] Retriever 준비 완료 ({time.perf_counter() - started:.1f}s)")
    if SUMMARY_ENABLED:
        _ensure_warming()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retriever_ready": R is not None,
            "summary_enabled": SUMMARY_ENABLED,
            "model_loaded": S.loaded if SUMMARY_ENABLED else None, "since": STARTED_AT}


@app.get("/stats")
def stats() -> dict:
    return {**gpu_stats(), "counts": COUNTS, "since": STARTED_AT,
            "retriever_ready": R is not None, "model_loaded": S.loaded}


@app.get("/diag")
def diag() -> dict:
    """지연 규명 — GPU 클럭/전력(스로틀?) + 모델 attn구현/device/dtype(오프로드?)."""
    import subprocess
    d: dict = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,power.draw,"
             "power.limit,temperature.gpu,utilization.gpu,pstate",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        d["nvidia_smi"] = r.stdout.strip() or r.stderr.strip()
    except Exception as e:  # noqa: BLE001
        d["nvidia_smi"] = f"err: {type(e).__name__}: {e}"
    m = getattr(S, "_model", None)
    if m is not None:
        try:
            p = next(m.parameters())
            d["attn_impl"] = getattr(getattr(m, "config", None), "_attn_implementation", "?")
            d["param_device"] = str(p.device)
            d["param_dtype"] = str(p.dtype)
        except Exception as e:  # noqa: BLE001
            d["model"] = f"err: {e}"
    else:
        d["model_loaded"] = False
    return d


@app.get("/search")
def search(q: str, k: int = 30) -> dict:
    t = time.perf_counter()
    results = R.search(q, topk=k)
    COUNTS["search"] += 1
    return {"query": q, "results": results,
            "latency_ms": round((time.perf_counter() - t) * 1000), "gpu": gpu_stats()}


@app.get("/panel")
def panel(appid: int, q: str | None = None) -> dict:
    """A 3필드 실시간 요약(색인 lookup 아님). 백프레셔(락+VRAM)를 적용한다."""
    def fields_of(summary: dict) -> list[PanelField]:
        return [PanelField(k, str(summary[k]))
                for k in ("장르", "핵심플레이", "특징") if summary.get(k)]

    def out(status, fields, note, gen_ms=None, out_chars=None, in_chars=None):
        return {"status": status, "fields": [{"k": f.k, "v": f.v} for f in fields],
                "note": note, "gpu": gpu_stats(),
                "gen_ms": gen_ms, "out_chars": out_chars,   # 서버측 생성시간·출력길이
                "in_chars": in_chars}                       # 입력 원문 길이(프리필 상관 검증)

    if not SUMMARY_ENABLED:
        return out("unavailable", [], "실시간 요약 전용 서버로 분리됨")

    # vLLM 아직 예열 전 → 락 안 잡고 즉시 '예열 중' 반환(재시도 유도). 로딩 중 락 점유로 재시도가
    # '혼잡'으로 오집계되는 것 방지. 예열 안 돌고 있으면 백그라운드로 시작(멱등).
    if not S.loaded:
        _ensure_warming()
        return out("unavailable", [],
                   "모델 예열 중(콜드스타트) — 실시간 생성 준비되면 자동 갱신")

    reason = ""
    if _GEN_LOCK.acquire(timeout=GEN_QUEUE_WAIT_S):
        try:
            free = _vram_free_mb()
            need = MIN_FREE_GEN_MB if S.loaded else MIN_FREE_LOAD_MB
            if not VLLM_MODE and free is not None and free < need:
                COUNTS["rejected"] += 1; sc.bump("rejected")
                reason = f"GPU 여유 부족({free:.0f}/{need}MB)"
            else:
                raw = R.raw_source(appid)
                if not raw:
                    reason = "원문 없음"
                else:
                    t0 = time.perf_counter()
                    summary, gentext = S.summarize(R.names.get(appid, ""), [], raw)
                    gen_ms = round((time.perf_counter() - t0) * 1000)   # 네트워크 제외 순수 생성
                    if summary:
                        COUNTS["gen"] += 1; sc.bump("gen")
                        return out("ok", fields_of(summary), "파인튜닝 모델 실시간 생성 ⚡",
                                   gen_ms, len(gentext), len(raw))
                    reason = "형식 파싱 실패"
        finally:
            _GEN_LOCK.release()
    else:
        COUNTS["rejected"] += 1; sc.bump("rejected")
        reason = "생성 혼잡(백프레셔)"
    return out("unavailable", [], reason or "요약 없음")
