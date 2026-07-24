"""CPU 게이트웨이 — 오케스트레이션 + 팀원 파트(B/C/D). GPU 무관, 콜드스타트 초 단위.

검색·A 요약은 GPU 백엔드(gpu_backend, 별도 Modal 앱)로 원격/프록시한다. B/C/D·페이지·집계는
여기서 처리하므로 **GPU가 자도 팀원 파트는 계속 동작**(가용성 분리) + 페이지 열어둬도 GPU 안 깨움.
  · GET /                       데모 페이지(프론트; 최종은 Vercel 정적)
  · GET /search                 GPU 백엔드로 프록시(검색은 GPU 필요)
  · GET /game/{appid}/parts     파트 목록(즉답)
  · POST /events/{search,click}  C 행동 이벤트 전달
  · GET /game/{appid}/panel/{p} A=GPU 원격 / B=팀원 서버 / C·D=검색 전용
  · GET /stats · /health        게이트웨이 상태(VRAM 없음 — GPU 안 깨움)

env: GPU_BASE_URL(필수, GPU 앱 URL) · PART_{B,C,D}_URL · ALLOW_ORIGINS · INTEGRATION_MOCK
run:  uv run --no-sync uvicorn demo_app:app --app-dir backend --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import shared_counters as sc   # 누적(modal.Dict 공유·영속) — 세션 카운터는 프론트가 별도 집계
from demo_gateway import PARTS, Provider, RemotePanelProvider, aggregate, build_providers

app = FastAPI(title="Steam Part A — CPU gateway")
_origins = os.environ.get("ALLOW_ORIGINS", "*").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)

GPU_URL = os.environ.get("GPU_BASE_URL", "").rstrip("/")   # GPU 백엔드(gpu_backend) 앱 URL
GPU_SUMMARY_URL = os.environ.get("GPU_SUMMARY_URL", GPU_URL).rstrip("/")
C_URL = os.environ.get("PART_C_URL", "").rstrip("/")       # C 행동 수집 + 개인화 추천 서버
FRONTEND_VERSION = os.environ.get("FRONTEND_VERSION", "")
SEARCH_TIMEOUT = 120                                       # A/C/D 검색(컨테이너 콜드스타트 + 계산 여유)
# A 생성: 웜 ~4초 ≪ 25 ≪ vLLM 콜드 ~131초. 짧게 잡아 콜드면 게이트웨이가 붙잡지 않고 빠르게 JSON
# 반환(unavailable) → 프론트가 "예열 중" 재시도. 길게(120초) 잡으면 함수 타임아웃→modal-http 비-JSON.
PANEL_TIMEOUT = 25                                         # A/B/C 패널 요청당 제한; 예열 신호는 프론트가 재시도
HTML = (Path(__file__).parent.parent / "frontend" / "index.html").read_text(encoding="utf-8")
PROVIDERS: list[Provider] = []
COUNTS = {"search": 0, "game": 0}
KST = timezone(timedelta(hours=9))
STARTED_AT = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
_SUMMARY_WAKE_LOCK = threading.Lock()
_summary_wake_thread: threading.Thread | None = None


@app.on_event("startup")
def _startup() -> None:
    global PROVIDERS
    mock = os.environ.get("INTEGRATION_MOCK", "1") != "0"
    # A = GPU 백엔드 원격. B/C/D = 각 팀원 서버 URL(미설정 시 명시적 예시).
    # B 파일 어댑터는 과거/비상 호환용으로만 남아 있고, 팀 통합 계약은 서버 배포를 전제로 한다.
    a = RemotePanelProvider(
        "A", GPU_SUMMARY_URL or "http://gpu.invalid", timeout_s=PANEL_TIMEOUT
    )
    PROVIDERS = build_providers(a, os.environ, mock)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "gpu_url_set": bool(GPU_URL),
            "gpu_summary_url_set": bool(GPU_SUMMARY_URL), "since": STARTED_AT,
            "frontend_version": FRONTEND_VERSION}


@app.get("/stats")
def stats() -> dict:
    # VRAM 없음 — GPU를 깨우지 않기 위해 게이트웨이는 GPU에 묻지 않는다(프론트 VRAM은 검색/A 응답에서).
    # totals = modal.Dict 공유 누적(컨테이너·앱·콜드스타트 무관). GPU가 쓴 gen/rejected도 여기서 읽음(GPU 안 깨움).
    return {"device": "게이트웨이(CPU)", "totals": sc.totals(), "since": STARTED_AT}


MODE_REASON = {"C": "최근 취향과 유사 (예시)", "D": "질의 조건에 부합 (예시)"}


class SearchEvent(BaseModel):
    userId: str
    query: str


class ClickEvent(BaseModel):
    userId: str
    appid: int
    game_name: str | None = None


def _post_c_event(path: str, payload: dict) -> dict:
    """C 이벤트 API에 JSON을 전달한다. 공개 엔드포인트에서는 실패를 그대로 알린다."""
    if not C_URL:
        raise RuntimeError("PART_C_URL 미설정")
    req = urllib.request.Request(
        f"{C_URL}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def _try_log_search(user_id: str, query: str) -> None:
    """검색 결과 자체는 C 이벤트 서버 장애와 무관하게 제공한다."""
    if not query:
        return
    try:
        _post_c_event("/events/search", {"userId": user_id, "query": query})
    except Exception:  # noqa: BLE001 — 행동 로그 장애를 검색 장애로 전파하지 않음
        pass


@app.post("/events/search")
def record_search_event(event: SearchEvent) -> dict:
    try:
        return _post_c_event("/events/search", event.model_dump())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"C 검색 이벤트 전달 실패: {type(e).__name__}") from e


@app.post("/events/click")
def record_click_event(event: ClickEvent) -> dict:
    try:
        return _post_c_event("/events/click", event.model_dump(exclude_none=True))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"C 클릭 이벤트 전달 실패: {type(e).__name__}") from e


def _normalize_partner_results(body: dict, k: int) -> dict:
    """팀원 검색 응답을 프론트 카드 형식으로 얇게 정규화한다.

    배열 순서가 계약상 순위이므로 rank 는 게이트웨이가 부여한다. 필수 조인 키가 없는 행은
    제외하고, 선택 필드는 타입만 화면 친화적으로 맞춘다.
    """
    raw = body.get("results", [])
    if not isinstance(raw, list):
        raise ValueError("results must be a JSON array")
    rows = []
    for item in raw[:max(0, k)]:
        if not isinstance(item, dict) or item.get("appid") is None:
            continue
        try:
            appid = int(item["appid"])
        except (TypeError, ValueError):
            continue
        row = dict(item)
        row.update({"rank": len(rows) + 1, "appid": appid,
                    "name": str(item.get("name") or ""),
                    "reason": str(item.get("reason") or "")})
        rows.append(row)
    body["results"] = rows
    body["note"] = str(body.get("note") or body.get("recent_context") or "")
    return body


def _gpu_search(q: str, k: int) -> dict:
    url = f"{GPU_URL}/search?{urllib.parse.urlencode({'q': q, 'k': k})}"
    with urllib.request.urlopen(url, timeout=SEARCH_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _prewarm_summary() -> None:
    """명시적 검색과 동시에 요약 L4를 깨우되 검색 응답은 기다리지 않는다."""
    global _summary_wake_thread
    if not GPU_SUMMARY_URL:
        return
    with _SUMMARY_WAKE_LOCK:
        if _summary_wake_thread is not None and _summary_wake_thread.is_alive():
            return

        def wake() -> None:
            try:
                with urllib.request.urlopen(f"{GPU_SUMMARY_URL}/health", timeout=180):
                    pass
            except Exception as exc:  # noqa: BLE001
                print(f"[gateway] summary prewarm failed: {type(exc).__name__}")

        _summary_wake_thread = threading.Thread(target=wake, daemon=True)
        _summary_wake_thread.start()


@app.get("/search")
def search(q: str = "", k: int = 30, mode: str = "A", uid: str | None = None) -> dict:
    """검색 모드 라우팅. A=GPU 하이브리드 / C=성향추천 / D=조건추출.
    C·D는 팀원 서버(PART_{X}_URL/search) 있으면 그쪽, 없으면 예시(A 검색 + 모드 근거)."""
    mode = (mode or "A").upper()
    user_id = uid or "guest_user"
    # 어느 탭에서 검색했든 동일 사용자 행동으로 C에 축적한다. C 추천 요청보다 먼저 기록해야
    # 방금 입력한 검색어가 현재 추천 랭킹과 사유에 즉시 반영된다.
    _try_log_search(user_id, q)
    # 프론트가 명시적으로 /search를 호출한 경우에만 A 실시간 요약 GPU도 병렬 예열한다.
    _prewarm_summary()

    if mode in ("C", "D"):
        part_url = (os.environ.get(f"PART_{mode}_URL") or "").strip()
        if part_url:   # 팀원 서버 연동됨 → 계약 /search 호출
            params = ({"userId": user_id, "k": k} if mode == "C"
                      else {"q": q, "k": k, "uid": user_id})
            try:
                with urllib.request.urlopen(
                        f"{part_url}/search?{urllib.parse.urlencode(params)}",
                        timeout=SEARCH_TIMEOUT) as r:
                    body = json.loads(r.read().decode("utf-8"))
                body = _normalize_partner_results(body, k)
                body["mode"] = mode
                COUNTS["search"] += 1
                sc.bump("search")
                return body
            except Exception:  # noqa: BLE001 — 연동 실패 시 예시로 degrade
                pass
        # 예시(mock): A 검색 결과 재사용 + 모드 근거(연동 전 UX 확인용)
        if not GPU_URL:
            return {"query": q, "mode": mode, "results": [], "error": f"PART_{mode}_URL 응답 없음"}
        try:
            body = _gpu_search(q or "인기 게임", k)
        except Exception as e:  # noqa: BLE001
            return {"query": q, "mode": mode, "results": [], "error": f"GPU 응답 없음: {type(e).__name__}"}
        for it in body.get("results", []):
            it["reason"] = MODE_REASON[mode]
        body["mode"] = mode
        body["mock"] = True
        body["note"] = ("예시 — C 서버 연동 전 (계약: GET /search?userId=&k=)" if mode == "C"
                        else "예시 — D 서버 연동 전 (계약: GET /search?q=&uid=)")
        COUNTS["search"] += 1
        sc.bump("search")
        return body

    # mode A: GPU 하이브리드 검색
    if not GPU_URL:
        return {"query": q, "mode": "A", "results": [], "error": "GPU_BASE_URL 미설정"}
    try:
        body = _gpu_search(q, k)
    except Exception as e:  # noqa: BLE001
        return {"query": q, "mode": "A", "results": [], "error": f"GPU 백엔드 응답 없음: {type(e).__name__}"}
    body["mode"] = "A"
    COUNTS["search"] += 1
    sc.bump("search")   # 누적(공유)
    return body


def _provider_for(part: str) -> Provider | None:
    return next((p for p in PROVIDERS if getattr(p, "part", None) == part), None)


@app.get("/game/{appid}/parts")
def game_parts(appid: int, source: str | None = None) -> dict:
    """패널 목록. A·B는 항상. C·D의 근거는 검색 결과의 reason으로 표시한다."""
    COUNTS["game"] += 1
    sc.bump("game")   # 누적(공유) — 통합상세 진입 1회당
    parts = [{"part": p.part,
              "label": PARTS.get(p.part, {}).get("label", p.part),
              "owner": PARTS.get(p.part, {}).get("owner", "")}
             for p in PROVIDERS if p.part != "C"]
    order = {"A": 0, "B": 1}
    parts.sort(key=lambda x: order.get(x["part"], 9))
    return {"appid": appid, "name": "", "parts": parts}   # name은 프론트가 검색결과서 이미 앎


@app.get("/game/{appid}/panel/{part}")
def game_panel(appid: int, part: str, q: str | None = None, uid: str | None = None) -> dict:
    """단일 파트 패널. A=GPU 원격 요약 / B=리뷰토픽. 서로 독립."""
    prov = _provider_for(part)
    if prov is None:
        return {"part": part, "label": part, "owner": "", "status": "unavailable",
                "fields": [], "note": "알 수 없는 파트", "latency_ms": 0}
    ctx: dict = {}
    if q:
        ctx["q"] = q
    if uid:
        ctx["uid"] = uid   # C 개인화(기기 UUID) 통과
    return prov.panel(appid, ctx).to_dict()


@app.get("/game/{appid}")
def game(appid: int, q: str | None = None) -> dict:
    t = time.perf_counter()
    panels = aggregate(appid, PROVIDERS, {"q": q} if q else {})
    COUNTS["game"] += 1
    return {"appid": appid, "name": "", "panels": panels,
            "latency_ms": round((time.perf_counter() - t) * 1000)}
