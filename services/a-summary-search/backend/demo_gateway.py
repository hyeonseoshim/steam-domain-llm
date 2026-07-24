"""팀 통합(MSA) 게이트웨이 — appid 기준 게임 상세 페더레이션.

BFF(demo_app `/game/{appid}`)가 상세 패널(A 요약 / B 리뷰토픽 / C 동적추천 근거)을
fan-out 하여 하나의 게임 상세로 합친다. D는 질의 종속 검색 전용이다. 파트마다 입출력 계약이 다르지만, 프론트가 균일하게
렌더하도록 **GamePanel** 봉투로 정규화한다. 한 파트가 죽어도 나머지는 그대로 뜬다(degrade).

원격 파트가 구현할 엔드포인트(상세 스펙 = CONTRACT.md, local/ 미추적):
    GET  {base_url}/panel?appid=<int>[&q=<질의>][&uid=<익명ID>]
    200  {"status":"ok","fields":[{"k":"긍정 토픽","v":"gameplay, story"}, ...],"note":""}

status: ok | unavailable(미탑재·타임아웃) | error(서버 처리실패) | mock(연동 전 예시).
새 의존성 없이 stdlib(urllib)만 사용 — 게이트웨이는 얇게 유지.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

# 파트 메타(라벨·파트 태그) — 단일 출처. 새 파트 추가 시 여기만 수정.
PARTS: dict[str, dict[str, str]] = {
    "A": {"label": "요약", "owner": "Part A"},
    "B": {"label": "리뷰 토픽", "owner": "Part B"},
    "C": {"label": "동적 추천", "owner": "Part C"},
    "D": {"label": "조건 근거", "owner": "Part D"},
}


@dataclass
class PanelField:
    """게임 상세 카드의 한 줄(라벨-값). 파트마다 값 종류는 달라도 렌더는 균일."""
    k: str
    v: str


@dataclass
class GamePanel:
    """한 파트가 특정 게임(appid)에 대해 내놓는 정규화 패널."""
    part: str                       # "A".."D"
    label: str                      # 예: "요약"
    owner: str                      # 담당자
    status: str = "unavailable"     # ok | unavailable | error | mock
    fields: list[PanelField] = field(default_factory=list)
    note: str = ""                  # 사유/안내(오류·예시 표기 등)
    latency_ms: int | None = None   # 게이트웨이가 잰 왕복(원격은 게이트웨이↔GPU 홉 포함)
    gpu: dict | None = None         # A(원격 GPU)는 생성 후 VRAM 상태를 실어옴(프론트 계기판용)
    gen_ms: int | None = None       # A(GPU)의 순수 서버 생성시간(홉 제외) — 프론트 '왕복 vs 생성' 분리용

    def to_dict(self) -> dict:
        return asdict(self)


class Provider(Protocol):
    part: str

    def panel(self, appid: int, ctx: dict) -> GamePanel: ...


def _meta(part: str) -> tuple[str, str]:
    m = PARTS.get(part, {})
    return m.get("label", part), m.get("owner", "")


class LocalSummaryProvider:
    """Part A — 인메모리 색인 요약(장르/핵심플레이/특징)에서 패널 생성. HTTP 없음."""

    part = "A"

    def __init__(self, get_summary: Callable[[int], dict]):
        self._get = get_summary  # appid -> {"장르":..,"핵심플레이":..,"특징":..}

    def panel(self, appid: int, ctx: dict) -> GamePanel:
        t = time.perf_counter()
        label, owner = _meta(self.part)
        s = self._get(appid) or {}
        fields = [PanelField(k, str(s[k])) for k in ("장르", "핵심플레이", "특징") if s.get(k)]
        status = "ok" if fields else "unavailable"
        note = "" if fields else "이 게임의 색인 요약이 아직 없습니다."
        return GamePanel(self.part, label, owner, status, fields, note,
                         round((time.perf_counter() - t) * 1000))


class CallablePanelProvider:
    """임의 함수로 패널 생성 — 무겁거나 상태 있는 로컬 파트용(예: A 실시간 요약).

    fn(appid, ctx) -> (status, list[PanelField], note). 예외는 error 패널로 흡수한다.
    latency_ms는 fn 실행 전체(A의 경우 생성 시간)를 담아 '실측'을 그대로 노출.
    """

    def __init__(self, part: str, fn: Callable[[int, dict], tuple]):
        self.part = part
        self._fn = fn

    def panel(self, appid: int, ctx: dict) -> GamePanel:
        t = time.perf_counter()
        label, owner = _meta(self.part)
        try:
            status, fields, note = self._fn(appid, ctx)
        except Exception as e:  # noqa: BLE001
            return GamePanel(self.part, label, owner, "error", [],
                             f"{type(e).__name__}: {e}",
                             round((time.perf_counter() - t) * 1000))
        return GamePanel(self.part, label, owner, status, list(fields), note,
                         round((time.perf_counter() - t) * 1000))


class RemotePanelProvider:
    """Part B/C — 팀원 서버의 GET {base_url}/panel?appid=..&q=.. 호출(계약).

    성공: 계약 JSON을 GamePanel로 매핑. 실패(미탑재/타임아웃/네트워크/형식): unavailable.
    """

    def __init__(self, part: str, base_url: str, timeout_s: float = 25.0):
        self.part = part
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def panel(self, appid: int, ctx: dict) -> GamePanel:
        t = time.perf_counter()
        label, owner = _meta(self.part)
        params = {"appid": appid}
        if ctx.get("q"):
            params["q"] = ctx["q"]
        if ctx.get("uid"):
            params["uid"] = ctx["uid"]   # C 개인화용(기기 UUID) — B/A는 무시
        url = f"{self.base_url}/panel?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as r:
                body = json.loads(r.read().decode("utf-8"))
            fields = [PanelField(str(f.get("k", "")), str(f.get("v", "")))
                      for f in body.get("fields", []) if f.get("k")]
            status = body.get("status", "ok")
            if status == "ok" and not fields:
                status = "unavailable"
            return GamePanel(self.part, label, owner, status, fields,
                             body.get("note", ""),
                             round((time.perf_counter() - t) * 1000),
                             body.get("gpu"),    # GPU 백엔드가 실어온 VRAM 상태 통과(A 계기판용)
                             body.get("gen_ms"))  # GPU 순수 생성시간 통과(왕복과 분리 표시)
        except Exception as e:  # noqa: BLE001 — 게이트웨이는 어떤 실패든 삼켜 degrade
            return GamePanel(self.part, label, owner, "unavailable", [],
                             f"연동 대기(서버 미응답): {type(e).__name__}",
                             round((time.perf_counter() - t) * 1000))


class MockProvider:
    """팀원 서버 연동 전 데모용 예시 패널. 화면에서 status='mock'으로 '예시'임을 명시."""

    def __init__(self, part: str, sample: list[tuple[str, str]]):
        self.part = part
        self._sample = sample

    def panel(self, appid: int, ctx: dict) -> GamePanel:
        label, owner = _meta(self.part)
        return GamePanel(self.part, label, owner, "mock",
                         [PanelField(k, v) for k, v in self._sample],
                         "예시(스텁) — 팀원 서버 연동 전", 0)


def _fmt(v) -> str:
    """리스트→쉼표결합, 정수→천단위, 그 외→문자열. 파일 값의 균일 렌더."""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


class FileTablePanelProvider:
    """appid별 배치 테이블 파일 어댑터(과거/비상 호환용).

    지원 포맷: JSONL(줄마다 dict) 또는 JSON(dict 배열). 각 행은 appid + 필드들.
    spec = [(json_key, 라벨, 접미사)]; None이면 appid 제외 전 필드를 그대로 렌더.
    """

    def __init__(self, part: str, path: str,
                 spec: list[tuple[str, str, str]] | None = None,
                 appid_key: str = "appid"):
        self.part = part
        self.spec = spec
        self.appid_key = appid_key
        self.table: dict[int, dict] = {}
        self._err = ""
        try:
            self.table = self._load(Path(path))
        except Exception as e:  # noqa: BLE001 — 파일 문제도 degrade로 흡수
            self._err = f"{type(e).__name__}: {e}"

    def _load(self, p: Path) -> dict[int, dict]:
        raw = p.read_text(encoding="utf-8").strip()
        rows: list[dict]
        try:                                   # JSON 배열 우선
            data = json.loads(raw)
            rows = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:           # 아니면 JSONL
            rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
        table: dict[int, dict] = {}
        for r in rows:
            if self.appid_key in r:
                table[int(r[self.appid_key])] = r
        return table

    def panel(self, appid: int, ctx: dict) -> GamePanel:
        t = time.perf_counter()
        label, owner = _meta(self.part)
        if self._err:
            return GamePanel(self.part, label, owner, "error", [],
                             f"테이블 로드 실패: {self._err}", 0)
        row = self.table.get(appid)
        if not row:
            return GamePanel(self.part, label, owner, "unavailable", [],
                             "이 게임은 아직 테이블에 없습니다.", 0)
        if self.spec:
            fields = [PanelField(lbl, _fmt(row[k]) + suf)
                      for k, lbl, suf in self.spec if row.get(k) not in (None, "", [])]
        else:
            fields = [PanelField(k, _fmt(v)) for k, v in row.items()
                      if k != self.appid_key and v not in (None, "", [])]
        return GamePanel(self.part, label, owner, "ok", fields, "",
                         round((time.perf_counter() - t) * 1000))


# 파트별 파일 필드 매핑(json_key, 화면 라벨, 접미사). 정의 없으면 전 필드 그대로 렌더.
FILE_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "B": [("positive_topics", "긍정 토픽", ""),
          ("negative_topics", "부정 토픽", ""),
          ("review_count", "표본 리뷰 수", "건")],
}


# 연동 전 데모가 4파트 모두 채워 보이도록 하는 대표 예시(명백히 '예시'로 표기됨).
MOCK_SAMPLES: dict[str, list[tuple[str, str]]] = {
    "B": [("긍정 토픽", "gameplay, story"), ("부정 토픽", "bugs, performance"),
          ("표본 리뷰 수", "1,240건")],
    "C": [("가상 유저 성향", "탐험·서사 선호(최근 경험 가중)"),
          ("추천 근거", "최근 서사형 플레이 이력과 장르 적합도가 높음")],
    "D": [("추출 조건", "가격 무료 · 장르 협동 · 멀티플레이"),
          ("추천 근거", "질의의 하드 제약(멀티/무료)에 부합")],
}


def build_providers(a_provider: Provider, env: dict,
                    mock_when_unset: bool = True) -> list[Provider]:
    """A 프로바이더(요약 — 실시간 생성/색인)는 호출부가 주입. B/C 패널은 우선순위
    PART_{X}_FILE(파일) > PART_{X}_URL(원격) > 예시 스텁.

    팀 통합 계약은 각 파트 서버 URL을 전제로 한다. FILE은 과거/비상 호환용이며 설정 시 우선한다.
    mock_when_unset=False면 미설정 파트는 'unavailable'로 표시(발표에서 예시 숨김).
    """
    # 패널 = 게임 고유/지속 컨텍스트로 산출: B(리뷰), C(uid 성향). D는 질의 종속이라 검색 전용(패널 X).
    providers: list[Provider] = [a_provider]
    for part in ("B", "C"):
        fpath = (env.get(f"PART_{part}_FILE") or "").strip()
        url = (env.get(f"PART_{part}_URL") or "").strip()
        if fpath:
            providers.append(FileTablePanelProvider(part, fpath, FILE_SPECS.get(part)))
        elif url:
            providers.append(RemotePanelProvider(part, url))
        elif mock_when_unset:
            providers.append(MockProvider(part, MOCK_SAMPLES.get(part, [])))
        else:
            providers.append(RemotePanelProvider(part, "http://unset.invalid"))
    return providers


def aggregate(appid: int, providers: list[Provider], ctx: dict | None = None,
              max_workers: int = 4) -> list[dict]:
    """프로바이더들을 동시에 호출 → 파트 순서(A,B,C,D)대로 패널 dict 리스트."""
    ctx = ctx or {}
    order = {p: i for i, p in enumerate(PARTS)}  # A,B,C,D 순
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        panels = list(ex.map(lambda p: p.panel(appid, ctx), providers))
    panels.sort(key=lambda gp: order.get(gp.part, 99))
    return [gp.to_dict() for gp in panels]
