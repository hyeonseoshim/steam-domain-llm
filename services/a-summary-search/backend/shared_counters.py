"""분산·영속 카운터 — modal.Dict 공유(컨테이너·앱·콜드스타트 무관 누적 총량).

인메모리 COUNTS 는 컨테이너마다 따로 + scale-to-zero 로 리셋 → '누적 처리량' 표현 불가.
modal.Dict 는 컨테이너 밖 공유 상태라 게이트웨이·GPU 두 앱이 같은 카운터를 증가시키고,
게이트웨이 /stats 가 이를 읽으면 GPU 를 깨우지 않고도 합산을 준다(Dict 는 GPU 컨테이너 밖).

증가는 get→put 이라 완전 원자적이진 않음(간헐적 누락 가능) — 데모 카운터엔 무해.
Modal 밖(로컬 dev)이거나 Dict 접근 실패 시 프로세스 인메모리로 폴백(개발 무해).
"""

from __future__ import annotations

KEYS = ("search", "game", "gen", "rejected")
_local: dict[str, int] = {k: 0 for k in KEYS}
_shared = "uninit"   # lazy: 첫 사용 때 modal.Dict 조회(import-time 네트워크 회피)


def _dict():
    global _shared
    if _shared == "uninit":
        try:
            import modal
            _shared = modal.Dict.from_name("steam-part-a-counters", create_if_missing=True)
        except Exception:  # noqa: BLE001 — 로컬/미인증/버전차 → 인메모리 폴백
            _shared = None
    return _shared


def bump(key: str, n: int = 1) -> None:
    _local[key] = _local.get(key, 0) + n
    d = _dict()
    if d is not None:
        try:
            d.put(key, (d.get(key, 0) or 0) + n)
        except Exception:  # noqa: BLE001
            pass


def totals() -> dict[str, int]:
    d = _dict()
    if d is not None:
        try:
            return {k: int(d.get(k, 0) or 0) for k in KEYS}
        except Exception:  # noqa: BLE001
            pass
    return dict(_local)
