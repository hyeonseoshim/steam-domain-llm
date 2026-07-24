"""
Part A — Steam 게임 설명 요약: 1단계 데이터 전처리.

입력:  games_5k.json.gz  (GitHub 5K raw 샘플, 중첩 Steam-API JSON)
출력:  data/processed/
         clean.jsonl        정제된 전체 usable 레코드
         train/val/test.jsonl  분할(80/10/10, seed 고정)
         stats.json         정제 전후 통계 (발표/문서용)

의존성: 표준 라이브러리만 사용 (HTML 정제는 html.parser).
        → 어느 환경에서도 pip 없이 재현 가능.

파이프라인:
  load → filter(success & type=game) → HTML 정제 → 중복 제거
       → usable 필터(길이) → 정제 전후 통계 → train/val/test 분할
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import statistics
from html.parser import HTMLParser
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
SEED = 42
SPLIT = (0.8, 0.1, 0.1)  # train / val / test
MIN_CLEAN_CHARS = 200    # 요약할 내용이 있으려면 최소 이 길이 이상
SPLIT_NAMES = ("train", "val", "test")

# HTML 정제 시 내용을 통째로 버릴 태그 (이미지/영상/스크립트 등)
DROP_TAGS = {"script", "style", "noscript", "img", "video", "source",
             "iframe", "svg", "canvas"}
# 줄바꿈으로 치환할 블록 레벨 태그
BLOCK_TAGS = {"p", "br", "div", "ul", "ol", "h1", "h2", "h3", "h4", "h5",
              "h6", "tr", "table", "section", "article", "header", "footer"}


# ---------------------------------------------------------------------------
# HTML → 순수 텍스트
# ---------------------------------------------------------------------------
class _TextExtractor(HTMLParser):
    """Steam detailed_description의 HTML을 순수 텍스트로 변환.

    - img/video/script 등은 내용까지 제거
    - 블록 태그는 줄바꿈으로, <li>는 '- ' 불릿으로
    - convert_charrefs=True 로 &amp; 등 엔티티 자동 디코딩
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0  # DROP_TAGS 내부 깊이

    def handle_starttag(self, tag, attrs):
        if tag in DROP_TAGS:
            self._skip_depth += 1
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        # <br/>, <img .../> 같은 self-closing
        if tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in DROP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


_WS_RUN = re.compile(r"[ \t\f\v]+")
_NL_RUN = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")  # 연속 빈 줄 → 하나


def clean_html(raw: str) -> str:
    """HTML 문자열 → 정규화된 순수 텍스트."""
    if not raw:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
        text = parser.get_text()
    except Exception:
        # 깨진 HTML 방어: 태그만 거칠게 제거
        text = re.sub(r"<[^>]+>", " ", raw)
    text = _WS_RUN.sub(" ", text)
    text = _NL_RUN.sub("\n\n", text)
    # 각 줄 앞뒤 공백 제거
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


# ---------------------------------------------------------------------------
# 필드 추출
# ---------------------------------------------------------------------------
def extract_names(items) -> list[str]:
    """genres/categories 의 [{id, description}, ...] → [description, ...]."""
    if not isinstance(items, list):
        return []
    return [it.get("description", "") for it in items if isinstance(it, dict)]


def build_record(game: dict) -> dict | None:
    """게임 1건 → 정제 레코드. 필터 탈락 시 None."""
    ad = game.get("app_details") or {}
    if not ad.get("success"):
        return None
    data = ad.get("data") or {}
    if data.get("type") != "game":
        return None

    raw_desc = data.get("detailed_description") or ""
    clean_desc = clean_html(raw_desc)

    return {
        "appid": game.get("appid"),
        "name": data.get("name", ""),
        "is_free": data.get("is_free", False),
        "detailed_description_raw": raw_desc,
        "detailed_description_clean": clean_desc,
        "short_description": (data.get("short_description") or "").strip(),
        "genres": extract_names(data.get("genres")),
        "categories": extract_names(data.get("categories")),
        "release_date": (data.get("release_date") or {}).get("date", ""),
        # 정제 전후 길이 (통계·필터용)
        "len_raw": len(raw_desc),
        "len_clean": len(clean_desc),
    }


# ---------------------------------------------------------------------------
# 통계 헬퍼
# ---------------------------------------------------------------------------
def _pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def length_stats(vals: list[int]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "min": s[0],
        "median": int(statistics.median(s)),
        "mean": round(statistics.mean(s), 1),
        "p90": _pct(s, 0.90),
        "p99": _pct(s, 0.99),
        "max": s[-1],
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Part A 데이터 전처리")
    ap.add_argument("--input", default="games_5k.json.gz")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--min-chars", type=int, default=MIN_CLEAN_CHARS)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]  # 프로젝트 루트
    in_path = root / args.input
    out_dir = root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {in_path}")
    with gzip.open(in_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    games = payload.get("games", [])
    print(f"[load] 전체 레코드: {len(games)}")

    # --- 필터 + 정제 ---
    n_total = len(games)
    n_fail_success = n_not_game = 0
    records: list[dict] = []
    for g in games:
        ad = g.get("app_details") or {}
        if not ad.get("success"):
            n_fail_success += 1
            continue
        if (ad.get("data") or {}).get("type") != "game":
            n_not_game += 1
            continue
        rec = build_record(g)
        if rec is not None:
            records.append(rec)
    print(f"[filter] type=game & success: {len(records)}")

    # --- 중복 제거 (appid, 그리고 정제 텍스트 완전 동일) ---
    seen_appid: set = set()
    seen_text: set = set()
    deduped: list[dict] = []
    n_dup_appid = n_dup_text = 0
    for rec in records:
        if rec["appid"] in seen_appid:
            n_dup_appid += 1
            continue
        seen_appid.add(rec["appid"])
        key = rec["detailed_description_clean"]
        if key and key in seen_text:
            n_dup_text += 1
            continue
        seen_text.add(key)
        deduped.append(rec)
    print(f"[dedup] appid중복 {n_dup_appid}, 텍스트중복 {n_dup_text} → {len(deduped)}")

    # --- usable 필터 (정제 후 최소 길이) ---
    usable = [r for r in deduped if r["len_clean"] >= args.min_chars]
    n_too_short = len(deduped) - len(usable)
    print(f"[usable] 정제후 <{args.min_chars}자 제외 {n_too_short} → {len(usable)}")

    # --- 정제 전후 통계 ---
    stats = {
        "input_file": args.input,
        "counts": {
            "total_records": n_total,
            "dropped_success_false": n_fail_success,
            "dropped_not_game": n_not_game,
            "after_filter": len(records),
            "dropped_dup_appid": n_dup_appid,
            "dropped_dup_text": n_dup_text,
            "after_dedup": len(deduped),
            "dropped_too_short": n_too_short,
            "usable": len(usable),
        },
        "length_before_clean": length_stats([r["len_raw"] for r in usable]),
        "length_after_clean": length_stats([r["len_clean"] for r in usable]),
        "short_description": {
            "present": sum(1 for r in usable if r["short_description"]),
            "median_chars": length_stats(
                [len(r["short_description"]) for r in usable if r["short_description"]]
            ).get("median", 0),
        },
        "min_chars_threshold": args.min_chars,
        "seed": SEED,
    }

    # --- train/val/test 분할 ---
    rng = random.Random(SEED)
    shuffled = usable[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
    stats["split"] = {k: len(v) for k, v in splits.items()}

    # --- 저장 ---
    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "clean.jsonl", usable)
    for name in SPLIT_NAMES:
        write_jsonl(out_dir / f"{name}.jsonl", splits[name])
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # --- 리포트 출력 ---
    print("\n=== 정제 전후 통계 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[done] 산출물 → {out_dir}")


if __name__ == "__main__":
    main()
