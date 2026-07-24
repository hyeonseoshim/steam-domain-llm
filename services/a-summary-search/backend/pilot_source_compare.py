"""RAG 파이프라인 Step0 — 요약 소스 검증 (short_description vs detailed).

pilot_cheap_summary.py 는 gold `input`(=detailed 원문)을 소스로 Gemini 요약을 만들어
검색 MRR .462 를 얻었다. 여기선 같은 60게임을 **CSV short_description**(우리가 이미 가진
150K 소스)로 요약해, detailed 없이도 충분한지 잰다. short ≈ detailed 면 무거운 PG추출
없이 CSV 만으로 150K 코퍼스를 만들 수 있다.

usage:
    uv run --with google-genai --with python-dotenv backend/pilot_source_compare.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from pilot_cheap_summary import SYSTEM  # 동일 시스템 프롬프트 재사용

EVAL = Path(__file__).resolve().parent / "eval_queries.jsonl"
GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
CSV_APP = Path("steam_dataset_2025_csv/applications.csv")
OUT = Path(__file__).resolve().parent / "pilot_short_summ.jsonl"


def load_targets() -> dict[int, dict]:
    """60 평가대상 appid → {name, genres, short}. name/genres 는 gold, short 는 CSV."""
    want = {json.loads(l)["appid"] for l in EVAL.open(encoding="utf-8") if l.strip()}
    meta = {}
    for line in GOLD.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["appid"] in want:
            meta[r["appid"]] = {"name": r.get("name", ""), "genres": r.get("genres") or [], "short": ""}
    csv.field_size_limit(10**7)
    with CSV_APP.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                aid = int(row["appid"])
            except (ValueError, TypeError):
                continue
            if aid in meta:
                meta[aid]["short"] = (row.get("short_description") or "").strip()
    return meta


def gen(client, model: str, m: dict) -> dict | None:
    from google.genai import types  # noqa: PLC0415
    genres = ", ".join(m["genres"]) or "(정보 없음)"
    user = f"게임명: {m['name']}\nSteam 장르 태그: {genres}\n\n[게임 설명]\n{m['short']}"
    r = client.models.generate_content(
        model=model, contents=f"{SYSTEM}\n\n{user}",
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           max_output_tokens=400, temperature=0.3))
    try:
        s = json.loads((r.text or "").strip())
        if all(k in s for k in ("장르", "핵심플레이", "특징")):
            return {k: str(s[k]).strip() for k in ("장르", "핵심플레이", "특징")}
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--sleep", type=float, default=4.5)
    args = ap.parse_args()

    from dotenv import load_dotenv  # noqa: PLC0415
    from google import genai  # noqa: PLC0415
    load_dotenv("/mnt/c/Users/jungs/Downloads/steam-domain-specific-llm/.env")
    client = genai.Client()

    meta = load_targets()
    have_short = sum(1 for m in meta.values() if m["short"])
    print(f"[step0] 대상 {len(meta)}게임 · short_description 있는 것 {have_short}개 · 모델={args.model}")
    ok = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, (aid, m) in enumerate(meta.items(), 1):
            if not m["short"]:
                continue
            summ = None
            for attempt in range(4):
                try:
                    summ = gen(client, args.model, m); break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        print(f"  [err] {aid}: {str(e)[:50]}")
                    else:
                        time.sleep(8)
            if summ:
                ok += 1
                f.write(json.dumps({"appid": aid, "name": m["name"], "summary": summ},
                                   ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {i}/{len(meta)} 생성 {ok}")
            time.sleep(args.sleep)
    print(f"[step0] short기반 요약 {ok}개 → {OUT}")


if __name__ == "__main__":
    main()
