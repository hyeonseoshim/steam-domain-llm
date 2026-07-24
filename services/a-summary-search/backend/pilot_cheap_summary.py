"""RAG 파일럿 — 싼 프론티어(Gemini Flash-Lite)로 검색용 한국어 요약 생성.

목적: "파인튜닝 정답지가 아니라 검색 문서니 싼 모델로 충분하다"를 실측 검증.
gold(Opus)와 동일 스키마·시스템 프롬프트로, 60개 평가 대상 게임의 원문 설명
(gold `input`)을 Gemini Flash-Lite 로 요약한다. 이후 pilot_eval.py 가 이 요약과
gold 요약을 같은 60질의로 검색해 Recall@K·MRR 을 비교한다.

usage:
    uv run --with google-genai --with python-dotenv \
        backend/pilot_cheap_summary.py --model gemini-flash-lite-latest
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
EVAL = Path(__file__).resolve().parent / "eval_queries.jsonl"
OUT = Path(__file__).resolve().parent / "pilot_cheap_summ.jsonl"

# gold generate_references.py 와 동일한 규칙(공정 비교).
SYSTEM = """너는 Steam 게임 설명을 한국어로 요약하는 중립적 편집자다. 아래 규칙을 반드시 지켜 요약한다.
1. 오직 제공된 게임 설명에 근거해서만 작성한다. 설명에 없는 정보를 추측·창작하지 않는다(환각 금지).
2. 마케팅 과장·감탄·홍보 문구를 제거하고 사실 위주 중립 톤으로 쓴다.
3. 세 필드: 장르(명사구만, 괄호·영어병기 금지, 쉼표 나열) / 핵심플레이(플레이어의 핵심 행동·루프 1~2문장) / 특징(구별짓는 세계관·시스템·연출 1~2문장).
4. 고유명사는 원문 표기 유지 가능.
5. 불릿·이모지·머리말("이 게임은") 없이 내용만.
반드시 JSON 하나만 출력: {"장르": "...", "핵심플레이": "...", "특징": "..."}"""


def targets() -> list[dict]:
    """60 평가 대상 appid 에 해당하는 gold 레코드(원문 input 포함)."""
    want = {json.loads(l)["appid"] for l in EVAL.open(encoding="utf-8") if l.strip()}
    rows = []
    for line in GOLD.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["appid"] in want and (r.get("input") or "").strip():
            rows.append(r)
    return rows


def gen(client, model: str, rec: dict) -> dict | None:
    from google.genai import types  # noqa: PLC0415
    genres = ", ".join(rec.get("genres") or []) or "(정보 없음)"
    user = f"게임명: {rec.get('name','')}\nSteam 장르 태그: {genres}\n\n[게임 설명]\n{rec['input']}"
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
    ap.add_argument("--sleep", type=float, default=4.5)  # 무료티어 15RPM
    args = ap.parse_args()

    from dotenv import load_dotenv  # noqa: PLC0415
    from google import genai  # noqa: PLC0415
    load_dotenv()
    client = genai.Client()

    rows = targets()
    print(f"[pilot] 대상 {len(rows)}게임 · 모델={args.model}")
    ok = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(rows, 1):
            summ = None
            for attempt in range(4):
                try:
                    summ = gen(client, args.model, rec); break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        print(f"  [err] {rec['appid']}: {str(e)[:50]}")
                    else:
                        time.sleep(8)
            if summ:
                ok += 1
                f.write(json.dumps({"appid": rec["appid"], "name": rec.get("name", ""),
                                    "summary": summ}, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {i}/{len(rows)} 생성 {ok}")
            time.sleep(args.sleep)
    print(f"[pilot] 요약 {ok}개 → {OUT}")


if __name__ == "__main__":
    main()
