"""RAG 파일럿 (Haiku) — 중간 티어 요약 생성.

gold(Opus)와 동일 스키마·시스템 프롬프트·구조화출력으로 claude-haiku-4-5 요약.
싼 티어(.44~.46)와 Opus(.565) 사이 무릎을 확인. → pilot_eval.py 로 비교.

usage:
    uv run --with anthropic --with python-dotenv backend/pilot_haiku_summary.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
EVAL = Path(__file__).resolve().parent / "eval_queries.jsonl"

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "장르": {"type": "string"},
        "핵심플레이": {"type": "string"},
        "특징": {"type": "string"},
    },
    "required": ["장르", "핵심플레이", "특징"],
    "additionalProperties": False,
}
SYSTEM = """너는 Steam 게임 설명을 한국어로 요약하는 중립적 편집자다. 아래 규칙을 반드시 지켜 요약한다.
1. 오직 제공된 게임 설명에 근거해서만 작성한다. 설명에 없는 정보를 추측·창작하지 않는다(환각 금지).
2. 마케팅 과장·감탄·홍보 문구를 제거하고 사실 위주 중립 톤으로 쓴다.
3. 세 필드: 장르(명사구만, 괄호·영어병기 금지, 쉼표 나열) / 핵심플레이(플레이어의 핵심 행동·루프 1~2문장) / 특징(구별짓는 세계관·시스템·연출 1~2문장).
4. 고유명사는 원문 표기 유지 가능.
5. 불릿·이모지·머리말("이 게임은") 없이 내용만."""


def targets() -> list[dict]:
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
    genres = ", ".join(rec.get("genres") or []) or "(정보 없음)"
    user = f"게임명: {rec.get('name','')}\nSteam 장르 태그: {genres}\n\n[게임 설명]\n{rec['input']}"
    r = client.messages.create(
        model=model, max_tokens=1024, system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}})
    txt = next(b.text for b in r.content if b.type == "text")
    try:
        s = json.loads(txt)
        return {k: str(s[k]).strip() for k in ("장르", "핵심플레이", "특징")}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--out", default="backend/pilot_haiku_summ.jsonl")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    import anthropic  # noqa: PLC0415
    from dotenv import load_dotenv  # noqa: PLC0415
    load_dotenv("/mnt/c/Users/jungs/Downloads/steam-domain-specific-llm/.env")
    client = anthropic.Anthropic()

    rows = targets()
    print(f"[pilot-haiku] 대상 {len(rows)}게임 · 모델={args.model}")
    ok = 0
    with Path(args.out).open("w", encoding="utf-8") as f:
        for i, rec in enumerate(rows, 1):
            summ = None
            for attempt in range(4):
                try:
                    summ = gen(client, args.model, rec); break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        print(f"  [err] {rec['appid']}: {str(e)[:80]}")
                    else:
                        time.sleep(5)
            if summ:
                ok += 1
                f.write(json.dumps({"appid": rec["appid"], "name": rec.get("name", ""),
                                    "summary": summ}, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {i}/{len(rows)} 생성 {ok}")
            time.sleep(args.sleep)
    print(f"[pilot-haiku] 요약 {ok}개 → {args.out}")


if __name__ == "__main__":
    main()
