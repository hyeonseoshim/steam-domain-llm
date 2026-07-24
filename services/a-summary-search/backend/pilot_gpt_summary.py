"""RAG 파일럿 (GPT) — OpenAI 싼 모델로 검색용 한국어 요약 생성.

pilot_cheap_summary.py 의 GPT 판. 같은 60게임·같은 시스템 프롬프트로 GPT-5 nano(또는
지정 모델)를 써서 요약 → pilot_gpt_summ.jsonl. pilot_eval.py 가 gold/Gemini 와 3자 비교.

usage:
    uv run --with openai --with python-dotenv backend/pilot_gpt_summary.py --model gpt-5-nano
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
EVAL = Path(__file__).resolve().parent / "eval_queries.jsonl"
OUT = Path(__file__).resolve().parent / "pilot_gpt_summ.jsonl"

SYSTEM = """너는 Steam 게임 설명을 한국어로 요약하는 중립적 편집자다. 아래 규칙을 반드시 지켜 요약한다.
1. 오직 제공된 게임 설명에 근거해서만 작성한다. 설명에 없는 정보를 추측·창작하지 않는다(환각 금지).
2. 마케팅 과장·감탄·홍보 문구를 제거하고 사실 위주 중립 톤으로 쓴다.
3. 세 필드: 장르(명사구만, 괄호·영어병기 금지, 쉼표 나열) / 핵심플레이(플레이어의 핵심 행동·루프 1~2문장) / 특징(구별짓는 세계관·시스템·연출 1~2문장).
4. 고유명사는 원문 표기 유지 가능.
5. 불릿·이모지·머리말("이 게임은") 없이 내용만.
반드시 JSON 하나만 출력: {"장르": "...", "핵심플레이": "...", "특징": "..."}"""


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


def gen(client, model: str, rec: dict, reasoning: str | None) -> dict | None:
    genres = ", ".join(rec.get("genres") or []) or "(정보 없음)"
    user = f"게임명: {rec.get('name','')}\nSteam 장르 태그: {genres}\n\n[게임 설명]\n{rec['input']}"
    kw = {}
    if reasoning:  # gpt-5 계열: 추론 예산 최소화(요약엔 추론 불필요) + 출력 여유
        kw["reasoning_effort"] = reasoning
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_completion_tokens=1500, **kw)
    txt = r.choices[0].message.content or ""
    try:
        s = json.loads(txt.strip())
        if all(k in s for k in ("장르", "핵심플레이", "특징")):
            return {k: str(s[k]).strip() for k in ("장르", "핵심플레이", "특징")}
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--reasoning", default=None, help="gpt-5 계열: minimal|low|medium|high")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()
    out_path = Path(args.out)

    from dotenv import load_dotenv  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415
    load_dotenv()
    client = OpenAI()

    rows = targets()
    print(f"[pilot-gpt] 대상 {len(rows)}게임 · 모델={args.model} · reasoning={args.reasoning}")
    ok = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(rows, 1):
            summ = None
            for attempt in range(4):
                try:
                    summ = gen(client, args.model, rec, args.reasoning); break
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
    print(f"[pilot-gpt] 요약 {ok}개 → {out_path}")


if __name__ == "__main__":
    main()
