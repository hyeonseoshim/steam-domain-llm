"""RAG 연습 ① — 검색 평가셋(질의→정답 appid) 생성.

실무통합([[silmu-integration-project]]) 대비 RAG 뼈대 연습. 스팀 카탈로그(우리가
정규화한 한국어 3필드 요약)를 코퍼스로 쓰고, 각 게임에 대해 "유저가 이 게임을 찾을 때
칠 법한 짧은 한국어 검색어"를 프론티어(최저가 Gemini)로 생성해 {query, appid} 평가셋을
만든다. 이걸로 hybrid_search.py 에서 BM25/dense/hybrid 의 Recall@K·MRR 을 잰다.

질의 설계 원칙(하이브리드 이점이 드러나게):
- 게임 제목을 그대로 넣지 않는다(그럼 BM25가 너무 쉬움).
- 플레이 방식·장르·분위기를 자연어로 → 패러프레이즈라 dense 가 유리한 케이스 + 희귀
  고유명사·장르어가 섞이면 BM25 가 유리한 케이스가 공존 → 하이브리드가 둘 다 잡음.

usage:
    uv run --with google-genai --with python-dotenv backend/build_eval.py \
        --n 60 --out backend/eval_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"

PROMPT = """다음 게임을 스팀에서 찾으려는 한국 유저가 검색창에 칠 법한 짧은 검색어를 하나만 만들어줘.
규칙: 제목을 그대로 쓰지 말 것. 플레이 방식·장르·분위기 위주로 5~12자 내외 자연어. 따옴표·설명 없이 검색어만 출력.

[게임]
제목: {name}
장르: {genre}
핵심플레이: {core}
특징: {feat}

검색어:"""


def call_gemini(model: str, prompt: str) -> str:
    from google import genai  # noqa: PLC0415
    from google.genai import types  # noqa: PLC0415
    client = genai.Client()
    r = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=40, temperature=0.4))
    return (r.text or "").strip().splitlines()[0].strip().strip('"' "'").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="질의 생성 개수")
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--out", default="backend/eval_queries.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=4.5)  # 무료티어 15RPM
    args = ap.parse_args()

    from dotenv import load_dotenv  # noqa: PLC0415
    load_dotenv()

    rows = [json.loads(l) for l in GOLD.open(encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    picks = rows[:args.n]

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    n_ok = 0
    with out.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(picks, 1):
            s = rec.get("summary") or {}
            prompt = PROMPT.format(name=rec.get("name", ""), genre=s.get("장르", ""),
                                   core=s.get("핵심플레이", ""), feat=s.get("특징", ""))
            for attempt in range(5):
                try:
                    q = call_gemini(args.model, prompt)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 4:
                        q = ""; print(f"  [err] {rec['appid']}: {str(e)[:60]}")
                    else:
                        time.sleep(8)
            if q:
                n_ok += 1
                f.write(json.dumps({"query": q, "appid": rec["appid"],
                                    "name": rec.get("name", "")}, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {i}/{len(picks)}  생성 {n_ok}")
            time.sleep(args.sleep)

    print(f"[eval] 질의 {n_ok}개 → {out}")


if __name__ == "__main__":
    main()
