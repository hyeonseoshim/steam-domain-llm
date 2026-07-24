"""Part A · Step 3 — 프론티어 baseline arm (3대장을 우리 태스크로 예측).

손익분기([[cost_crossover]])가 최저가 프론티어(Gemini Flash-Lite)와는 24만서 2.6배로
좁혀졌다 → "그럼 그냥 그거 쓰지"에 답하려면 **품질**을 재야 한다. 프론티어를 우리
gold_test 프롬프트로 zero-shot 돌려 **완전히 동일한 하브니스**(baseline_predict의
SYSTEM·user_prompt·parse_summary, 입력 8000자 절단)로 예측을 만들고, run_eval(kiwi)+
bertscore로 채점해 LoRA(형식준수 99.5%, R1 .595, BERT-F .788)와 비교한다.

가설: 프론티어는 **의미(BERTScore)는 대등**하겠지만 **형식준수·ROUGE(우리 고정 스키마·
중립 스타일 일치)는 파인튜닝이 이긴다** — DB 적재용 결정성이 값하는 지점. 특히 최저가
Flash-Lite급이 약하면 "품질 동급이면 N배 싸다"의 '동급' 전제가 무너짐을 실증.

⚠️ 프론티어 API 유료 호출(.env 키). 샘플 소량(--limit)으로. 프로바이더별 SDK.

usage:
    uv run --with google-genai --with python-dotenv part_a/eval/frontier_predict.py \
        --provider gemini --model gemini-2.5-flash-lite --limit 60 \
        --out part_a/eval/preds_frontier_flashlite.jsonl
    # anthropic: --with anthropic  --provider anthropic --model claude-haiku-4-5
    # openai:    --with openai     --provider openai    --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from baseline_predict import build_messages, load_jsonl, parse_summary  # noqa: E402


def _split_system(msgs: list[dict]) -> tuple[str, list[dict]]:
    """system 메시지를 분리(Anthropic/Gemini는 top-level 파라미터로 받음)."""
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    conv = [m for m in msgs if m["role"] != "system"]
    return system, conv


def call_openai(model: str, msgs: list[dict], max_new: int) -> str:
    from openai import OpenAI  # noqa: PLC0415
    client = OpenAI()
    # ⚠️ GPT-5 계열은 추론모델 → 추론토큰이 max_completion_tokens 를 먹어 답이 잘림.
    # reasoning_effort="minimal" 로 추론 최소화 + 토큰 여유. (추론토큰=출력과금 →
    # GPT-5 실질 원가는 순수요약보다 높음 = 손익분기에선 프론티어 불리.)
    kw = dict(model=model, messages=msgs, max_completion_tokens=max(max_new, 1500))
    try:
        r = client.chat.completions.create(reasoning_effort="minimal", **kw)
    except Exception:  # noqa: BLE001 — 비추론모델이면 파라미터 거부 → 재시도
        r = client.chat.completions.create(**kw)
    return r.choices[0].message.content or ""


def call_anthropic(model: str, msgs: list[dict], max_new: int) -> str:
    from anthropic import Anthropic  # noqa: PLC0415
    client = Anthropic()
    system, conv = _split_system(msgs)
    r = client.messages.create(
        model=model, max_tokens=max_new, system=system, messages=conv)
    return "".join(b.text for b in r.content if b.type == "text")


def call_gemini(model: str, msgs: list[dict], max_new: int) -> str:
    from google import genai  # noqa: PLC0415
    from google.genai import types  # noqa: PLC0415
    client = genai.Client()
    system, conv = _split_system(msgs)
    contents = [
        types.Content(role=("user" if m["role"] == "user" else "model"),
                      parts=[types.Part(text=m["content"])])
        for m in conv
    ]
    r = client.models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_new, temperature=0))
    return r.text or ""


PROVIDERS = {"openai": call_openai, "anthropic": call_anthropic, "gemini": call_gemini}


def _with_retry(fn, *a, tries: int = 6):
    """레이트리밋(429/RESOURCE_EXHAUSTED)·일시오류에 지수백오프 재시도.

    무료 티어 RPM 한도(예: Gemini 15 RPM) 때문에 429가 뜨는데, 이건 모델 품질과
    무관한 하브니스 아티팩트다 → 재시도해서 예측 자체는 반드시 받아낸다.
    """
    import re as _re  # noqa: PLC0415
    delay = 5.0
    for attempt in range(tries):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = any(t in msg for t in
                            ("429", "RESOURCE_EXHAUSTED", "rate", "overloaded",
                             "503", "500", "529", "timeout"))
            if not transient or attempt == tries - 1:
                raise
            m = _re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", msg)
            wait = float(m.group(1)) + 1 if m else delay
            time.sleep(wait)
            delay = min(delay * 1.6, 60)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=list(PROVIDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--gold", default="part_a/data/references/gold_test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=60, help="샘플 건수(비용 제어)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--sleep", type=float, default=0.3, help="레이트리밋 완화 간격(초)")
    args = ap.parse_args()

    from dotenv import load_dotenv  # noqa: PLC0415
    load_dotenv()

    call = PROVIDERS[args.provider]
    test = load_jsonl(Path(args.gold))[:args.limit]
    out_path = Path(args.out)
    n_ok = n_bad = n_err = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(test, 1):
            msgs = build_messages(rec, shots=[])  # zero-shot(유저가 그냥 프롬프트하듯)
            try:
                raw = _with_retry(call, args.model, msgs, args.max_new)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                raw = f"[ERROR] {type(e).__name__}: {e}"
            summary = parse_summary(raw)
            n_ok += summary is not None
            n_bad += summary is None
            f.write(json.dumps({"appid": rec["appid"], "summary": summary,
                                "raw": raw}, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {i}/{len(test)}  파싱성공 {n_ok} 실패 {n_bad} 에러 {n_err}")
            time.sleep(args.sleep)

    print(f"[frontier:{args.provider}/{args.model}] {len(test)}건 "
          f"파싱성공 {n_ok} / 실패 {n_bad} / 호출에러 {n_err} → {out_path}")
    print(f"  다음: run_eval.py --pred {out_path} (kiwi) + bertscore.py 로 LoRA와 비교")


if __name__ == "__main__":
    main()
