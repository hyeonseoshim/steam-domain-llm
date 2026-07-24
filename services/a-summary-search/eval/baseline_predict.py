"""Part A · Step 3 — 베이스라인 예측 생성 (Qwen2.5-3B-Instruct, zero/few-shot).

파인튜닝 '전' 기준선. 베이스 모델에 gold_test 입력을 넣어 3필드 요약을 생성하고,
JSON 파싱해 predictions jsonl 로 저장 → run_eval.py 로 채점.
파인튜닝 후 같은 스크립트(--adapter)로 예측을 다시 뽑아 동일 하브니스로 비교한다.

⚠️ torch/transformers/GPU 필요 → 로컬 WSL(무 GPU) 아님. **Lightning AI Studio에서 실행.**
  pip: transformers, torch, accelerate  (few-shot 예시는 gold_train 에서 추출)

usage (Lightning):
    python eval/baseline_predict.py --shots 0 --out preds_zeroshot.jsonl
    python eval/baseline_predict.py --shots 2 --out preds_2shot.jsonl
    # 파인튜닝 후: --adapter path/to/qlora_adapter
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_INPUT_CHARS = 8000  # 긴 본문 절단(메모리·시간 제어; 서빙 단계서 청킹 별도 실험)

SYSTEM = """\
너는 Steam 게임 설명을 한국어로 요약하는 중립적 편집자다. 규칙:
1. 오직 제공된 게임 설명에 근거해서만 작성한다(환각 금지).
2. 마케팅 과장·홍보 문구를 제거하고 사실 위주 중립 톤으로 쓴다.
3. 세 필드로 나눈다 — 장르: 명사구(괄호·영어병기 없이, 여러 개는 쉼표) / 핵심플레이: 1~2문장 / 특징: 1~2문장.
4. 머리말·불릿·이모지 없이 내용만.
5. 반드시 아래 JSON 형식으로만 출력한다:
{"장르": "...", "핵심플레이": "...", "특징": "..."}"""


def user_prompt(rec: dict) -> str:
    genres = ", ".join(rec.get("genres") or []) or "(정보 없음)"
    body = (rec.get("input") or "")[:MAX_INPUT_CHARS]
    return (f"게임명: {rec.get('name', '')}\n"
            f"Steam 장르 태그: {genres}\n\n[게임 설명]\n{body}")


def build_messages(rec: dict, shots: list[dict]) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:  # few-shot: 입력→정답 JSON
        msgs.append({"role": "user", "content": user_prompt(ex)})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex["summary"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_prompt(rec)})
    return msgs


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_summary(text: str) -> dict | None:
    """모델 출력에서 3필드 JSON 추출. 실패 시 None(형식 위반으로 집계)."""
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(k in obj for k in ("장르", "핵심플레이", "특징")):
        return None
    return {k: str(obj.get(k, "")) for k in ("장르", "핵심플레이", "특징")}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="part_a/data/references/gold_test.jsonl")
    ap.add_argument("--train", default="part_a/data/references/gold_train.jsonl",
                    help="few-shot 예시 추출용")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--adapter", default="", help="QLoRA 어댑터 경로(파인튜닝 후)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="디버그: 앞 N건만")
    args = ap.parse_args()

    import torch  # noqa: PLC0415 — GPU 환경에서만 import
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel  # noqa: PLC0415
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    test = load_jsonl(Path(args.gold))
    if args.limit:
        test = test[:args.limit]
    shots = load_jsonl(Path(args.train))[:args.shots] if args.shots else []

    out_path = Path(args.out)
    n_ok = n_bad = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(test, 1):
            text = tok.apply_chat_template(
                build_messages(rec, shots), tokenize=False,
                add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new,
                                     do_sample=False)
            out = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            summary = parse_summary(out)
            n_ok += summary is not None
            n_bad += summary is None
            f.write(json.dumps({"appid": rec["appid"], "summary": summary,
                                "raw": out}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(test)}  파싱성공 {n_ok} 실패 {n_bad}")

    print(f"[predict] {len(test)}건  JSON파싱 성공 {n_ok} / 실패 {n_bad} → {out_path}")
    print(f"  다음: python eval/run_eval.py --pred {out_path} --gold {args.gold}")


if __name__ == "__main__":
    main()
