"""Part A · Step 2 — 레퍼런스 요약(정답 레이블) 생성.

입력 게임 설명(영어, detailed_description_clean)을 Claude API로
한국어 고정 포맷 요약(장르 / 핵심플레이 / 특징)으로 변환한다.

silver → (사람 검수) → gold 파이프라인의 silver 생성 단계.
구조화 출력(structured outputs)으로 3필드 포맷을 강제하고,
한 건씩 즉시 append 저장 → 중간에 끊겨도 --resume 으로 이어서 생성.

파일럿(순차) 모드만 우선 구현. 프롬프트 검증 후 전체는 Batch API로 확장 예정.

usage:
    uv run python generate_references.py --n 40
    uv run python generate_references.py --n 40 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

MODEL = "claude-opus-4-8"

# 요약 3필드 스키마 — 구조화 출력으로 포맷을 강제한다.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "장르": {
            "type": "string",
            "description": "게임의 장르와 분류. 예: '액션 RPG, 트윈스틱 슈터'",
        },
        "핵심플레이": {
            "type": "string",
            "description": "플레이어가 실제로 하는 핵심 행동/루프. 1~2문장.",
        },
        "특징": {
            "type": "string",
            "description": "이 게임을 구별짓는 세계관·시스템·연출 등의 특징. 1~2문장.",
        },
    },
    "required": ["장르", "핵심플레이", "특징"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
너는 Steam 게임 설명을 한국어로 요약하는 중립적 편집자다.
아래 규칙을 반드시 지켜 요약한다.

1. 오직 제공된 게임 설명에 근거해서만 작성한다. 설명에 없는 정보를 추측·창작하지 않는다 (환각 금지).
2. 마케팅 과장 표현("최고의", "숨막히는", "당신을 사로잡을")과 감탄·홍보 문구를 제거하고, 사실 위주의 중립적 톤으로 쓴다.
3. 세 필드로 나눈다:
   - 장르: 게임의 장르/분류를 명사구로만 쓴다. 괄호·부연 설명·영어 원어 병기를 넣지 말 것 (예: "1인칭 슈터" O / "1인칭 슈터(레트로 FPS)" X). 여러 장르는 쉼표로 나열한다.
   - 핵심플레이: 플레이어가 실제로 하는 핵심 행동과 게임 루프. 1~2문장.
   - 특징: 이 게임을 구별짓는 세계관/시스템/연출 등. 1~2문장.
4. 자연스러운 한국어로 쓰되, 고유명사(게임명·지명 등)는 원문 표기를 유지해도 된다.
5. 각 필드는 간결하게. 불릿·이모지·머리말("이 게임은") 없이 내용만.
"""


def build_user_prompt(rec: dict) -> str:
    genres = ", ".join(rec.get("genres") or []) or "(정보 없음)"
    return (
        f"게임명: {rec.get('name', '')}\n"
        f"Steam 장르 태그: {genres}\n\n"
        f"[게임 설명]\n{rec['detailed_description_clean']}"
    )


def load_done_appids(out_path: Path) -> set[int]:
    """이미 생성 완료된 appid 집합 (resume용)."""
    done: set[int] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["appid"])
    return done


def generate_one(client: anthropic.Anthropic, rec: dict) -> dict:
    """한 게임에 대한 한국어 3필드 요약을 생성."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(rec)}],
        output_config={"format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    summary = json.loads(text)
    return {
        "appid": rec["appid"],
        "name": rec.get("name", ""),
        "genres": rec.get("genres") or [],
        "len_clean": rec.get("len_clean"),
        "input_clean": rec["detailed_description_clean"],
        "short_description": rec.get("short_description", ""),  # 참고용(정답 아님)
        "summary": summary,
        "model": MODEL,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/processed/train.jsonl",
        help="입력 jsonl (전처리 산출물)",
    )
    ap.add_argument(
        "--out",
        default="data/references/pilot.jsonl",
        help="출력 jsonl (silver 요약)",
    )
    ap.add_argument("--n", type=int, default=40, help="파일럿 생성 개수")
    ap.add_argument(
        "--resume", action="store_true", help="기존 출력의 appid는 건너뛰고 이어서 생성"
    )
    args = ap.parse_args()

    load_dotenv()  # .env 의 ANTHROPIC_API_KEY 로드
    client = anthropic.Anthropic()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_appids(out_path) if args.resume else set()
    if not args.resume and out_path.exists():
        print(f"[!] {out_path} 가 이미 존재합니다. 이어서 하려면 --resume, "
              f"다시 하려면 파일을 지우세요.", file=sys.stderr)
        sys.exit(1)

    # 입력에서 파일럿 대상 n개 선택 (done 제외)
    targets: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["appid"] in done:
                continue
            targets.append(rec)
            if len(targets) >= args.n:
                break

    print(f"[gen] 대상 {len(targets)}건 (이미 완료 {len(done)}건 skip), model={MODEL}")
    tot_in = tot_out = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(targets, 1):
            try:
                result = generate_one(client, rec)
            except Exception as e:  # noqa: BLE001 — 개별 실패는 건너뛰고 계속
                print(f"  [{i}/{len(targets)}] appid={rec['appid']} 실패: {e}",
                      file=sys.stderr)
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()  # 크래시 대비 즉시 기록
            tot_in += result["usage"]["input_tokens"]
            tot_out += result["usage"]["output_tokens"]
            print(f"  [{i}/{len(targets)}] {result['name']} "
                  f"(in {result['usage']['input_tokens']}, "
                  f"out {result['usage']['output_tokens']})")
            time.sleep(0.3)  # 가벼운 레이트 완화

    # 대략 비용 추정 (Opus 4.8: $5/M in, $25/M out)
    est = tot_in / 1e6 * 5 + tot_out / 1e6 * 25
    print(f"\n[done] 토큰 in={tot_in:,} out={tot_out:,}  →  약 ${est:.2f} "
          f"(파일럿 {len(targets)}건 기준)")
    print(f"       전체 4,165건 환산 시 대략 ${est / max(len(targets),1) * 4165:.2f} "
          f"(Batch API 사용 시 절반)")
    print(f"[out]  {out_path}")


if __name__ == "__main__":
    main()
