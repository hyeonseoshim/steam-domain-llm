"""Part A · Step 2 — 전체 레퍼런스 요약 생성 (Message Batches API).

clean.jsonl(usable 4,165건 = train+val+test 합집합) 전체를 Batch API로
한국어 3필드 요약(장르/핵심플레이/특징)으로 변환한다. 표준 대비 50% 저렴.

크래시 내성:
  - 제출 시 batch_id 를 batch_state.json 에 저장.
  - 로컬 Claude Code / 이 스크립트가 죽어도, 배치는 Anthropic 서버에서 계속 처리됨.
  - 재실행하면 저장된 batch_id 로 상태를 이어서 폴링하고 결과를 수거한다.

프롬프트/스키마는 generate_references.py 에서 재사용(단일 소스).

usage:
    # 1) 제출 (아직 all.jsonl 에 없는 appid만) + 완료까지 폴링 + 결과 수거
    uv run python generate_references_batch.py

    # 상태만 확인
    uv run python generate_references_batch.py --status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

# 프롬프트/스키마/입력빌더를 파일럿 스크립트와 공유 (단일 소스 오브 트루스)
from generate_references import (  # type: ignore  # noqa: E402
    MODEL,
    SUMMARY_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def load_done_appids(out_path: Path) -> set[int]:
    done: set[int] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["appid"])
    return done


def load_input(in_path: Path) -> dict[int, dict]:
    recs: dict[int, dict] = {}
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["appid"]] = r
    return recs


def build_requests(recs: dict[int, dict], todo: list[int]) -> list[Request]:
    reqs: list[Request] = []
    for appid in todo:
        rec = recs[appid]
        reqs.append(
            Request(
                custom_id=str(appid),
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_user_prompt(rec)}],
                    output_config={
                        "format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}
                    },
                ),
            )
        )
    return reqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/clean.jsonl")
    ap.add_argument("--out", default="data/references/all.jsonl")
    ap.add_argument(
        "--state", default="data/references/batch_state.json",
        help="batch_id 저장 파일 (크래시 후 재개용)",
    )
    ap.add_argument("--status", action="store_true", help="상태만 출력하고 종료")
    args = ap.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()

    out_path = Path(args.out)
    state_path = Path(args.state)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recs = load_input(Path(args.input))
    done = load_done_appids(out_path)

    # --- 1) batch_id 확보: 기존 상태 재개 or 신규 제출 ---
    batch_id: str | None = None
    if state_path.exists():
        batch_id = json.loads(state_path.read_text())["batch_id"]
        print(f"[resume] 기존 배치 이어받기: {batch_id}")

    if args.status:
        if not batch_id:
            print("진행 중인 배치 없음.")
            return
        b = client.messages.batches.retrieve(batch_id)
        print(f"status={b.processing_status}  counts={b.request_counts}")
        return

    if batch_id is None:
        todo = [aid for aid in recs if aid not in done]
        if not todo:
            print(f"[done] 이미 전부 완료 ({len(done)}건). 할 일 없음.")
            return
        print(f"[submit] 신규 배치: {len(todo)}건 (완료 {len(done)}건 skip)")
        batch = client.messages.batches.create(requests=build_requests(recs, todo))
        batch_id = batch.id
        state_path.write_text(json.dumps({"batch_id": batch_id}, indent=2))
        print(f"[submit] batch_id={batch_id} → {state_path} 저장")

    # --- 2) 완료까지 폴링 (크래시 나도 재실행하면 여기로 복귀) ---
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"  status={b.processing_status}  "
              f"처리중={c.processing} 성공={c.succeeded} 실패={c.errored}")
        if b.processing_status == "ended":
            break
        time.sleep(30)

    # --- 3) 결과 수거 → all.jsonl append ---
    n_ok = n_err = 0
    with out_path.open("a", encoding="utf-8") as f:
        for res in client.messages.batches.results(batch_id):
            appid = int(res.custom_id)
            if res.result.type != "succeeded":
                n_err += 1
                print(f"  [err] appid={appid}: {res.result.type}", file=sys.stderr)
                continue
            msg = res.result.message
            text = next(b.text for b in msg.content if b.type == "text")
            rec = recs[appid]
            f.write(json.dumps({
                "appid": appid,
                "name": rec.get("name", ""),
                "genres": rec.get("genres") or [],
                "len_clean": rec.get("len_clean"),
                "input_clean": rec["detailed_description_clean"],
                "short_description": rec.get("short_description", ""),
                "summary": json.loads(text),
                "model": MODEL,
                "usage": {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                },
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"\n[fetch] 성공 {n_ok}건, 실패 {n_err}건 → {out_path}")
    # 배치 완료 후 상태파일 제거(실패분은 재실행 시 새 배치로 재시도)
    state_path.unlink(missing_ok=True)
    if n_err:
        print(f"[!] 실패 {n_err}건은 재실행하면 새 배치로 자동 재시도됩니다.")


if __name__ == "__main__":
    main()
