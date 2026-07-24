"""Part A · Step 2c 보강 — '얼리 액세스' 환각 타겟 재심사.

생성기가 상투적으로 붙이는 '얼리 액세스/앞서 해보기' 마무리 문구가 원문에 근거 없는
경우가 auto-gold 에 남아 있다(심사 v2가 경미로 처리해 통과). 이를 정리한다.

- 대상: all.jsonl 중 요약에 EA 문구가 있는 건(단 critical 23건은 이미 처리했으므로 제외).
- 방법: judge(Sonnet)가 원문(다국어/로드맵/후원/에피소드예정 함의 포함)이 출시·개발 상태를
  지지하는지 yes/no 판정. no면 **해당 출시상태 절만 제거한 특징**을 직접 재작성(다른 내용 불변).
- 산출: ea_fixes.jsonl(appid→수정 특징; 지지=no만) + ea_fix_report.json.
  build_gold.py 가 ea_fixes.jsonl 을 읽어 gold 에 반영한다.

usage:
    uv run python fix_early_access.py
    uv run python fix_early_access.py --status
    python3 fix_early_access.py --report-only
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

sys.path.insert(0, str(Path(__file__).parent))
import build_gold as B  # noqa: E402  (DROP/FIX/KEEP + resolve_decisions 재사용)

JUDGE_MODEL = "claude-sonnet-4-6"
EA_MARKERS = ("얼리 액세스", "앞서 해보기", "앞서해보기")

SCHEMA = {
    "type": "object",
    "properties": {
        "지지": {"type": "string", "enum": ["yes", "no"],
                 "description": "원문이 게임의 얼리액세스/개발중/출시예정 상태를 지지하는가. 로드맵·계획된 기능·후원 요청·에피소드 예정 등의 함의도 지지로 본다(언어 불문)."},
        "특징_수정": {"type": "string",
                     "description": "지지=yes면 원본 특징을 그대로. 지지=no면 출시·개발 상태를 언급한 문장/절만 제거하고 나머지는 한 글자도 바꾸지 않은 특징."},
    },
    "required": ["지지", "특징_수정"],
    "additionalProperties": False,
}

SYSTEM = """\
너는 게임 요약의 사실성을 점검하는 편집자다. [원문]과 요약의 [특징]을 본다.
[특징]은 게임의 출시·개발 상태(얼리 액세스, 앞서 해보기, 개발 중, 출시 예정 등)를 언급하고 있다.

1) 원문이 그 상태를 지지하는지 판단하라. 원문이 영어/중국어/일본어여도, 그리고
   '계획된 기능', '후원이 실현을 돕는다', '에피소드가 추가될 예정', '로드맵', '데모/베타',
   '아직 개발 중' 같은 **함의**가 있으면 지지(yes)로 본다.
2) 지지(yes)면 '특징_수정'에 원본 [특징]을 그대로 반환한다.
3) 지지하지 않으면(no) — 원문 어디에도 출시·개발 상태 근거가 없으면 — '특징_수정'에
   해당 출시상태를 말하는 문장/절만 삭제하고 **나머지 문장은 한 글자도 바꾸지 않은** 특징을 반환한다.
   문장이 자연스럽게 끝나도록 구두점만 다듬는다. 새로운 내용을 추가하지 마라.
"""


def load_raw(path: Path) -> list[dict]:
    """judge 원응답 로드 (없으면 빈 리스트 — 첫 실행 대비)."""
    return B.load_jsonl(path) if path.exists() else []


def load_targets(all_path: Path, todo_path: Path) -> list[dict]:
    crit = B.resolve_decisions(B.load_jsonl(todo_path))
    out = []
    for r in B.load_jsonl(all_path):
        if r["appid"] in crit:
            continue
        if any(m in json.dumps(r["summary"], ensure_ascii=False) for m in EA_MARKERS):
            out.append(r)
    return out


def build_prompt(rec: dict) -> str:
    return (f"[원문]\n{rec['input_clean']}\n\n"
            f"[특징]\n{rec['summary'].get('특징', '')}")


def build_requests(recs: list[dict]) -> list[Request]:
    return [Request(
        custom_id=str(r["appid"]),
        params=MessageCreateParamsNonStreaming(
            model=JUDGE_MODEL, max_tokens=512, system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(r)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        ),
    ) for r in recs]


def write_outputs(results: list[dict], fixes_path: Path, report_path: Path) -> None:
    """results: [{appid,name,원본특징,지지,특징수정}] → ea_fixes.jsonl + report."""
    changed = [r for r in results if r["지지"] == "no"
               and r["특징수정"].strip() and r["특징수정"] != r["원본특징"]]
    with fixes_path.open("w", encoding="utf-8") as f:
        for r in changed:
            f.write(json.dumps({"appid": r["appid"], "특징": r["특징수정"]},
                               ensure_ascii=False) + "\n")
    report = {
        "judge_model": JUDGE_MODEL,
        "targets": len(results),
        "supported_yes": sum(1 for r in results if r["지지"] == "yes"),
        "unsupported_no": sum(1 for r in results if r["지지"] == "no"),
        "applied_fixes": len(changed),
        "samples": [{"name": r["name"], "before": r["원본특징"], "after": r["특징수정"]}
                    for r in changed[:8]],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[ea] 대상 {report['targets']}  지지 yes {report['supported_yes']}  "
          f"no {report['unsupported_no']}  → 실제 수정 {report['applied_fixes']}")
    print(f"[out] {fixes_path.name}({len(changed)}) / {report_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="data/references/all.jsonl")
    ap.add_argument("--todo", default="data/references/review_todo.jsonl")
    ap.add_argument("--raw", default="data/references/ea_raw.jsonl",
                    help="judge 원응답 저장(재개용)")
    ap.add_argument("--fixes", default="data/references/ea_fixes.jsonl")
    ap.add_argument("--report", default="data/references/ea_fix_report.json")
    ap.add_argument("--state", default="data/references/ea_batch_state.json")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    all_path, todo_path = Path(args.all), Path(args.todo)
    raw_path, fixes_path = Path(args.raw), Path(args.fixes)
    report_path, state_path = Path(args.report), Path(args.state)

    targets = load_targets(all_path, todo_path)
    orig = {r["appid"]: r["summary"].get("특징", "") for r in targets}
    names = {r["appid"]: r.get("name", "") for r in targets}

    def collect(raw: list[dict]) -> list[dict]:
        return [{"appid": x["appid"], "name": names.get(x["appid"], ""),
                 "원본특징": orig.get(x["appid"], ""),
                 "지지": x["지지"], "특징수정": x["특징_수정"]} for x in raw]

    if args.report_only:
        raw = load_raw(raw_path)
        write_outputs(collect(raw), fixes_path, report_path)
        return

    load_dotenv()
    client = anthropic.Anthropic()

    batch_id = None
    if state_path.exists():
        batch_id = json.loads(state_path.read_text())["batch_id"]
        print(f"[resume] {batch_id}")

    if args.status:
        if not batch_id:
            print("진행 중 배치 없음.")
            return
        b = client.messages.batches.retrieve(batch_id)
        print(f"status={b.processing_status} counts={b.request_counts}")
        return

    if batch_id is None:
        done = {r["appid"] for r in load_raw(raw_path)}
        todo = [r for r in targets if r["appid"] not in done]
        if not todo:
            print(f"[done] 이미 전부 처리 ({len(done)}). 집계만.")
            write_outputs(collect(load_raw(raw_path)), fixes_path, report_path)
            return
        print(f"[submit] EA 재심사 {len(todo)}건, model={JUDGE_MODEL}")
        batch = client.messages.batches.create(requests=build_requests(todo))
        batch_id = batch.id
        state_path.write_text(json.dumps({"batch_id": batch_id}, indent=2))
        print(f"[submit] {batch_id}")

    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"  status={b.processing_status} 처리중={c.processing} "
              f"성공={c.succeeded} 실패={c.errored}")
        if b.processing_status == "ended":
            break
        time.sleep(30)

    n_ok = n_err = 0
    with raw_path.open("a", encoding="utf-8") as f:
        for res in client.messages.batches.results(batch_id):
            if res.result.type != "succeeded":
                n_err += 1
                continue
            msg = res.result.message
            text = next(bl.text for bl in msg.content if bl.type == "text")
            obj = json.loads(text)
            f.write(json.dumps({"appid": int(res.custom_id), **obj},
                               ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[fetch] 성공 {n_ok} 실패 {n_err}")
    state_path.unlink(missing_ok=True)
    write_outputs(collect(load_raw(raw_path)), fixes_path, report_path)


if __name__ == "__main__":
    main()
