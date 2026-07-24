"""Part A · Step 2b — 독립 LLM 심사(judge) → 사람 확정 대상 추림.

silver 요약(all.jsonl)을 **생성기와 다른 모델(Sonnet)** 로, **생성 프롬프트를 모른 채**
원문↔요약만 대조해 4개 축(환각/누락/톤/형식)을 0~2점 채점한다.
→ 규칙 QC(형식) 다음의 '내용·충실성' 게이트. 자기채점 순환을 끊기 위해:
  1) 생성=Opus, 심사=Sonnet 로 모델을 분리하고
  2) 심사는 저점/불일치만 `review_todo.jsonl` 로 뽑아 **사람이 최종 확정**한다.

크래시 내성은 generate_references_batch.py 와 동일 (batch_id 를 상태파일에 저장 → 재실행 재개).

usage:
    # 제출 + 폴링 + 결과 수거 → reviews.jsonl, review_todo.jsonl
    uv run python review_references.py

    uv run python review_references.py --status       # 배치 상태만
    python3 review_references.py --report-only        # 이미 받은 결과로 집계만
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

# 심사관 = 생성기(Opus)와 다른 모델 → 독립성 확보(방어). 짧은 채점이라 비용도 소액.
JUDGE_MODEL = "claude-sonnet-4-6"

# 0~2 점수 축. 생성 프롬프트를 재사용하지 않고 QA 관점의 독립 루브릭으로 재정의.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "환각": {"type": "integer", "enum": [0, 1, 2],
                 "description": "요약이 원문에 근거하는가. 0=원문에 없는 내용 창작, 1=경미한 추측/과장, 2=전부 원문 근거"},
        "누락": {"type": "integer", "enum": [0, 1, 2],
                 "description": "핵심 장르·플레이를 담았는가. 0=핵심 누락, 1=일부 누락, 2=충실"},
        "톤": {"type": "integer", "enum": [0, 1, 2],
               "description": "중립·정보 위주인가. 0=마케팅 과장 다수, 1=약간, 2=중립"},
        "형식": {"type": "integer", "enum": [0, 1, 2],
                 "description": "장르가 명사구(괄호·영어병기·문장형 아님)인가. 쉼표로 여러 장르를 나열하는 것은 정상이며 감점 아님. 0=괄호/영어병기/문장형 위반, 1=경미, 2=준수"},
        "판정": {"type": "string", "enum": ["pass", "review"],
                 "description": "실질적 문제(환각·핵심누락·마케팅톤·형식위반)가 있으면 review, 명확히 좋으면 pass"},
        "근거": {"type": "string", "description": "판정 이유 1~2문장(한국어). 문제 있으면 구체적으로."},
    },
    "required": ["환각", "누락", "톤", "형식", "판정", "근거"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """\
너는 게임 설명 요약의 품질을 검수하는 독립 평가자다.
[원문 게임 설명]과 그로부터 만들어진 [요약]만 보고, 요약의 품질을 4개 축으로 0~2점 채점한다.

- 환각: 요약의 모든 내용이 원문에 실제로 있는가? 원문에 없는 사실·수치·설정을 지어냈다면 감점. (원문이 영어/중국어여도 의미가 일치하면 OK. 고유명사 원문 표기는 허용.)
- 누락: 게임의 핵심 장르와 실제 플레이가 요약에 담겼는가?
- 톤: 마케팅 과장·홍보 문구 없이 사실 위주·중립적인가? (단, 게임 세계관/설정 안에 등장하는 표현은 과장이 아니다.)
- 형식: '장르'가 괄호·영어병기·문장형 없이 명사구인가? **쉼표로 여러 장르를 나열하는 것은 정상이며 감점하지 않는다**(예: "액션, 어드벤처, 인디"는 형식 위반이 아님). 각 필드가 간결한가?

엄격하되 실제 문제에만 감점하라. 환각·핵심누락·마케팅톤·형식위반 중 실질적 문제가 있으면 '판정'을 review, 명확히 좋으면 pass 로 둔다.
오직 제공된 원문에 근거해 판단하고, 너의 배경지식으로 원문에 없는 내용을 채우지 마라.
"""


def build_judge_prompt(rec: dict) -> str:
    s = rec.get("summary") or {}
    return (
        f"[원문 게임 설명]\n{rec.get('input_clean', '')}\n\n"
        f"[요약]\n"
        f"장르: {s.get('장르', '')}\n"
        f"핵심플레이: {s.get('핵심플레이', '')}\n"
        f"특징: {s.get('특징', '')}"
    )


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def load_done_appids(path: Path) -> set[int]:
    return {r["appid"] for r in load_jsonl(path)}


def build_requests(recs: list[dict]) -> list[Request]:
    reqs: list[Request] = []
    for rec in recs:
        reqs.append(Request(
            custom_id=str(rec["appid"]),
            params=MessageCreateParamsNonStreaming(
                model=JUDGE_MODEL,
                max_tokens=512,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": build_judge_prompt(rec)}],
                output_config={"format": {"type": "json_schema", "schema": REVIEW_SCHEMA}},
            ),
        ))
    return reqs


AXES = ("환각", "누락", "톤", "형식")


def _emit(f, r: dict) -> None:
    f.write(json.dumps({
        "appid": r["appid"], "name": r.get("name", ""),
        "scores": r["scores"],
        "summary": r.get("summary"),
        # 사람이 채우는 최종 확정칸: keep / fix / drop
        "verdict": {"결정": "", "코멘트": ""},
    }, ensure_ascii=False) + "\n")


def write_report(reviews: list[dict], todo_path: Path, report_path: Path,
                 audit_path: Path, seed: int = 42, audit_n: int = 60) -> None:
    """reviews.jsonl 집계 → 3티어로 분류.

    critical(어느 축이든 0점) → review_todo.jsonl: 사람이 전수 검수.
    minor(판정 review거나 어느 축 1점, 단 0점 없음) → 랜덤 audit_n건만 review_audit.jsonl.
    나머지(auto-gold) → 검수 없이 gold 채택.
    """
    n = len(reviews)
    dist = {a: {0: 0, 1: 0, 2: 0} for a in AXES}
    critical: list[dict] = []
    minor: list[dict] = []
    auto_gold = 0
    for r in reviews:
        sc = r["scores"]
        for a in AXES:
            dist[a][sc[a]] += 1
        if any(sc[a] == 0 for a in AXES):
            critical.append(r)
        elif sc["판정"] == "review" or any(sc[a] == 1 for a in AXES):
            minor.append(r)
        else:
            auto_gold += 1

    # critical: 최저점 순 전수
    with todo_path.open("w", encoding="utf-8") as f:
        for r in sorted(critical, key=lambda x: min(x["scores"][a] for a in AXES)):
            _emit(f, r)

    # minor: 심사관 경미 플래그가 타당한지 검증할 랜덤 감사 표본
    rng = random.Random(seed)
    audit = rng.sample(minor, min(audit_n, len(minor)))
    with audit_path.open("w", encoding="utf-8") as f:
        for r in audit:
            _emit(f, r)

    report = {
        "judge_model": JUDGE_MODEL,
        "reviewed": n,
        "tiers": {
            "critical_review_all": len(critical),
            "minor_flagged": len(minor),
            "minor_audit_sample": len(audit),
            "auto_gold": auto_gold,
        },
        "human_workload": len(critical) + len(audit),
        "score_dist": dist,
        "mean_scores": {a: round(sum(r["scores"][a] for r in reviews) / n, 3)
                        if n else 0.0 for a in AXES},
        "seed": seed,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print(f"[report] 심사 {n}건")
    print(f"  critical(0점·전수검수) : {len(critical)}")
    print(f"  minor(경미 플래그)     : {len(minor)}  → 감사표본 {len(audit)}")
    print(f"  auto-gold(검수 불필요) : {auto_gold} ({auto_gold/n*100:.1f}%)")
    print(f"  → 사람 실작업량        : {len(critical)+len(audit)}건")
    for a in AXES:
        d = dist[a]
        print(f"  {a}: 0={d[0]} 1={d[1]} 2={d[2]}  (mean {report['mean_scores'][a]})")
    print(f"[out] {todo_path.name}({len(critical)}) / "
          f"{audit_path.name}({len(audit)}) / {report_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/references/all.jsonl")
    ap.add_argument("--reviews", default="data/references/reviews.jsonl")
    ap.add_argument("--todo", default="data/references/review_todo.jsonl")
    ap.add_argument("--audit", default="data/references/review_audit.jsonl")
    ap.add_argument("--report", default="data/references/review_report.json")
    ap.add_argument("--state", default="data/references/review_batch_state.json")
    ap.add_argument("--seed", type=int, default=42, help="감사표본 랜덤 시드")
    ap.add_argument("--audit-n", type=int, default=60, help="minor 감사표본 크기")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="API 호출 없이 기존 reviews.jsonl 로 집계만")
    args = ap.parse_args()

    reviews_path = Path(args.reviews)
    todo_path = Path(args.todo)
    audit_path = Path(args.audit)
    report_path = Path(args.report)
    state_path = Path(args.state)

    src = {r["appid"]: r for r in load_jsonl(Path(args.input))}

    if args.report_only:
        reviews = load_jsonl(reviews_path)
        if not reviews:
            print("reviews.jsonl 없음. 먼저 심사를 실행하세요.", file=sys.stderr)
            sys.exit(1)
        write_report(reviews, todo_path, report_path, audit_path, args.seed, args.audit_n)
        return

    load_dotenv()
    client = anthropic.Anthropic()

    batch_id: str | None = None
    if state_path.exists():
        batch_id = json.loads(state_path.read_text())["batch_id"]
        print(f"[resume] 기존 심사 배치 이어받기: {batch_id}")

    if args.status:
        if not batch_id:
            print("진행 중인 심사 배치 없음.")
            return
        b = client.messages.batches.retrieve(batch_id)
        print(f"status={b.processing_status}  counts={b.request_counts}")
        return

    # 신규 제출: 아직 심사 안 된 appid만
    if batch_id is None:
        done = load_done_appids(reviews_path)
        todo_recs = [r for aid, r in src.items() if aid not in done]
        if not todo_recs:
            print(f"[done] 이미 전부 심사됨 ({len(done)}건). 집계만 갱신.")
            write_report(load_jsonl(reviews_path), todo_path, report_path, audit_path, args.seed, args.audit_n)
            return
        print(f"[submit] 심사 대상 {len(todo_recs)}건 (완료 {len(done)} skip), model={JUDGE_MODEL}")
        batch = client.messages.batches.create(requests=build_requests(todo_recs))
        batch_id = batch.id
        state_path.write_text(json.dumps({"batch_id": batch_id}, indent=2))
        print(f"[submit] batch_id={batch_id} → {state_path}")

    # 완료까지 폴링 (크래시 나도 재실행하면 복귀)
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"  status={b.processing_status}  처리중={c.processing} "
              f"성공={c.succeeded} 실패={c.errored}")
        if b.processing_status == "ended":
            break
        time.sleep(30)

    # 결과 수거 → reviews.jsonl append
    n_ok = n_err = 0
    with reviews_path.open("a", encoding="utf-8") as f:
        for res in client.messages.batches.results(batch_id):
            appid = int(res.custom_id)
            if res.result.type != "succeeded":
                n_err += 1
                print(f"  [err] appid={appid}: {res.result.type}", file=sys.stderr)
                continue
            msg = res.result.message
            text = next(bl.text for bl in msg.content if bl.type == "text")
            rec = src.get(appid, {})
            f.write(json.dumps({
                "appid": appid,
                "name": rec.get("name", ""),
                "summary": rec.get("summary"),
                "scores": json.loads(text),
                "judge_model": JUDGE_MODEL,
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[fetch] 성공 {n_ok}건, 실패 {n_err}건 → {reviews_path}")
    state_path.unlink(missing_ok=True)
    if n_err:
        print(f"[!] 실패 {n_err}건은 재실행하면 새 배치로 재시도됩니다.")

    write_report(load_jsonl(reviews_path), todo_path, report_path, audit_path, args.seed, args.audit_n)


if __name__ == "__main__":
    main()
