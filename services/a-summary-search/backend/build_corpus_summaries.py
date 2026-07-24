"""RAG 파이프라인 Step2 — Gemini 배치로 150K 검색문서(한국어 요약) 생성 (청크).

Tier1 배치 enqueue 쿼터상 138K를 한 잡으로 못 넣어, 청크로 쪼개 제출한다.
build(청크 파일 생성) → submit(쿼터 되는 만큼 제출, 나머지는 다음에) → collect(완료분
파싱·append). resume: 완료 appid는 재요청 안 함, 이미 제출/수집한 청크는 건너뜀.

usage:
    uv run --with google-genai backend/build_corpus_summaries.py build --chunk 25000
    uv run --with google-genai --with python-dotenv backend/build_corpus_summaries.py submit
    uv run --with google-genai --with python-dotenv backend/build_corpus_summaries.py collect
    # submit↔collect 반복(쿼터 배출되며 남은 청크 계속 제출)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pilot_cheap_summary import SYSTEM

RP = Path(__file__).resolve().parent
SOURCE = RP / "corpus_source.jsonl"
GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
REQ_DIR = RP / "batch_chunks"
STATE = RP / "corpus_batch_state.json"
OUT = RP / "corpus_summaries.jsonl"
MODEL = "gemini-flash-lite-latest"
CAP = 2000


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"jobs": []}  # [{chunk, file, job, collected}]


def _save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2))


def gold_appids() -> set[int]:
    return {json.loads(l)["appid"] for l in GOLD.open(encoding="utf-8") if l.strip()}


def done_appids() -> set[int]:
    if not OUT.exists():
        return set()
    return {json.loads(l)["appid"] for l in OUT.open(encoding="utf-8") if l.strip()}


def _req(r: dict) -> dict:
    user = f"게임명: {r['name']}\n\n[게임 설명]\n{r['source_text'][:CAP]}"
    return {"key": str(r["appid"]),
            "request": {"system_instruction": {"parts": [{"text": SYSTEM}]},
                        "contents": [{"parts": [{"text": user}], "role": "user"}],
                        "generation_config": {"response_mime_type": "application/json",
                                              "max_output_tokens": 400, "temperature": 0.3}}}


def build(chunk: int) -> None:
    skip = gold_appids() | done_appids()
    REQ_DIR.mkdir(exist_ok=True)
    for old in REQ_DIR.glob("req_*.jsonl"):
        old.unlink()
    buf, ci, total = [], 0, 0

    def flush() -> None:
        nonlocal buf, ci
        if not buf:
            return
        (REQ_DIR / f"req_{ci:03d}.jsonl").write_text("".join(buf), encoding="utf-8")
        ci += 1; buf = []

    for line in SOURCE.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["appid"] in skip:
            continue
        buf.append(json.dumps(_req(r), ensure_ascii=False) + "\n"); total += 1
        if len(buf) >= chunk:
            flush()
    flush()
    print(f"[build] 요청 {total}개 → 청크 {ci}개(각 ≤{chunk}) in {REQ_DIR} (gold+완료 {len(skip)} 제외)")


def submit() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415
    from google import genai  # noqa: PLC0415
    from google.genai import errors  # noqa: PLC0415
    load_dotenv("/mnt/c/Users/jungs/Downloads/steam-domain-specific-llm/.env")
    c = genai.Client()
    s = _load_state()
    submitted = {j["chunk"] for j in s["jobs"]}
    chunks = sorted(REQ_DIR.glob("req_*.jsonl"))
    n_new = 0
    for f in chunks:
        ci = int(f.stem.split("_")[1])
        if ci in submitted:
            continue
        try:
            up = c.files.upload(file=str(f), config={"mime_type": "application/jsonl",
                                                     "display_name": f.stem})
            job = c.batches.create(model=MODEL, src=up.name, config={"display_name": f.stem})
            s["jobs"].append({"chunk": ci, "file": up.name, "job": job.name, "collected": False})
            _save_state(s); n_new += 1
            print(f"  chunk {ci} 제출 {job.name} state={job.state}")
        except errors.ClientError as e:
            if e.code == 429:
                print(f"  chunk {ci} 429(쿼터 소진) — 여기서 중단. collect 후 다시 submit.")
                break
            raise
    print(f"[submit] 이번에 {n_new}개 제출 (총 {len(s['jobs'])}/{len(chunks)}청크)")


def collect() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415
    from google import genai  # noqa: PLC0415
    load_dotenv("/mnt/c/Users/jungs/Downloads/steam-domain-specific-llm/.env")
    c = genai.Client()
    s = _load_state()
    tin = tout = n_ok = n_done_jobs = 0
    with OUT.open("a", encoding="utf-8") as w:
        for j in s["jobs"]:
            if j["collected"]:
                continue
            job = c.batches.get(name=j["job"])
            if not str(job.state).endswith("SUCCEEDED"):
                print(f"  chunk {j['chunk']}: {job.state} (대기)")
                continue
            txt = c.files.download(file=job.dest.file_name).decode("utf-8")
            for line in txt.splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                resp = o.get("response") or {}
                try:
                    cand = resp["candidates"][0]["content"]["parts"][0]["text"]
                    summ = {k: str(json.loads(cand)[k]).strip() for k in ("장르", "핵심플레이", "특징")}
                except (KeyError, IndexError, json.JSONDecodeError, TypeError):
                    continue
                um = resp.get("usageMetadata") or {}
                tin += um.get("promptTokenCount", 0); tout += um.get("candidatesTokenCount", 0)
                w.write(json.dumps({"appid": int(o["key"]), "summary": summ}, ensure_ascii=False) + "\n")
                n_ok += 1
            j["collected"] = True; n_done_jobs += 1
            print(f"  chunk {j['chunk']}: 수집 완료")
    _save_state(s)
    cost = tin / 1e6 * 0.05 + tout / 1e6 * 0.20
    remaining = sum(1 for j in s["jobs"] if not j["collected"])
    print(f"[collect] 잡 {n_done_jobs}개 수집 · 요약 +{n_ok} · 토큰 in={tin:,} out={tout:,} · ~${cost:.2f}")
    print(f"  누적 요약: {len(done_appids()):,}개 · 미수집 잡 {remaining}개")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "submit", "collect"])
    ap.add_argument("--chunk", type=int, default=25000)
    args = ap.parse_args()
    {"build": lambda: build(args.chunk), "submit": submit, "collect": collect}[args.action]()


if __name__ == "__main__":
    main()
