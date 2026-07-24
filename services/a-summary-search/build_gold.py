"""Part A · Step 2c — silver → gold 확정.

2b 검수 결과(규칙 QC + 독립 LLM 심사 + 사람 확정)를 all.jsonl(silver) 에 적용해
gold.jsonl 을 만든다. 사람이 내린 critical 결정을 **코드에 인라인**으로 박아
(파생 데이터가 gitignore 이므로) 결정 자체를 버전관리·재현 가능하게 한다.

결정은 이름 키워드로 정의하되 **critical 집합(review_todo.jsonl)에만 적용** —
전체 4,165건에 이름매칭하면 다른 게임과 오매칭될 수 있으므로 범위를 가둔다.

- drop : 원문이 게임 설명이 아니어서 라벨이 창작됨(복구 불가) → gold 에서 제외.
- fix  : 진짜 게임인데 원문에 없는 '얼리 액세스' 절을 환각 → 해당 필드만 교정.
- keep : 심사관 오탐(장르가 Steam 태그 기반) 또는 심사관 채점 오류 → 그대로 채택.
- 그 외(auto-gold + minor) : 심사에서 치명 결함 없음 → 그대로 채택.

gold 를 전처리 split(train/val/test) appid 로 나눠 gold_{split}.jsonl 도 생성한다.

usage:
    python3 build_gold.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 원문이 게임 설명이 아님 → 라벨 창작, 복구 불가 → 제외 (7)
DROP = {
    "I Am Sakuya": "원문=팬게임 법적 고지문뿐 → 플레이 창작",
    "Three Sons": "원문=세계관 blurb만, Features 비어있음 → 플레이 없음",
    "Midnight Riff": "원문=디스코드 가입 안내 보일러플레이트 → 플레이 없음",
    "Childhood grocery": "원문=1인개발 후원요청문 → 잡화점 시뮬 창작",
    "Super Phantom Cat": "원문=여름세일 번들 안내 페이지(게임설명 아님) → 전부 창작",
    "Super Snow Tubes": "원문=위시리스트 추가 보일러플레이트 → 플레이 없음",
    "Horizon Forbidden West": "원문=PSN 계정연동 아이템 안내 → 배경지식으로 창작",
}

# 진짜 게임 + 원문에 없는 얼리액세스 절만 환각 → 해당 필드만 교정 (6)
FIX = {
    "TrackTime": {"특징": "챔피언을 목표로 하는 육상 리그 세계를 배경으로 하며, 훈련과 개인 생활의 균형을 맞추는 선택이 평판과 진행에 영향을 준다."},
    "Wind Up": {"특징": "낮은 태엽 인형에게 유리한 시간이고 밤은 사칭자의 무대가 되는 낮-밤 순환 시스템을 갖추고 있다."},
    "TOiLET": {"장르": "캐주얼, 퍼즐",
               "특징": "화장실을 배경으로 사용자 간의 시선과 관계, 규칙을 활용해 문제를 해결하는 퍼즐 구조를 가진다."},
    "Frenzy Freak Fantasy": {"특징": "공룡, 고블린 등 기괴한 유닛과 밈 기반의 조야한 분위기를 가진 게임으로, 수 시간 분량의 싱글플레이를 제공하며 멀티플레이가 추가될 예정이다."},
    "Kirmis": {"특징": "박람회로 가는 길에 차가 멈추면서 벌어지는 이야기를 배경으로 하며, 대마초를 소재로 한 스토리와 하늘을 나는 연출을 포함한다."},
    "Frog Attack": {"특징": "모두를 먹으려는 개구리를 주인공으로 하며, 숨겨진 도전과제와 비밀 요소, 축구 경기 참여 등의 상호작용이 포함된다."},
}

# 심사관 오탐/오류지만 사람이 통과 확정 (기록용; 동작상 '그대로 채택') (10)
KEEP = {
    "Territory": "장르는 Steam 태그 기반, 플레이는 원문 근거",
    "Kakuriyo": "장르 태그 기반, 플롯 생략은 요약 특성",
    "Lab Runner": "심사관도 '핵심 위반 없음' 명시 — 형식 오채점",
    "Lab Sorters": "심사관 근거 '문제없음'인데 0점 → 채점 오류",
    "Magic or Machinations": "장르 태그 기반, 플레이 원문 근거",
    "Ed & Edda": "장르 태그 기반, 플레이 원문 근거",
    "Beyond The Brink": "장르 태그 기반, 플레이 원문 근거",
    "男女入れ替わり": "장르 '시뮬레이션'만 태그성, 플레이 근거",
    "Blazing Dragon": "원문 빈약하나 플레이·태그 근거",
    "Fell from another world": "장르 '캐주얼'만 태그성, 플레이 근거",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _match(name: str, table: dict) -> str | None:
    for key in table:
        if key in name:
            return key
    return None


def resolve_decisions(critical: list[dict]) -> dict[int, dict]:
    """critical(review_todo) 각 건을 이름으로 분류 → appid → 결정 맵. 미매칭 시 중단."""
    decisions: dict[int, dict] = {}
    unmatched: list[str] = []
    for r in critical:
        name, aid = r["name"], r["appid"]
        if (k := _match(name, DROP)):
            decisions[aid] = {"결정": "drop", "코멘트": DROP[k]}
        elif (k := _match(name, FIX)):
            decisions[aid] = {"결정": "fix", "코멘트": "원문에 없는 얼리액세스 절 제거",
                              "fields": FIX[k]}
        elif (k := _match(name, KEEP)):
            decisions[aid] = {"결정": "keep", "코멘트": KEEP[k]}
        else:
            unmatched.append(f"{aid} {name}")
    if unmatched:
        sys.exit(f"[!] 미분류 critical {len(unmatched)}건: {unmatched}")
    return decisions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="data/references/all.jsonl")
    ap.add_argument("--critical", default="data/references/review_todo.jsonl")
    ap.add_argument("--ea-fixes", default="data/references/ea_fixes.jsonl",
                    help="fix_early_access.py 산출(있으면 특징 반영)")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--outdir", default="data/references")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    records = load_jsonl(Path(args.all))
    decisions = resolve_decisions(load_jsonl(Path(args.critical)))

    # 얼리액세스 타겟 재심사 결과(있으면): appid → 수정된 특징
    ea_path = Path(args.ea_fixes)
    ea_fixes = {r["appid"]: r["특징"] for r in load_jsonl(ea_path)} if ea_path.exists() else {}

    gold: list[dict] = []
    n_drop = n_fix = n_keep = n_ea = 0
    for r in records:
        aid = r["appid"]
        d = decisions.get(aid)
        if d and d["결정"] == "drop":
            n_drop += 1
            continue
        summary = dict(r["summary"])
        if d and d["결정"] == "fix":
            summary.update(d["fields"])
            review = "fix"
            n_fix += 1
        elif d and d["결정"] == "keep":
            review = "keep_reviewed"
            n_keep += 1
        elif aid in ea_fixes:
            summary["특징"] = ea_fixes[aid]
            review = "ea_fix"
            n_ea += 1
        else:
            review = "auto"
        gold.append({
            "appid": aid,
            "name": r.get("name", ""),
            "genres": r.get("genres") or [],
            "input": r["input_clean"],   # 파인튜닝 입력
            "summary": summary,          # 파인튜닝 타깃 (gold)
            "source_model": r.get("model", ""),
            "review": review,            # auto / fix / keep_reviewed
        })

    gold_by_id = {g["appid"]: g for g in gold}
    with (outdir / "gold.jsonl").open("w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    # 전처리 split(appid)에 gold 조인 → gold_{split}.jsonl
    proc = Path(args.processed)
    split_counts = {}
    for split in ("train", "val", "test"):
        sp = proc / f"{split}.jsonl"
        if not sp.exists():
            continue
        ids = [json.loads(l)["appid"] for l in sp.open(encoding="utf-8") if l.strip()]
        kept = [gold_by_id[i] for i in ids if i in gold_by_id]
        with (outdir / f"gold_{split}.jsonl").open("w", encoding="utf-8") as f:
            for g in kept:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        split_counts[split] = (len(ids), len(kept))

    manifest = {
        "silver_total": len(records),
        "dropped": n_drop,
        "fixed": n_fix,
        "keep_reviewed": n_keep,
        "ea_fixed": n_ea,
        "gold_total": len(gold),
        "split": {s: {"orig": o, "gold": k} for s, (o, k) in split_counts.items()},
        "decisions": {str(aid): d for aid, d in decisions.items()},
    }
    (outdir / "gold_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gold] silver {len(records)} → drop {n_drop} / fix {n_fix} "
          f"/ keep-reviewed {n_keep} / ea-fix {n_ea} → gold {len(gold)}")
    for s, (o, k) in split_counts.items():
        print(f"  {s}: {o} → {k}  (-{o - k})")
    print("[out] gold.jsonl / gold_{train,val,test}.jsonl / gold_manifest.json")


if __name__ == "__main__":
    main()
