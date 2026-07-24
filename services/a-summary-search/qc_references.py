"""Part A · Step 2b — 레퍼런스 요약 품질검사(QC) & 사람 검수 샘플링.

all.jsonl(silver 요약 4,165건)에 규칙 기반 품질검사를 돌려
  - 프롬프트 제약(장르 괄호/영어병기 금지, 머리말·불릿·이모지 금지, 마케팅 과장 제거)
  - 한국어 지배율(원문 언어 그대로 복사 방지)
  - short_description 과의 n-gram 겹침 = "복붙 아님"을 정량화 (방어 논리 핵심)
을 검사하고, 하드룰 위반/소프트룰 다수 건을 flagged.jsonl 로,
층화 랜덤 표본을 review_sample.jsonl(사람 판정칸 비움)로 뽑는다.

외부 의존성 0 (표준 라이브러리만) → preprocess.py 와 동일하게 어디서든 재현.

usage:
    python3 qc_references.py
    python3 qc_references.py --sample 60 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from pathlib import Path

FIELDS = ("장르", "핵심플레이", "특징")

# 장르에 영어로 나와도 허용하는 정착된 약어/용어 (RPG 등은 한국어 관용 표기)
GENRE_LATIN_ALLOW = {
    "rpg", "mmo", "mmorpg", "fps", "tps", "moba", "trpg", "srpg", "arpg",
    "jrpg", "pvp", "pve", "pvpve", "avg", "slg", "rts", "tcg", "css", "vr",
    "ar", "sf", "fmv", "midi", "roguelike", "roguelite", "2d", "3d",
}

# 마케팅 과장/홍보 표현 (프롬프트에서 제거하라고 지시한 톤) — 잔존 시 소프트 플래그
HYPE_LEXICON = [
    "최고의", "최고급", "압도적", "숨막히", "숨 막히", "환상적", "황홀",
    "당신을 사로잡", "잊을 수 없는", "짜릿한", "전율", "경이로운",
    "궁극의", "역대급", "완벽한", "놀라운 재미", "손에 땀",
]

# 금지 머리말 (프롬프트: 머리말 없이 내용만)
LEADINS = ["이 게임은", "본 게임은", "이 작품은", "본 작품은", "해당 게임은"]

BULLET_RE = re.compile(r"(^|\n)\s*[-*•·▪◦]\s")
PAREN_RE = re.compile(r"[()（）]")
# 숫자 붙은 형태(2D/3D)를 통째로 잡아 allowlist 와 대조 (D 단독 오탐 방지)
LATIN_RE = re.compile(r"[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*")
SENT_SPLIT_RE = re.compile(r"[.!?。！？]\s|\n")


def is_emoji(ch: str) -> bool:
    """포맷팅용 이모지(픽토그래프)만 탐지. ♡ 등 타이틀 장식 기호(<0x1F000)는 제외 —
    프롬프트가 고유명사 원문 표기를 허용하므로 게임명 속 기호는 위반이 아니다."""
    return ord(ch) >= 0x1F000


def hangul_ratio(s: str) -> float:
    """한글 음절 / (한글 + CJK한자 + 라틴 문자) 비율. 문장부호·공백·숫자 제외.

    낮으면 원문(영어/중국어 등)을 요약 없이 그대로 복사했을 가능성 → 하드 플래그.
    """
    hangul = cjk = latin = 0
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF:
            hangul += 1
        elif 0x4E00 <= o <= 0x9FFF:  # CJK 한자
            cjk += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    denom = hangul + cjk + latin
    return hangul / denom if denom else 1.0


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def overlap_jaccard(a: str, b: str, n: int = 3) -> float:
    """요약과 short_description 의 char n-gram Jaccard. 높으면 복붙 의심."""
    A, B = char_ngrams(a, n), char_ngrams(b, n)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def sentence_count(s: str) -> int:
    parts = [p for p in SENT_SPLIT_RE.split(s.strip()) if p.strip()]
    return max(1, len(parts))


def check_record(rec: dict) -> dict:
    """한 레코드에 대한 하드/소프트 플래그 목록 반환."""
    s = rec.get("summary")
    hard: list[str] = []
    soft: list[str] = []

    # --- H1 스키마: 3필드 존재 & 공백 아닌 문자열 ---
    if not isinstance(s, dict):
        return {"hard": ["H1_schema_missing"], "soft": [], "overlap": 0.0}
    for k in FIELDS:
        v = s.get(k)
        if not isinstance(v, str) or not v.strip():
            hard.append(f"H1_field_empty:{k}")

    genre = (s.get("장르") or "")
    play = (s.get("핵심플레이") or "")
    feat = (s.get("특징") or "")
    body = f"{play}\n{feat}"
    alltext = f"{genre}\n{play}\n{feat}"

    # --- H2 장르 괄호(원어 병기 형태) 금지 ---
    if PAREN_RE.search(genre):
        hard.append("H2_genre_paren")

    # --- H3 불릿/이모지 금지 ---
    if BULLET_RE.search(alltext):
        hard.append("H3_bullet")
    if any(is_emoji(ch) for ch in alltext):
        hard.append("H3_emoji")

    # --- H4 금지 머리말 ---
    for field_name, val in (("핵심플레이", play), ("특징", feat)):
        if any(val.lstrip().startswith(p) for p in LEADINS):
            hard.append(f"H4_leadin:{field_name}")

    # --- H5 한국어 지배율 (원문 언어 그대로 복사 방지) ---
    # 임계 0.40: 고유명사(Marvel/VR 등) 다수 보존 시 0.5대로 내려가나 이는 정상.
    # 원문 언어를 통째로 복사한 미번역 케이스(≈0.0)만 잡는 것이 목적.
    hr = hangul_ratio(alltext)
    if hr < 0.40:
        hard.append(f"H5_low_hangul:{hr:.2f}")

    # --- S1 장르 영어 병기(허용 약어 제외) ---
    latins = [m.group(0) for m in LATIN_RE.finditer(genre)]
    bad_latin = [w for w in latins if w.lower() not in GENRE_LATIN_ALLOW]
    if bad_latin:
        soft.append("S1_genre_latin:" + ",".join(bad_latin[:3]))

    # --- S2 마케팅 과장어 잔존 ---
    hits = [w for w in HYPE_LEXICON if w in alltext]
    if hits:
        soft.append("S2_hype:" + ",".join(hits[:3]))

    # --- S3 과장 길이 / S4 과소 길이 ---
    for field_name, val in (("핵심플레이", play), ("특징", feat)):
        if sentence_count(val) > 3 or len(val) > 220:
            soft.append(f"S3_long:{field_name}")
        if len(val.strip()) < 10:
            soft.append(f"S4_short:{field_name}")
    if len(genre.strip()) < 2:
        soft.append("S4_short:장르")

    # --- S5 short_description 과의 겹침(복붙 의심) ---
    sd = rec.get("short_description") or ""
    ov = overlap_jaccard(body, sd) if sd else 0.0
    if ov >= 0.5:
        soft.append(f"S5_copy:{ov:.2f}")

    return {"hard": hard, "soft": soft, "overlap": ov}


def bucket(len_clean: int | None) -> str:
    n = len_clean or 0
    if n < 500:
        return "short"
    if n < 1500:
        return "mid"
    return "long"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/references/all.jsonl")
    ap.add_argument("--outdir", default="data/references")
    ap.add_argument("--sample", type=int, default=60, help="사람 검수 표본 크기")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    hard_counts: dict[str, int] = {}
    soft_counts: dict[str, int] = {}
    overlaps: list[float] = []
    flagged: list[dict] = []
    per_rec: list[tuple[dict, dict]] = []

    for rec in records:
        res = check_record(rec)
        per_rec.append((rec, res))
        overlaps.append(res["overlap"])
        for tag in res["hard"]:
            hard_counts[tag.split(":")[0]] = hard_counts.get(tag.split(":")[0], 0) + 1
        for tag in res["soft"]:
            soft_counts[tag.split(":")[0]] = soft_counts.get(tag.split(":")[0], 0) + 1
        # flagged: 하드룰 위반 OR 소프트룰 1개 이상 (기계가 잡은 검토 후보 전량)
        if res["hard"] or res["soft"]:
            flagged.append({
                "appid": rec["appid"],
                "name": rec.get("name", ""),
                "hard": res["hard"],
                "soft": res["soft"],
                "overlap": round(res["overlap"], 3),
                "summary": rec.get("summary"),
            })

    n = len(records)
    n_hard = sum(1 for _, r in per_rec if r["hard"])
    n_soft_only = sum(1 for _, r in per_rec if not r["hard"] and r["soft"])
    n_clean = n - sum(1 for _, r in per_rec if r["hard"] or r["soft"])

    def dist(xs: list[float]) -> dict:
        xs = sorted(xs)
        return {
            "min": round(xs[0], 3), "median": round(st.median(xs), 3),
            "mean": round(st.mean(xs), 3),
            "p90": round(xs[int(len(xs) * 0.9)], 3),
            "p99": round(xs[min(int(len(xs) * 0.99), len(xs) - 1)], 3),
            "max": round(xs[-1], 3),
        }

    report = {
        "input_file": in_path.name,
        "total": n,
        "hard_fail_records": n_hard,
        "soft_only_records": n_soft_only,
        "clean_records": n_clean,
        "hard_rule_counts": dict(sorted(hard_counts.items())),
        "soft_rule_counts": dict(sorted(soft_counts.items())),
        "overlap_short_desc": {
            "note": "요약↔short_description char-3gram Jaccard. 낮을수록 복붙 아님(방어 지표).",
            **dist(overlaps),
            "ge_0.5_count": sum(1 for o in overlaps if o >= 0.5),
        },
        "flagged_count": len(flagged),
        "sample_size": args.sample,
        "seed": args.seed,
    }

    (outdir / "qc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (outdir / "flagged.jsonl").open("w", encoding="utf-8") as f:
        for r in flagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 층화 랜덤 사람 검수 표본 (길이 버킷별 비례) ---
    rng = random.Random(args.seed)
    by_bucket: dict[str, list[dict]] = {"short": [], "mid": [], "long": []}
    for rec in records:
        by_bucket[bucket(rec.get("len_clean"))].append(rec)
    sample: list[dict] = []
    for b, recs_b in by_bucket.items():
        k = max(1, round(args.sample * len(recs_b) / n))
        sample.extend(rng.sample(recs_b, min(k, len(recs_b))))
    rng.shuffle(sample)
    sample = sample[:args.sample]

    with (outdir / "review_sample.jsonl").open("w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps({
                "appid": rec["appid"],
                "name": rec.get("name", ""),
                "len_clean": rec.get("len_clean"),
                "input_clean": rec.get("input_clean", "")[:1200],
                "short_description": rec.get("short_description", ""),
                "summary": rec.get("summary"),
                # 사람이 채우는 판정칸 (1=적절, 0=부적절, ""=미검수)
                "review": {"장르_ok": "", "핵심플레이_ok": "", "특징_ok": "",
                           "환각_없음": "", "코멘트": ""},
            }, ensure_ascii=False) + "\n")

    # --- 콘솔 요약 ---
    print(f"[qc] 전체 {n}건")
    print(f"  하드룰 위반    : {n_hard}건 ({n_hard/n*100:.1f}%)")
    print(f"  소프트룰만     : {n_soft_only}건 ({n_soft_only/n*100:.1f}%)")
    print(f"  무결(clean)    : {n_clean}건 ({n_clean/n*100:.1f}%)")
    print(f"  하드룰 세부    : {report['hard_rule_counts'] or '없음'}")
    print(f"  소프트룰 세부  : {report['soft_rule_counts'] or '없음'}")
    od = report["overlap_short_desc"]
    print(f"  복붙지표(겹침) : median={od['median']} p90={od['p90']} "
          f"max={od['max']}  (≥0.5 {od['ge_0.5_count']}건)")
    print(f"[out] qc_report.json / flagged.jsonl({len(flagged)}) / "
          f"review_sample.jsonl({len(sample)})")


if __name__ == "__main__":
    main()
