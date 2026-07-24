"""RAG 파이프라인 Step3 — 한국어 요약 코퍼스를 bge-m3로 임베딩 (Lightning GPU).

요약 소스 병합: gold 4158(Opus) + corpus_summaries.jsonl(Gemini). 이름은
corpus_names.jsonl(slim appid→name, ~7MB)에서 — 없으면 원문 corpus_source.jsonl 폴백.
문서 포맷은 hybrid_search와 동일:
  doc = f"{name} 장르 {장르} {핵심플레이} {특징}"
bge-m3로 인코딩(정규화) → corpus_emb.npy(N×1024 float32) + corpus_appid_map.csv.
GPU-시간·처리량 로깅(멘토 '클라우드 GPU 계측' 증거).

usage (로컬):
    uv run --with sentence-transformers --with numpy backend/embed_corpus.py
usage (Lightning, 평평한 경로로 업로드 시):
    python embed_corpus.py --gold gold.jsonl --names corpus_names.jsonl \
        --gem corpus_summaries.jsonl --out corpus_emb.npy --map corpus_appid_map.csv \
        --batch 512 2>&1 | tee .embed.log
    # 업로드 번들(~66MB): corpus_names.jsonl + corpus_summaries.jsonl + gold.jsonl
    # GPU면 자동 cuda + 큰 배치. CPU 폴백 시 --batch 32.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

RP = Path(__file__).resolve().parent
GOLD = Path(__file__).resolve().parent.parent / "data/references/gold.jsonl"
NAMES = RP / "corpus_names.jsonl"    # slim appid→name (Lightning 업로드용 ~7MB)
SOURCE = RP / "corpus_source.jsonl"  # 원문 355MB — names 폴백(로컬 전용)
GEM = RP / "corpus_summaries.jsonl"
EMB = RP / "corpus_emb.npy"
MAP = RP / "corpus_appid_map.csv"


def doc(name: str, s: dict) -> str:
    return f"{name} 장르 {s.get('장르','')} {s.get('핵심플레이','')} {s.get('특징','')}".strip()


def build_docs(gold: Path = GOLD, names_src: Path | None = None,
               gem: Path = GEM) -> tuple[list[int], list[str]]:
    # name: slim corpus_names.jsonl 우선, 없으면 원문 corpus_source.jsonl 폴백
    npath = Path(names_src) if names_src else (NAMES if NAMES.exists() else SOURCE)
    names: dict[int, str] = {}
    for line in npath.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line); names[r["appid"]] = r.get("name", "")
    summ: dict[int, dict] = {}
    # gold(Opus) 우선
    for line in Path(gold).open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("summary"):
                summ[r["appid"]] = r["summary"]; names.setdefault(r["appid"], r.get("name", ""))
    # Gemini(없는 것만 채움)
    if Path(gem).exists():
        for line in Path(gem).open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line); summ.setdefault(r["appid"], r["summary"])
    appids, docs = [], []
    for aid, s in summ.items():
        appids.append(aid); docs.append(doc(names.get(aid, ""), s))
    return appids, docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--names", type=Path, default=None, help="slim appid→name (기본: corpus_names→source 폴백)")
    ap.add_argument("--gem", type=Path, default=GEM)
    ap.add_argument("--out", type=Path, default=EMB)
    ap.add_argument("--map", dest="map_out", type=Path, default=MAP)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    appids, docs = build_docs(gold=args.gold, names_src=args.names, gem=args.gem)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"[embed] 문서 {len(docs):,} · device={dev}({gpu}) · batch={args.batch}")

    st = SentenceTransformer(args.model, device=dev)
    t0 = time.time()
    emb = st.encode(docs, normalize_embeddings=True, batch_size=args.batch,
                    show_progress_bar=True).astype(np.float32)
    dt = time.time() - t0

    np.save(args.out, emb)
    with args.map_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "appid"])
        for i, a in enumerate(appids):
            w.writerow([i, a])

    thru = len(docs) / dt
    print(f"[embed] 완료 {emb.shape} → {args.out.name} · {args.map_out.name}")
    print(f"[GPU계측] device={gpu} · {dt:.0f}s · {thru:.0f} doc/s · "
          f"(150K 환산 {150000/thru/60:.1f}분)")


if __name__ == "__main__":
    main()
