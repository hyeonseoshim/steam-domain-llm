"""중간발표 데모 — 검색 백엔드(하이브리드+리랭커) + 라이브 문서 추가.

Step3~5 산출물 재사용: corpus_emb.npy · .bm25_tokens.pkl · embed_corpus.build_docs(142K).
검색 = BM25(kiwi) + dense(bge-m3) + RRF → bge-reranker-v2-m3 재정렬(Step5 파이프라인).
      + 인기 보정: recommendations_total(corpus_pop.json)을 후보선정·최종정렬에 반영해
        넓은 장르질의에서 유명 관련게임이 무명 클론에 묻히지 않게 한다(pop_w_*=0 이면 옛 동작).
add() = 라이브 등록 게임의 요약을 bge-m3로 임베딩해 인메모리 인덱스에 추가 → 즉시 검색.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

from embed_corpus import build_docs, doc as make_doc

RP = Path(__file__).resolve().parent
EMB = RP / "corpus_emb.npy"
TOKCACHE = RP / ".bm25_tokens.pkl"
NAMES = RP / "corpus_names.jsonl"
SRC = RP / "corpus_source.jsonl"          # 원문(HTML) — appid로 온디맨드 조회
POP = RP / "corpus_pop.json"              # appid→recommendations_total(인기 신호, 덤프 추출)

_APPID = re.compile(rb'"appid":\s*(\d+)')
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_html(t: str) -> str:
    return _WS.sub(" ", _TAG.sub(" ", t or "")).strip()


class Retriever:
    def __init__(self, pool: int = 1000, rrf_k: int = 60, rerank_topk: int = 40,
                 pop_w_fuse: float = 0.6, pop_w_final: float = 0.3):
        init_started = time.perf_counter()
        # pool=1000: 후보 풀. 200이면 유명게임(예 슬더스 BM25 218위)이 풀에 못 들어옴 → 넓힘.
        # 인기 보정: 추천수(recommendations_total)를 ⓐ리랭크 후보 선정 ⓑ최종 정렬에 반영.
        #   pop_w_fuse/pop_w_final=0 이면 순수 관련도(옛 동작). 파일 없으면 자동 비활성.
        self.pop_w_fuse, self.pop_w_final = pop_w_fuse, pop_w_final
        import torch
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.pool, self.rrf_k, self.rerank_topk = pool, rrf_k, rerank_topk
        self._kiwi = Kiwi()
        self._BM25 = BM25Okapi
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        self.appids, self.docs = build_docs()          # Step3/4/5 와 동일 순서
        self.emb = np.load(EMB)
        assert self.emb.shape[0] == len(self.docs), "임베딩과 문서 수 불일치"
        self.names: dict[int, str] = {}
        for line in NAMES.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line); self.names[r["appid"]] = r.get("name", "")
        self.pop = self._load_pop()                 # appid → recommendations_total(인기)
        self.logpop = np.log1p(np.array([self.pop.get(a, 0) for a in self.appids],
                                        dtype=np.float32))
        self.summ = self._load_summaries()          # appid → {장르,핵심플레이,특징} 표시용
        self._src_off = self._index_source()        # appid → 원문파일 바이트오프셋(온디맨드)
        self._gold_raw = self._load_gold_raw()      # appid → 원문(gold input) 폴백
        self.source_ready = SRC.exists()
        print(f"[retriever] 데이터 준비 {time.perf_counter() - init_started:.1f}s")

        self.toks = (pickle.loads(TOKCACHE.read_bytes()) if TOKCACHE.exists()
                     else [self.tok(d) for d in self.docs])
        self.bm25 = self._BM25(self.toks)
        print(f"[retriever] BM25 준비 {time.perf_counter() - init_started:.1f}s")
        self.st = SentenceTransformer("BAAI/bge-m3", device=dev)
        print(f"[retriever] 임베더 준비 {time.perf_counter() - init_started:.1f}s")
        self.ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev)
        print(f"[retriever] 리랭커 준비 {time.perf_counter() - init_started:.1f}s")
        try:                                            # CUDA 커널 워밍업 → 첫 검색도 빠르게
            self.search("게임", topk=3)
        except Exception:  # noqa: BLE001
            pass
        print(f"[retriever] 코퍼스 {len(self.docs):,} · device={dev} · rerank_topk={self.rerank_topk} "
              f"준비완료 {time.perf_counter() - init_started:.1f}s")

    def _load_pop(self) -> dict[int, int]:
        """appid→추천수 맵(build_pop.py, 덤프의 recommendations_total). 없으면 인기 보정 off."""
        if not POP.exists():
            print("[retriever] corpus_pop.json 없음 — 인기 보정 비활성(순수 관련도)")
            return {}
        raw = json.loads(POP.read_text(encoding="utf-8"))
        return {int(k): int(v) for k, v in raw.items()}

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo + 1e-9)

    def _load_summaries(self) -> dict[int, dict]:
        """표시용 3필드 요약 맵(build_docs 와 동일 우선순위: gold → gemini)."""
        from embed_corpus import GEM, GOLD
        summ: dict[int, dict] = {}
        for line in Path(GOLD).open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("summary"):
                    summ[r["appid"]] = r["summary"]
        if Path(GEM).exists():
            for line in Path(GEM).open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line); summ.setdefault(r["appid"], r["summary"])
        return summ

    def _index_source(self) -> dict[int, int]:
        """corpus_source.jsonl 을 1회 스캔해 appid→바이트오프셋 맵(371MB 전량로드 회피)."""
        off_map: dict[int, int] = {}
        if not SRC.exists():
            print("[retriever] corpus_source.jsonl 없음 — 원문 재요약 비활성"); return off_map
        with SRC.open("rb") as f:
            off = f.tell(); line = f.readline()
            while line:
                m = _APPID.search(line)
                if m:
                    off_map[int(m.group(1))] = off
                off = f.tell(); line = f.readline()
        print(f"[retriever] 원문 인덱스 {len(off_map):,}건")
        return off_map

    def _load_gold_raw(self) -> dict[int, str]:
        """gold 4158 의 원문(input) — corpus_source 폴백/미탑재 대비."""
        from embed_corpus import GOLD
        m: dict[int, str] = {}
        for line in Path(GOLD).open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("input"):
                    m[r["appid"]] = r["input"]
        return m

    def raw_source(self, appid: int) -> str:
        """appid 원문(HTML 제거) 반환: corpus_source(전체) → gold input 폴백. 없으면 ''."""
        off = self._src_off.get(appid)
        if off is not None:
            with SRC.open("rb") as f:
                f.seek(off); line = f.readline()
            txt = clean_html(json.loads(line).get("source_text", ""))
            if txt:
                return txt
        return clean_html(self._gold_raw.get(appid, ""))

    def tok(self, s: str) -> list[str]:
        return [t.form for t in self._kiwi.tokenize(s)]

    def search(self, query: str, topk: int = 10) -> list[dict]:
        bm = list(np.argsort(self.bm25.get_scores(self.tok(query)))[::-1][:self.pool])
        qe = self.st.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        dn = list(np.argsort(qe @ self.emb.T)[::-1][:self.pool])
        # ⓐ 관련도(RRF) + 인기도로 리랭크 후보 선정 → 유명 관련게임이 후보에 들어오게.
        cand = list(dict.fromkeys(bm + dn)); pos = {i: k for k, i in enumerate(cand)}
        rel = np.zeros(len(cand), dtype=np.float32)
        for lst in (bm, dn):
            for r, i in enumerate(lst):
                rel[pos[i]] += 1.0 / (self.rrf_k + r)
        cand = np.asarray(cand)
        prelim = self._norm(rel) + self.pop_w_fuse * self._norm(self.logpop[cand])
        sel = cand[np.argsort(prelim)[::-1][:self.rerank_topk]]
        # ⓑ 리랭커(관련도) 재점수 후, 인기도를 살짝 더해 최종 정렬.
        scores = np.asarray(self.ce.predict([(query, self.docs[i]) for i in sel], batch_size=64))
        final = self._norm(scores) + self.pop_w_final * self._norm(self.logpop[sel])
        order = list(np.argsort(final)[::-1][:topk])
        out = []
        for rank, j in enumerate(order, 1):
            i = int(sel[j]); aid = self.appids[i]
            out.append({"rank": rank, "appid": int(aid),
                        "name": self.names.get(aid, ""),
                        "summary": self.summ.get(aid, {}),
                        "score": round(float(scores[j]), 3)})   # 배지 = 리랭커 관련도
        return out

    def add(self, name: str, summary: dict, appid: int | None = None) -> int:
        """라이브 등록: 요약 dict → 문서 임베딩 → 인메모리 인덱스 추가(즉시 검색)."""
        aid = appid if appid is not None else (max(self.appids) + 1 if self.appids else 1)
        d = make_doc(name, summary)
        e = self.st.encode([d], normalize_embeddings=True).astype(np.float32)
        self.emb = np.vstack([self.emb, e])
        self.docs.append(d)
        self.appids.append(aid)
        self.names[aid] = name
        self.summ[aid] = summary
        self.toks.append(self.tok(d))
        self.bm25 = self._BM25(self.toks)               # 재구축(수초, 데모 규모 OK)
        print(f"[retriever] 라이브 추가 appid={aid} '{name}' · 코퍼스 {len(self.docs):,}")
        return aid
