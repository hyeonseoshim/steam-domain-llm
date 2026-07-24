"""라이브 등록용 요약 생성기 — 파인튜닝 Qwen2.5-3B + LoRA.

eval/baseline_predict.py 의 SYSTEM·프롬프트·파서를 그대로 재사용(드리프트 방지).
lazy 로드 — 첫 /ingest 호출 때만 3B 모델을 올려 검색전용 사용 시 VRAM 절약.
프로덕션 서빙은 vLLM(텍스트 계약, [[steam-part-a-serving]])이지만, 데모는 동일 LoRA
어댑터를 인프로세스로 로드해 '파인튜닝 모델이 코퍼스 문서를 생성한다'를 눈앞에서 보인다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# baseline_predict 재사용(요약 프롬프트/스키마/파서의 단일 출처). 경로는 __file__ 기준
# (backend/ → 서비스 루트 → eval), CWD 무관.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from baseline_predict import build_messages, parse_summary  # noqa: E402

# LoRA 어댑터도 __file__ 기준(backend/ → 서비스 루트 → train). CWD·통합 위치 무관.
ADAPTER = str(Path(__file__).resolve().parent.parent / "train" / "qwen2.5-3b-lora")
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class Summarizer:
    def __init__(self, adapter: str = ADAPTER, model_id: str = MODEL_ID):
        self.adapter, self.model_id = adapter, model_id
        self._tok = self._model = None
        self._brace_ids: list[int] = []
        self._compiled = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _warmup(self) -> None:
        """compile 유발 — 첫 실사용 요청이 컴파일 지연을 안 먹도록 _ensure 에서 미리 그래프 캡처."""
        import torch
        msgs = build_messages({"name": "웜업", "genres": [], "input": "웜업용 짧은 설명."}, [])
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        for _ in range(2):   # 1회차=컴파일, 2회차=graph 재사용 확인
            with torch.no_grad():
                self._model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                    eos_token_id=[self._tok.eos_token_id, *self._brace_ids],
                    pad_token_id=self._tok.eos_token_id)

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # T4(Turing)는 bf16 가속이 없음(Ampere+) → fp16 이 훨씬 빠름. Ampere+면 bf16.
        dtype = torch.bfloat16 if torch.cuda.is_available() and \
            torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        print(f"[summarizer] 파인튜닝 모델 로드 중… ({self.model_id} + {self.adapter}, {dtype})")
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        base = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="auto")
        # LoRA 병합 → 매 forward의 어댑터 행렬곱 제거 = base 속도 추론(서빙 표준, cf serve/merge_lora.py)
        self._model = PeftModel.from_pretrained(base, self.adapter).merge_and_unload().eval()
        # '}' 를 포함하는 토큰ID 집합(정지용). stop_strings 는 매 스텝 전체 시퀀스를 디코딩해
        # 문자열 매칭(O(n²)·CPU 단일스레드) → 긴 출력서 폭증. 이를 eos_token_id(토큰ID 비교, O(1))로
        # 대체해 CPU-bound 디코드 제거(Modal 14초→기대 수초). 바이트레벨 BPE라 vocab 조각에 '}' 그대로.
        self._brace_ids = [i for t, i in self._tok.get_vocab().items() if "}" in t]

        # (A) CUDA graph 실험 결론: 디코드가 CPU-launch-bound(생성 중 GPU util ~20%)인 건 확정.
        # 허나 transformers 5.12.1 static cache 가 cumulative_length.add_() 등 in-place 변형이 있어
        # reduce-overhead(cudagraph)가 "mutated inputs" 로 스킵됨(=CPU-launch 제거 효과 0) + 텐서
        # 덮어쓰기 에러. 즉 이 버전 HF 에선 graph 를 못 켬 → 기본 off. graph 로 GPU-bound 전환은
        # vLLM 네이티브(PagedAttention+graph, 별도 앱 ②)로 간다. 실험 재현용 토글만 남김(기본 0).
        if os.environ.get("SUMM_COMPILE", "0") != "0":
            orig_forward = self._model.forward
            try:
                self._model.generation_config.cache_implementation = "static"
                self._model.forward = torch.compile(
                    self._model.forward, mode="reduce-overhead", fullgraph=True)
                self._warmup()
                self._compiled = True
                print("[summarizer] torch.compile(reduce-overhead)+static cache 적용 ✓")
            except Exception as e:  # noqa: BLE001
                self._model.forward = orig_forward                       # ⚠️ 원복(생성 깨짐 방지)
                self._model.generation_config.cache_implementation = None
                self._compiled = False
                print(f"[summarizer] compile 실패→eager 원복: {e}")

    def summarize(self, name: str, genres: list[str], description: str) -> tuple[dict | None, str]:
        """원문 게임설명 → 파인튜닝 모델 3필드 요약(장르/핵심플레이/특징). (요약, 원본출력)."""
        self._ensure()
        import torch
        rec = {"name": name, "genres": genres, "input": description}
        text = self._tok.apply_chat_template(
            build_messages(rec, []), tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            # 3필드 JSON 은 flat → 첫 '}' 에서 즉시 중단(뒤 헛생성 제거로 지연↓). 320 은 안전상한.
            # 정지='}' 포함 토큰 + eos 를 eos_token_id 로(토큰ID 비교, 디코드 없음 = stop_strings 대비 빠름).
            gen = self._model.generate(
                **inputs, max_new_tokens=320, do_sample=False,
                eos_token_id=[self._tok.eos_token_id, *self._brace_ids],
                pad_token_id=self._tok.eos_token_id)
        out = self._tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_summary(out), out
