"""vLLM 기반 A 요약 생성기 — 파인튜닝 LoRA를 vLLM으로 서빙(HF generate 대체).

HF `generate`가 환경 따라 느리고(Modal L4 9초) 최적화 취약해서, PagedAttention·최적 커널로
일관되게 빠르게(~수 초) 만든다. LoRA는 vLLM 네이티브(enable_lora)로 base+adapter 적용(병합 불필요).
검색 모델(bge-m3·리랭커)과 한 GPU 공존 위해 gpu_memory_utilization 을 낮춰 잡는다.

인터페이스는 demo_summarizer.Summarizer 와 동일(summarize, loaded) → gpu_backend 무수정 교체.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# baseline_predict 재사용(프롬프트/파서 단일 출처). eval 은 서비스 루트 하위(backend 의 형제).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from baseline_predict import build_messages, parse_summary  # noqa: E402

ADAPTER = str(Path(__file__).resolve().parent.parent / "train" / "qwen2.5-3b-lora")
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class VllmSummarizer:
    def __init__(self, adapter: str = ADAPTER, model_id: str = MODEL_ID):
        import threading
        self.adapter, self.model_id = adapter, model_id
        self._llm = None
        self._tok = None
        self._lora = None
        self._load_lock = threading.Lock()   # 부팅 백그라운드 예열과 첫 요청의 이중 로드 방지

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def _ensure(self) -> None:
        if self._llm is not None:
            return
        with self._load_lock:                # 동시 _ensure(백그라운드 예열 + 요청) 직렬화
            if self._llm is not None:
                return
            self._load()

    def _load(self) -> None:
        load_started = time.perf_counter()
        from transformers import AutoTokenizer
        from vllm import LLM
        util = float(os.environ.get("VLLM_GPU_UTIL", "0.5"))     # 검색 모델과 공존 위해 GPU 비율 제한
        # MERGED_MODEL_DIR 있으면 병합모델 서빙 → 토큰당 LoRA 적용(오버헤드) 제거 = base 속도(더 빠름).
        # 없으면 base + 네이티브 LoRA(enable_lora) 폴백.
        merged = os.environ.get("MERGED_MODEL_DIR", "")
        common = dict(gpu_memory_utilization=util, max_model_len=8192,
                      dtype="bfloat16", enforce_eager=False)     # CUDA graph 켬(GPU-bound 디코드)
        # 온라인 FP8 — 디스크 bf16 을 로드 시 메모리에서 FP8 로 양자화(저장물 없음, 순수 서빙단계).
        # 토큰당 가중치 대역폭 절반 → 대역폭-bound 디코드 ~2배. L4(Ada) FP8 지원. env 토글로 껐다켰다.
        quant = os.environ.get("VLLM_QUANT", "").strip()
        if quant:
            common["quantization"] = quant
            print(f"[vllm] 온라인 양자화 quantization={quant}")
        # 같은 target 모델의 출력을 작은 draft가 미리 제안하고 target이 검증한다.
        # env를 비우면 운영 롤백 시 코드 변경 없이 speculative decoding만 끌 수 있다.
        spec_model = os.environ.get("VLLM_SPEC_MODEL", "").strip()
        if spec_model:
            spec_tokens = int(os.environ.get("VLLM_SPEC_TOKENS", "1"))
            if spec_tokens < 1:
                raise ValueError("VLLM_SPEC_TOKENS must be >= 1")
            common["speculative_config"] = {
                "model": spec_model,
                "method": "draft_model",
                "num_speculative_tokens": spec_tokens,
                "draft_tensor_parallel_size": 1,
                "max_model_len": 8192,
            }
            print(f"[vllm] speculative decoding model={spec_model} K={spec_tokens}")
        if merged and Path(merged).exists():
            print(f"[vllm] 병합모델 로드 (LoRA 병합됨, gpu_util={util}) — {merged}")
            self._tok = AutoTokenizer.from_pretrained(merged)
            self._llm = LLM(model=merged, **common)
            self._lora = None
        else:
            from vllm.lora.request import LoRARequest
            print(f"[vllm] base+LoRA 로드 (gpu_util={util}) — {self.model_id}")
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._llm = LLM(model=self.model_id, enable_lora=True, max_lora_rank=16, **common)
            self._lora = LoRARequest("summ", 1, self.adapter)
        print(f"[vllm] 전체 모델 로드 {time.perf_counter() - load_started:.1f}s")

    def summarize(self, name: str, genres: list[str], description: str) -> tuple[dict | None, str]:
        """원문 → 파인튜닝 3필드 요약(장르/핵심플레이/특징). (요약, 원본출력)."""
        self._ensure()
        from vllm import SamplingParams
        rec = {"name": name, "genres": genres, "input": description}
        text = self._tok.apply_chat_template(
            build_messages(rec, []), tokenize=False, add_generation_prompt=True)
        params = SamplingParams(temperature=0.0, max_tokens=320, stop=["}"])
        out = self._llm.generate([text], params, lora_request=self._lora)
        gen = out[0].outputs[0].text
        # stop=["}"] 는 } 를 출력에 안 넣으므로 파싱 전에 복원(HF stop_strings 와 형식 맞춤).
        return parse_summary(gen + "}"), gen
