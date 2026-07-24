"""LLM 클라이언트 추상화.

다양한 백엔드 지원:
- OpenAI-compatible (vLLM, TGI, Ollama, OpenAI API) — `OpenAICompatibleClient`
- Mock (테스트/CI) — `MockLLMClient`
- 추후 파인튜닝 모델 — 동일 인터페이스로 교체

모든 클라이언트는 `generate(prompt, system=None, ...) -> str` 인터페이스 제공.
"""
from __future__ import annotations

import abc
import json
import os
from typing import Any

import httpx
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class LLMError(RuntimeError):
    pass


class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """프롬프트 → 응답 텍스트."""


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI-compatible /chat/completions 엔드포인트."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "not-required",
        timeout_sec: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "not-required":
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise LLMError(f"LLM HTTP error: {e}") from e

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected LLM response shape: {data}") from e


class MLXLoRAClient(BaseLLMClient):
    """MLX-based LoRA inference client."""

    def __init__(self, model_path: str, adapter_path: str | None = None) -> None:
        if not MLX_AVAILABLE:
            raise LLMError("mlx_lm not installed")
        self.model_path = model_path
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None:
            # Lazy loading
            self._model, self._tokenizer = load(
                self.model_path,
                adapter_path=self.adapter_path
            )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        self._ensure_model()
        full_prompt = prompt
        if system:
            # Simple system prompt wrapping if not handled by chat template
            full_prompt = f"{system}\n\n{prompt}"

        # EXAONE-2.4B chat template handling (simple version)
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            full_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        resp = generate(
            self._model,
            self._tokenizer,
            prompt=full_prompt,
            temp=temperature,
            max_tokens=max_tokens,
            verbose=False,
        )
        return resp


class MockLLMClient(BaseLLMClient):
    """테스트용 mock 클라이언트.

    응답은 환경변수 `MOCK_LLM_RESPONSES` (JSON list[str])에서 가져옴.
    비어있으면 빈 응답 반환.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        if responses is not None:
            self._responses = list(responses)
            self._idx = 0
            return
        env = os.getenv("MOCK_LLM_RESPONSES")
        if env:
            try:
                self._responses = json.loads(env)
                self._idx = 0
                return
            except json.JSONDecodeError:
                pass
        self._responses = []
        self._idx = 0

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        if self._idx >= len(self._responses):
            return ""
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


_client_singleton: BaseLLMClient | None = None


def get_llm_client(model_name: str | None = None) -> BaseLLMClient:
    """설정에 따른 기본 클라이언트 반환.

    환경변수 `STEAM_PART_D_MOCK_LLM=1` 이면 mock 사용.
    model_name 명시 시 해당 모델로 클라이언트 생성 (캐시 키에 포함).
    """
    global _client_singleton

    # model_name별 캐시 (요청마다 다른 모델 사용 가능)
    cache_key = model_name or "__default__"
    if (
        _client_singleton is not None
        and getattr(_client_singleton, "_cache_key", None) == cache_key
    ):
        return _client_singleton

    if os.getenv("STEAM_PART_D_MOCK_LLM") == "1":
        from steam_part_d.config import get_settings

        _ = get_settings()
        _client_singleton = MockLLMClient()
        _client_singleton._cache_key = cache_key
        return _client_singleton

    from steam_part_d.config import get_settings

    s = get_settings().llm
    # model_name 명시 → 우선 사용. lora_enabled=True면 .env의 model_name(LoRA 등록된 이름) 사용.
    if model_name is not None:
        chosen_model = model_name
    elif s.lora_enabled:
        chosen_model = s.model_name  # 이게 이미 ollama에 LoRA 등록된 이름
    else:
        chosen_model = s.model_name

    _client_singleton = OpenAICompatibleClient(
        base_url=s.base_url,
        model_name=chosen_model,
        api_key=s.api_key,
        timeout_sec=s.timeout_sec,
    )
    _client_singleton._cache_key = cache_key
    return _client_singleton


def reset_llm_client() -> None:
    """싱글톤 리셋 (테스트용)."""
    global _client_singleton
    _client_singleton = None
