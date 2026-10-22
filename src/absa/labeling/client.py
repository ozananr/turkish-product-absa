"""OpenAI uyumlu sohbet API'si için ince istemci.

OpenAI, Gemini (OpenAI uyumlu uç nokta) ve benzeri sağlayıcılar aynı kodla çalışır:
yalnızca LLM_BASE_URL ve LLM_MODEL değişir. Ayarlar .env dosyasından okunur.
Rate limit (429) ve sunucu hataları için yeniden deneme openai SDK'sı tarafından yapılır.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float | None = 0.0
    json_mode: bool = True
    reasoning_effort: str | None = None
    timeout: float = 60.0
    max_retries: int = 6

    @classmethod
    def from_env(cls, model: str | None = None, base_url: str | None = None) -> LLMConfig:
        """.env'i yükler ve ayarları ortam değişkenlerinden okur."""
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        model = model or os.getenv("LLM_MODEL") or ""
        if not api_key:
            raise RuntimeError("LLM_API_KEY (ya da OPENAI_API_KEY) .env dosyasında tanımlı değil.")
        if not model:
            raise RuntimeError("Model belirtilmedi: .env'e LLM_MODEL yazın ya da --model verin.")
        temperature = os.getenv("LLM_TEMPERATURE", "0")
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url or os.getenv("LLM_BASE_URL") or None,
            # Bazı akıl yürütme modelleri temperature kabul etmez: LLM_TEMPERATURE=none
            temperature=None if temperature.lower() == "none" else float(temperature),
            json_mode=os.getenv("LLM_JSON_MODE", "1") != "0",
            # Düşünme token'ları çıktı olarak faturalanır; basit sınıflandırmada "none"/"low" yeter.
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT") or None,
        )


@dataclass
class Completion:
    text: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


class ChatClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, str]]) -> Completion: ...


class OpenAIChatClient:
    """openai SDK ile Chat Completions çağrısı yapar."""

    def __init__(self, config: LLMConfig):
        from openai import OpenAI

        self.config = config
        self.model = config.model
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        kwargs: dict = {"model": self.model, "messages": messages}
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        response = self._client.chat.completions.create(**kwargs)

        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        return Completion(
            text=response.choices[0].message.content if response.choices else None,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
        )
