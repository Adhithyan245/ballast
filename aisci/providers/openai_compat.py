"""Groq and Mistral both expose an OpenAI-style /chat/completions endpoint."""
from __future__ import annotations

import os

import httpx

from .base import CallResult, Provider
from .http import get_json, post_json, require_key


class ChatCompletionsProvider(Provider):
    base_url: str = ""
    key_var: str = ""

    def __init__(self, model, limits, **kw):
        super().__init__(model, limits, **kw)
        self.client = httpx.Client()
        self.last_headers: dict = {}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {require_key(self.key_var)}"}

    @staticmethod
    def _is_daily(resp: httpx.Response) -> bool:
        msg = resp.text.lower()
        return "per day" in msg or "per month" in msg or "monthly" in msg

    def _call(self, prompt, temperature):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            body["temperature"] = temperature
        raw, headers = post_json(self.client, f"{self.base_url}/chat/completions",
                                 self._headers(), body, self.timeout, self._is_daily)
        self.last_headers = {k: v for k, v in headers.items() if "ratelimit" in k.lower()}
        res = self.parse(raw)
        res.meta["ratelimit_headers"] = self.last_headers
        return res

    @staticmethod
    def parse(raw: dict) -> CallResult:
        choices = raw.get("choices") or []
        msg = choices[0].get("message", {}) if choices else {}
        content = msg.get("content")
        if isinstance(content, list):  # Mistral may return typed chunks
            content = "".join(c.get("text", "") for c in content if c.get("type") == "text") or None
        usage = raw.get("usage") or {}
        # Reasoning models (gpt-oss on Groq, Qwen3 behind vLLM's reasoning parser) return
        # their reasoning in a separate field; it stays in `raw` and is never parsed as the answer.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        return CallResult(
            text=content, raw=raw, model_reported=raw.get("model"),
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
            meta={"finish_reason": choices[0].get("finish_reason") if choices else None,
                  "has_reasoning": bool(reasoning)},
        )

    def list_models(self) -> list[str]:
        raw, _ = get_json(f"{self.base_url}/models", self._headers())
        return sorted(m["id"] for m in raw.get("data", []))


class GroqProvider(ChatCompletionsProvider):
    family = "groq"
    base_url = "https://api.groq.com/openai/v1"
    key_var = "GROQ_API_KEY"


class MistralProvider(ChatCompletionsProvider):
    family = "mistral"
    base_url = "https://api.mistral.ai/v1"
    key_var = "MISTRAL_API_KEY"


class NvidiaProvider(ChatCompletionsProvider):
    """NVIDIA's hosted API catalog (build.nvidia.com), developer access."""
    family = "nvidia"
    base_url = "https://integrate.api.nvidia.com/v1"
    key_var = "NVIDIA_API_KEY"


class VLLMProvider(ChatCompletionsProvider):
    """Open-weight model we host ourselves with vLLM on a rented NVIDIA Brev GPU
    (see docs/BREV_SETUP.md). The server URL and its --api-key come from .env;
    nothing is daily-capped, so a 429 is always treated as retryable."""
    family = "vllm"
    key_var = "VLLM_API_KEY"

    def __init__(self, model, limits, **kw):
        super().__init__(model, limits, **kw)
        self.base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")

    @staticmethod
    def _is_daily(resp: httpx.Response) -> bool:
        return False
