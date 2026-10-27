"""Provider factory. Models and limits come from study_models.json, which is
filled in at run time from each provider's live model list and limits page."""
from __future__ import annotations

import json
from pathlib import Path

from .. import config
from ..ratelimit import Limits
from .base import CallResult, FatalError, Provider, ProviderUnavailable, QuotaExhausted, RetryableError
from .claude_code import ClaudeCodeProvider
from .gemini import GeminiProvider
from .openai_compat import GroqProvider, MistralProvider, NvidiaProvider, VLLMProvider

FAMILIES = {
    "claude": ClaudeCodeProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "nvidia": NvidiaProvider,
    "vllm": VLLMProvider,
}
STUDY_MODELS_FILE = config.ROOT / "study_models.json"


def load_study_models(path: Path = STUDY_MODELS_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def study_keys(enabled_only: bool = True, path: Path = STUDY_MODELS_FILE) -> list[str]:
    """Entries in study_models.json, e.g. "gemini" or "qwen_vllm"."""
    models = load_study_models(path)["models"]
    return [k for k, v in models.items() if v.get("enabled", True) or not enabled_only]


def adapter_class(key: str, spec: dict | None = None) -> type[Provider]:
    """An entry's adapter is its "adapter" field, defaulting to the entry key, so
    several entries (e.g. two self-hosted models) can share one adapter."""
    if spec is None:
        spec = load_study_models()["models"].get(key, {})
    return FAMILIES[spec.get("adapter", key)]


def make_provider(key: str, spec: dict | None = None) -> Provider:
    spec = spec or load_study_models()["models"][key]
    limits = Limits(**spec.get("limits", {}))
    extra = spec.get("options", {})
    p = adapter_class(key, spec)(spec["model"], limits, **extra)
    p.limiter.name = key  # daily budget per study entry (Groq limits are per model)
    return p


__all__ = ["CallResult", "FAMILIES", "FatalError", "Provider", "ProviderUnavailable",
           "QuotaExhausted", "RetryableError", "adapter_class", "load_study_models", "make_provider",
           "study_keys"]
