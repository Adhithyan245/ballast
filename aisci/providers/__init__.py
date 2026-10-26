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
from .openai_compat import GroqProvider, MistralProvider

FAMILIES = {
    "claude": ClaudeCodeProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
}
STUDY_MODELS_FILE = config.ROOT / "study_models.json"


def load_study_models(path: Path = STUDY_MODELS_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_provider(family: str, spec: dict | None = None) -> Provider:
    spec = spec or load_study_models()["models"][family]
    limits = Limits(**spec.get("limits", {}))
    extra = spec.get("options", {})
    return FAMILIES[family](spec["model"], limits, **extra)


__all__ = ["CallResult", "FAMILIES", "FatalError", "Provider", "ProviderUnavailable",
           "QuotaExhausted", "RetryableError", "load_study_models", "make_provider"]
