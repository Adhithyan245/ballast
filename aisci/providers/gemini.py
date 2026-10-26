"""Gemini via the Google AI Studio REST API (generateContent)."""
from __future__ import annotations

import httpx

from .base import CallResult, Provider
from .http import get_json, post_json, require_key

BASE = "https://generativelanguage.googleapis.com/v1beta"


def _is_daily(resp: httpx.Response) -> bool:
    try:
        details = resp.json()["error"].get("details", [])
    except Exception:
        return False
    for d in details:
        for v in d.get("violations", []):
            if "PerDay" in v.get("quotaId", "") or "per_day" in v.get("quotaMetric", ""):
                return True
    return False


def _retry_delay(resp: httpx.Response) -> float | None:
    try:
        raw = resp.json()
    except Exception:
        return None
    for d in raw.get("error", {}).get("details", []):
        if d.get("@type", "").endswith("RetryInfo"):
            try:
                return float(d.get("retryDelay", "").rstrip("s"))
            except ValueError:
                return None
    return None


class GeminiProvider(Provider):
    family = "gemini"

    def __init__(self, model, limits, **kw):
        super().__init__(model, limits, **kw)
        self.client = httpx.Client()

    def _headers(self) -> dict:
        return {"x-goog-api-key": require_key("GEMINI_API_KEY")}

    def _call(self, prompt, temperature):
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if temperature is not None:
            body["generationConfig"] = {"temperature": temperature}
        raw, _ = post_json(self.client, f"{BASE}/models/{self.model}:generateContent",
                           self._headers(), body, self.timeout, _is_daily, _retry_delay)
        return self.parse(raw)

    @staticmethod
    def parse(raw: dict) -> CallResult:
        text = None
        cands = raw.get("candidates") or []
        if cands:
            parts = (cands[0].get("content") or {}).get("parts") or []
            texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
            text = "".join(texts) if texts else None
        um = raw.get("usageMetadata") or {}
        out = (um.get("candidatesTokenCount") or 0) + (um.get("thoughtsTokenCount") or 0)
        return CallResult(
            text=text, raw=raw, model_reported=raw.get("modelVersion"),
            input_tokens=um.get("promptTokenCount"), output_tokens=out or None,
            meta={"finish_reason": cands[0].get("finishReason") if cands else None,
                  "thoughts_tokens": um.get("thoughtsTokenCount")},
        )

    def list_models(self) -> list[str]:
        raw, _ = get_json(f"{BASE}/models?pageSize=1000", self._headers())
        return sorted(m["name"].removeprefix("models/") for m in raw.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))
