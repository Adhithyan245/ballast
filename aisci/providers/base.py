"""Common provider interface, error classes and the retry loop."""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..cache import log_retry
from ..ratelimit import Limits, QuotaExhausted, RateLimiter


class RetryableError(RuntimeError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class FatalError(RuntimeError):
    """Bad request for this item (e.g. 400). Logged as missing, not retried."""


class ProviderUnavailable(RuntimeError):
    """Auth failure or similar: stop the provider, nothing will succeed."""


@dataclass
class CallResult:
    text: str | None             # model's final text (thinking excluded)
    raw: dict                    # full provider response, saved verbatim
    model_reported: str | None   # model name/version the provider says it served
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)


class Provider(ABC):
    family: str = ""
    key_var: str | None = None  # env var holding the API key, if any
    supports_temperature: bool = True

    max_retry_wait: float = 300.0  # seconds; longer retry-after hints stop the provider

    def __init__(self, model: str, limits: Limits, timeout: float = 120.0,
                 max_attempts: int = 5, sleep=time.sleep):
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.limiter = RateLimiter(self.family, limits, sleep=sleep)

    @property
    def study_id(self) -> str:
        return f"{self.family}:{self.model}"

    @abstractmethod
    def _call(self, prompt: str, temperature: float | None) -> CallResult:
        ...

    def call(self, prompt: str, temperature: float | None = None) -> CallResult:
        if temperature is not None and not self.supports_temperature:
            raise ValueError(f"{self.family} does not support setting temperature")
        est = len(prompt) // 3 + 800
        for attempt in range(1, self.max_attempts + 1):
            self.limiter.acquire(est_tokens=est)
            t0 = time.monotonic()
            try:
                res = self._call(prompt, temperature)
            except RetryableError as e:
                wait = max(min(120.0, 2.0 ** attempt + random.uniform(0, 1)), e.retry_after or 0.0)
                log_retry({"provider": self.limiter.name, "model": self.model, "attempt": attempt,
                           "error": str(e)[:300], "retry_after": e.retry_after, "wait_s": round(wait, 1)})
                if e.retry_after and e.retry_after > self.max_retry_wait:
                    # A provider asking us to wait this long is effectively out of quota:
                    # stop cleanly and resume later instead of sleeping for hours.
                    raise QuotaExhausted(f"retry-after {e.retry_after:.0f}s > {self.max_retry_wait:.0f}s: {e}") from None
                if attempt == self.max_attempts:
                    raise
                self.sleep(wait)
                continue
            res.latency_s = time.monotonic() - t0
            self.limiter.record_tokens((res.input_tokens or 0) + (res.output_tokens or 0))
            return res
        raise AssertionError("unreachable")


__all__ = ["CallResult", "FatalError", "Limits", "Provider", "ProviderUnavailable",
           "QuotaExhausted", "RetryableError"]
