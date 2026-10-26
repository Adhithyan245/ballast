"""Client-side pacing: requests/minute, tokens/minute and a daily request
budget persisted to disk so the budget survives restarts."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config


class QuotaExhausted(RuntimeError):
    """Daily / window quota reached: stop this provider cleanly and resume later."""


@dataclass
class Limits:
    rpm: float | None = None   # requests per minute
    tpm: int | None = None     # tokens per minute (input + output)
    rpd: int | None = None     # requests per day (UTC day, client-side count)
    safety: float = 0.8        # use only this share of each published limit


class RateLimiter:
    def __init__(self, name: str, limits: Limits, usage_dir: Path | None = None,
                 clock=time.monotonic, sleep=time.sleep):
        self.name, self.limits = name, limits
        self.usage_dir = usage_dir or config.USAGE_DIR
        self.clock, self.sleep = clock, sleep
        self._last: float | None = None
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    # --- daily budget -------------------------------------------------
    def _day_file(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.usage_dir / f"{self.name}_{day}.json"

    def used_today(self) -> int:
        p = self._day_file()
        return json.loads(p.read_text())["requests"] if p.exists() else 0

    def _record_request(self) -> None:
        p = self._day_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        n = self.used_today() + 1
        p.write_text(json.dumps({"provider": self.name, "requests": n}))

    def daily_budget(self) -> int | None:
        if self.limits.rpd is None:
            return None
        return int(self.limits.rpd * self.limits.safety)

    # --- pacing -------------------------------------------------------
    def min_interval(self) -> float:
        if not self.limits.rpm:
            return 0.0
        return 60.0 / (self.limits.rpm * self.limits.safety)

    def acquire(self, est_tokens: int = 0) -> None:
        with self._lock:
            self._acquire(est_tokens)

    def _acquire(self, est_tokens: int) -> None:
        budget = self.daily_budget()
        if budget is not None and self.used_today() >= budget:
            raise QuotaExhausted(f"{self.name}: client-side daily budget {budget} reached")
        if self._last is not None:
            wait = self.min_interval() - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        if self.limits.tpm:
            cap = self.limits.tpm * self.limits.safety
            while True:
                now = self.clock()
                while self._tokens and now - self._tokens[0][0] > 60:
                    self._tokens.popleft()
                if not self._tokens or sum(t for _, t in self._tokens) + est_tokens <= cap:
                    break
                self.sleep(max(0.5, 60 - (now - self._tokens[0][0])))
        self._last = self.clock()
        self._record_request()

    def record_tokens(self, n: int) -> None:
        if n:
            with self._lock:
                self._tokens.append((self.clock(), n))
