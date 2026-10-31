from datetime import datetime, timezone

from aisci.driver import next_resume


def test_next_resume_daily_and_claude_window():
    now = datetime(2026, 10, 30, 20, 0, tzinfo=timezone.utc)
    assert next_resume("groq", "QuotaExhausted: daily", now) == datetime(2026, 10, 31, 0, 5, tzinfo=timezone.utc)
    assert next_resume("gemini", "QuotaExhausted: budget", now) == datetime(2026, 10, 31, 8, 5, tzinfo=timezone.utc)
    early = datetime(2026, 10, 31, 3, 0, tzinfo=timezone.utc)
    assert next_resume("gemini", "QuotaExhausted", early) == datetime(2026, 10, 31, 8, 5, tzinfo=timezone.utc)
    assert next_resume("claude", "QuotaExhausted: claude usage limit: reached", now) == datetime(2026, 10, 30, 21, 0, tzinfo=timezone.utc)
    assert next_resume("claude", "QuotaExhausted: client-side daily budget", now) == datetime(2026, 10, 31, 0, 5, tzinfo=timezone.utc)


def test_transient_failures_are_counted_separately(tmp_path, monkeypatch):
    from aisci.cache import Cache
    from aisci.data import Item
    from aisci.providers.base import CallResult, FatalError, Provider, RetryableError
    from aisci.ratelimit import Limits
    from aisci.runner import run_provider

    monkeypatch.setattr("aisci.config.USAGE_DIR", tmp_path / "u")
    monkeypatch.setattr("aisci.config.LOG_DIR", tmp_path / "l")

    class Flaky(Provider):
        family = "fake"

        def __init__(self, errs):
            super().__init__("m", Limits(), max_attempts=1, sleep=lambda _: None)
            self.errs = list(errs)

        def _call(self, prompt, temperature, history=None, session=None):
            raise self.errs.pop(0)

    items = [Item("1", "q?", ("c",), ("",), "yes")]
    s = run_provider(Flaky([RetryableError("transport: ConnectError"), FatalError("400")]), items,
                     ["closed_default"], "pilot", runs=2, cache=Cache(tmp_path / "c"), log=lambda *_: None)
    assert (s.failed, s.failed_transient, s.done) == (2, 1, 0)


def test_groq_rolling_token_limit_waits_30_minutes():
    now = datetime(2026, 10, 31, 0, 6, tzinfo=timezone.utc)
    reason = "QuotaExhausted: 429 daily quota: Rate limit reached ... on tokens per day (TPD): Limit 200000"
    assert next_resume("groq", reason, now) == datetime(2026, 10, 31, 0, 36, tzinfo=timezone.utc)
