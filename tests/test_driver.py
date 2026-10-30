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
