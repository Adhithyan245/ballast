"""Multi-day driver: run one study entry to completion across quota resets.

    python -m aisci.driver main gemini

Loops run_provider. When the provider stops on quota, it waits until that quota
resets, then resumes from the cache. Ends when every planned call is cached,
when a pass makes no progress, or at the preregistered stopping date. Waiting is
done in short sleeps against the wall clock, so a laptop that sleeps and wakes
picks up where it left off.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from . import config, data
from .providers import load_study_models, make_provider
from .runner import conditions_for, run_provider

STOP_AT = datetime(2026, 11, 10, 0, 0, tzinfo=timezone.utc)  # end of the 2026-11-09 stopping date
# When each provider's daily quota resets (UTC). The client-side budget resets at 00:00 UTC;
# Google's free tier resets at midnight Pacific, which is 07:00 or 08:00 UTC, so wait for 08:05.
RESET_UTC = {"gemini": (8, 5)}
DEFAULT_RESET_UTC = (0, 5)
CLAUDE_WINDOW_WAIT = timedelta(minutes=60)  # Claude usage limits are rolling windows, not daily


def next_resume(family: str, reason: str, now: datetime) -> datetime:
    if family == "claude" and "usage limit" in reason:
        return now + CLAUDE_WINDOW_WAIT
    h, m = RESET_UTC.get(family, DEFAULT_RESET_UTC)
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def wait_until(t: datetime, log) -> None:
    log(f"waiting until {t.isoformat()}")
    while datetime.now(timezone.utc) < t:
        time.sleep(60)


def main(sample: str, key: str) -> None:
    load_dotenv(config.ROOT / ".env")

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"{sample}_{key}.log"

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} [{key}] {msg}"
        print(line, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    items = data.load()
    chosen = [items[q] for q in data.load_sample(sample)]
    spec = load_study_models()["models"][key]
    while datetime.now(timezone.utc) < STOP_AT:
        p = make_provider(key)
        s = run_provider(p, chosen, conditions_for(p, include_t0=True), sample,
                         workers=spec.get("workers", 1), log=log)
        log(f"pass: new={s.done} cached={s.cached} failed={s.failed} skipped={s.skipped} "
            f"tokens in/out={s.input_tokens}/{s.output_tokens} stopped={s.stopped}")
        if s.stopped and ("QuotaExhausted" in s.stopped or "usage limit" in s.stopped):
            wait_until(min(next_resume(p.family, s.stopped, datetime.now(timezone.utc)), STOP_AT), log)
            continue
        if s.stopped:  # ProviderUnavailable (e.g. auth): needs a person
            log("stopping: provider unavailable; fix it and rerun this command to resume")
            return
        if s.done == 0:
            log("finished: nothing left to request (remaining gaps, if any, are logged failures)")
            return
        if s.failed:
            continue  # retry calls that got no response, as the preregistration allows
        log("finished: all planned calls cached")
        return
    log("stopping date reached; remaining calls are reported as incomplete")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
