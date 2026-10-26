"""Resumable study runner: skips anything already cached, logs failures as
missing, and stops a provider cleanly when its quota is reached."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .cache import Cache, Key, log_error
from .data import Item
from .prompts import build_prompt, prompt_hash
from .providers import FatalError, Provider, ProviderUnavailable, QuotaExhausted, RetryableError


def conditions_for(provider: Provider, include_t0: bool) -> list[str]:
    conds = [f"{c}_default" for c in config.CONTEXTS]
    if include_t0 and provider.supports_temperature:
        conds += [f"{c}_t0" for c in config.CONTEXTS]
    return conds


def temperature_of(condition: str) -> float | None:
    return 0.0 if condition.endswith("_t0") else None


@dataclass
class RunSummary:
    model: str
    done: int = 0
    cached: int = 0
    failed: int = 0
    stopped: str | None = None
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: list[str] = field(default_factory=list)


def run_provider(provider: Provider, items: list[Item], conditions: list[str],
                 sample: str, runs: int = config.RUNS_PER_ITEM, cache: Cache | None = None,
                 max_calls: int | None = None, log=print) -> RunSummary:
    cache = cache or Cache()
    s = RunSummary(provider.study_id)
    t_start = time.monotonic()
    # Question-major order: a partial run leaves complete questions, not scattered runs.
    for cond in conditions:
        ctx = cond.split("_")[0]
        for item in items:
            prompt = build_prompt(ctx, item.question, item.abstract if ctx == "context" else None)
            for run in range(1, runs + 1):
                key = Key(provider.study_id, cond, item.qid, run)
                if cache.has(key):
                    s.cached += 1
                    continue
                if max_calls is not None and s.done + s.failed >= max_calls:
                    s.stopped = f"max_calls={max_calls} reached"
                    s.seconds = time.monotonic() - t_start
                    return s
                try:
                    res = provider.call(prompt, temperature_of(cond))
                except (QuotaExhausted, ProviderUnavailable) as e:
                    s.stopped = f"{type(e).__name__}: {e}"
                    log_error({"model": provider.study_id, **key.__dict__,
                               "error": s.stopped, "kind": type(e).__name__})
                    s.seconds = time.monotonic() - t_start
                    return s
                except (RetryableError, FatalError) as e:
                    s.failed += 1
                    s.failures.append(f"{cond}/{item.qid}/run{run}: {e}")
                    log_error({"model": provider.study_id, **key.__dict__,
                               "error": str(e)[:500], "kind": type(e).__name__})
                    continue
                cache.put(key, {
                    "sample": sample,
                    "family": provider.family,
                    "model_requested": provider.model,
                    "model_reported": res.model_reported,
                    "condition": cond,
                    "temperature": temperature_of(cond),
                    "prompt_hash": prompt_hash(prompt),
                    "prompt": prompt,
                    "text": res.text,
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "latency_s": round(res.latency_s, 3),
                    "meta": res.meta,
                    "raw": res.raw,
                })
                s.done += 1
                s.input_tokens += res.input_tokens or 0
                s.output_tokens += res.output_tokens or 0
                if s.done % 10 == 0:
                    log(f"[{provider.study_id}] {s.done} new calls, {s.cached} cached, {s.failed} failed")
    s.seconds = time.monotonic() - t_start
    return s
