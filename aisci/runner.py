"""Resumable study runner: skips anything already cached, logs failures as
missing, and stops a provider cleanly when its quota is reached.

Pushback (sycophancy probe): for "<ctx>_pushback" conditions, the runner takes
run PUSHBACK_RUN of "<ctx>_default" as the model's first answer and sends a
second turn in the same conversation pushing toward a different answer.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import config
from .cache import Cache, Key, log_error
from .data import Item
from .parsing import ParseError, parse_response
from .prompts import build_prompt, build_pushback, prompt_hash, pushback_suggestion
from .providers import FatalError, Provider, ProviderUnavailable, QuotaExhausted, RetryableError


def conditions_for(provider: Provider, include_t0: bool, include_pushback: bool = True) -> list[str]:
    conds = [f"{c}_default" for c in config.CONTEXTS]
    if include_t0 and provider.supports_temperature:
        conds += [f"{c}_t0" for c in config.CONTEXTS]
    if include_pushback:
        conds += [f"{c}_pushback" for c in config.CONTEXTS]
    return conds


def temperature_of(condition: str) -> float | None:
    return 0.0 if condition.endswith("_t0") else None


@dataclass
class RunSummary:
    model: str
    done: int = 0
    cached: int = 0
    failed: int = 0
    failed_transient: int = 0  # timeouts / connection / 5xx after all retries: worth trying later
    skipped: int = 0  # pushback turns with no valid first answer to push back on
    stopped: str | None = None
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: list[str] = field(default_factory=list)


@dataclass
class Task:
    key: Key
    prompt: str
    temperature: float | None = None
    history: list[dict] | None = None
    session: dict | None = None
    extra: dict = field(default_factory=dict)  # merged into the cached record


def _prompt_for(item: Item, ctx: str) -> str:
    return build_prompt(ctx, item.question, item.abstract if ctx == "context" else None)


def plan_calls(provider: Provider, items: list[Item], conditions: list[str], runs: int):
    """First-turn calls, question-major so a partial run leaves complete questions.
    Run PUSHBACK_RUN keeps its conversation so a pushback turn can follow it."""
    for cond in conditions:
        if cond.endswith("_pushback"):
            continue
        ctx = cond.split("_")[0]
        for item in items:
            prompt = _prompt_for(item, ctx)
            for run in range(1, runs + 1):
                persist = cond.endswith("_default") and run == config.PUSHBACK_RUN
                yield Task(Key(provider.study_id, cond, item.qid, run), prompt,
                           temperature_of(cond), session={"persist": True} if persist else None,
                           extra={"session_persisted": persist})


def _first_answer(rec: dict | None, ctx: str):
    """Parsed first answer of a cached record, or None if missing/unparseable."""
    if rec is None:
        return None
    try:
        return parse_response(rec.get("text"), require_quotes=(ctx == "context"))
    except ParseError:
        return None


def plan_pushback(provider: Provider, items: list[Item], conditions: list[str], cache: Cache,
                  s: RunSummary) -> list[Task]:
    """Second-turn tasks. Needs the first answer to be cached and valid; Claude also
    needs a saved session. Where a first answer exists but its Claude session was not
    saved (pilot runs made before the pushback condition existed), a fresh first turn
    is made under "<ctx>_pushinit" and the pushback continues that one."""
    tasks = []
    for cond in conditions:
        if not cond.endswith("_pushback"):
            continue
        ctx = cond.split("_")[0]
        for item in items:
            key = Key(provider.study_id, cond, item.qid, config.PUSHBACK_RUN)
            if cache.has(key):
                s.cached += 1
                continue
            init_cond = f"{ctx}_default"
            rec = cache.get(Key(provider.study_id, init_cond, item.qid, config.PUSHBACK_RUN))
            if (provider.family == "claude" and rec is not None
                    and not rec.get("session_persisted")):
                init_cond = f"{ctx}_pushinit"
                rec = cache.get(Key(provider.study_id, init_cond, item.qid, config.PUSHBACK_RUN))
            first = _first_answer(rec, ctx)
            if first is None:
                if rec is not None:
                    s.skipped += 1  # first answer unparseable: pushback is missing by design
                continue
            suggested = pushback_suggestion(first.answer, item.qid, config.SEED)
            session = None
            if provider.family == "claude":
                session = {"resume": (rec.get("meta") or {}).get("session_id")
                           or rec["raw"].get("session_id")}
            tasks.append(Task(
                key, build_pushback(suggested), None,
                history=[{"user": rec["prompt"], "assistant": rec["text"]}], session=session,
                extra={"initial_condition": init_cond, "initial_run": config.PUSHBACK_RUN,
                       "initial_answer": first.answer, "initial_confidence": first.confidence,
                       "suggested_answer": suggested, "history": [{"user": rec["prompt"],
                                                                    "assistant": rec["text"]}]}))
    return tasks


def plan_pushinit(provider: Provider, items: list[Item], conditions: list[str], cache: Cache):
    """Fresh, session-saving first turns for Claude where the cached first turn has none."""
    if provider.family != "claude":
        return []
    tasks = []
    for cond in conditions:
        if not cond.endswith("_pushback"):
            continue
        ctx = cond.split("_")[0]
        for item in items:
            rec = cache.get(Key(provider.study_id, f"{ctx}_default", item.qid, config.PUSHBACK_RUN))
            init_key = Key(provider.study_id, f"{ctx}_pushinit", item.qid, config.PUSHBACK_RUN)
            if (rec is not None and not rec.get("session_persisted") and not cache.has(init_key)
                    and not cache.has(Key(provider.study_id, cond, item.qid, config.PUSHBACK_RUN))):
                tasks.append(Task(init_key, _prompt_for(item, ctx), None, session={"persist": True},
                                  extra={"session_persisted": True}))
    return tasks


def run_provider(provider: Provider, items: list[Item], conditions: list[str],
                 sample: str, runs: int = config.RUNS_PER_ITEM, cache: Cache | None = None,
                 max_calls: int | None = None, workers: int = 1, log=print) -> RunSummary:
    """Run every planned call not yet cached. workers > 1 sends calls in
    parallel (only for providers without per-minute limits, e.g. self-hosted)."""
    cache = cache or Cache()
    s = RunSummary(provider.study_id)
    t_start = time.monotonic()
    lock = threading.Lock()
    stop = threading.Event()
    budget = [max_calls]

    def one(task: Task):
        key = task.key
        if stop.is_set():
            return
        try:
            res = provider.call(task.prompt, task.temperature, task.history, task.session)
        except (QuotaExhausted, ProviderUnavailable) as e:
            with lock:
                if not stop.is_set():
                    s.stopped = f"{type(e).__name__}: {e}"
                stop.set()
            log_error({"model": provider.study_id, **key.__dict__,
                       "error": str(e)[:500], "kind": type(e).__name__})
            return
        except (RetryableError, FatalError) as e:
            with lock:
                s.failed += 1
                s.failed_transient += isinstance(e, RetryableError)
                s.failures.append(f"{key.condition}/{key.qid}/run{key.run}: {e}")
            log_error({"model": provider.study_id, **key.__dict__,
                       "error": str(e)[:500], "kind": type(e).__name__})
            return
        cache.put(key, {
            "sample": sample,
            "family": provider.family,
            "model_requested": provider.model,
            "model_reported": res.model_reported,
            "condition": key.condition,
            "temperature": task.temperature,
            "prompt_hash": prompt_hash(task.prompt),
            "prompt": task.prompt,
            **task.extra,
            "text": res.text,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "latency_s": round(res.latency_s, 3),
            "meta": res.meta,
            "raw": res.raw,
        })
        with lock:
            s.done += 1
            s.input_tokens += res.input_tokens or 0
            s.output_tokens += res.output_tokens or 0
            if s.done % 10 == 0:
                log(f"[{provider.study_id}] {s.done} new calls, {s.cached} cached, {s.failed} failed")

    def execute(tasks: list[Task]) -> None:
        pending = []
        for t in tasks:
            if cache.has(t.key):
                s.cached += 1
            else:
                pending.append(t)
        if budget[0] is not None:
            if len(pending) > budget[0]:
                pending = pending[:budget[0]]
                s.stopped = f"max_calls={max_calls} reached"
            budget[0] -= len(pending)
        if workers <= 1:
            for t in pending:
                if stop.is_set():
                    break
                one(t)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one, pending))

    execute(list(plan_calls(provider, items, conditions, runs)))
    # Pushback turns depend on first answers, so they are planned after those are cached.
    for phase in (plan_pushinit, None):
        if stop.is_set() or (budget[0] is not None and budget[0] <= 0):
            break
        if phase is plan_pushinit:
            execute(plan_pushinit(provider, items, conditions, cache))
        else:
            pb = plan_pushback(provider, items, conditions, cache, s)
            pending = [t for t in pb if not cache.has(t.key)]
            execute(pending)
    s.seconds = time.monotonic() - t_start
    return s
