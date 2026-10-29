"""Turn cached raw records into per-question observations."""
from __future__ import annotations

from collections import defaultdict

from . import config
from .cache import Cache
from .metrics import Obs, Pushback, missing_counts
from .parsing import ParseError, parse_response


def record_to_obs(rec: dict) -> Obs:
    require_quotes = rec["condition"].startswith("context")
    try:
        p = parse_response(rec.get("text"), require_quotes=require_quotes)
    except ParseError:
        return Obs(None, status="unparseable")
    return Obs(p.answer, p.confidence, p.quotes)


def load_observations(sample: str | None = None, cache: Cache | None = None):
    """{(model, condition): {qid: [Obs, ...]}} for first-turn conditions. Pushback
    replies and Claude's "pushinit" first turns are loaded by load_pushbacks instead."""
    out: dict[tuple[str, str], dict[str, list[Obs]]] = defaultdict(lambda: defaultdict(list))
    for rec in (cache or Cache()).iter_records():
        if sample and rec.get("sample") != sample:
            continue
        if rec["condition"].endswith(("_pushback", "_pushinit")):
            continue
        k = rec["key"]
        out[(k["model"], k["condition"])][k["qid"]].append(record_to_obs(rec))
    return out


def status_table(sample: str | None = None, cache: Cache | None = None) -> list[dict]:
    from .data import load_sample

    n_q = len(load_sample(sample)) if sample else None
    rows = []
    for (model, cond), qs in sorted(load_observations(sample, cache).items()):
        rows.append({"model": model, "condition": cond, "questions_with_data": len(qs),
                     **missing_counts(qs, config.RUNS_PER_ITEM, n_q)})
    return rows


def load_pushbacks(sample: str | None = None, cache: Cache | None = None):
    """{(model, context): {qid: Pushback}}. The first answer is the record the pushback
    continued (its `initial_condition`), parsed by the same rule as every other answer."""
    recs = {}
    for rec in (cache or Cache()).iter_records():
        if sample and rec.get("sample") != sample:
            continue
        k = rec["key"]
        recs[(k["model"], k["condition"], k["qid"], k["run"])] = rec
    out: dict[tuple[str, str], dict[str, Pushback]] = defaultdict(dict)
    for (model, cond, qid, _run), rec in recs.items():
        if not cond.endswith("_pushback"):
            continue
        init = recs.get((model, rec["initial_condition"], qid, rec["initial_run"]))
        initial = record_to_obs(init) if init else Obs(None, status="no_response")
        out[(model, cond.split("_")[0])][qid] = Pushback(initial, record_to_obs(rec),
                                                        rec["suggested_answer"])
    return out
