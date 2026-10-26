"""Turn cached raw records into per-question observations."""
from __future__ import annotations

from collections import defaultdict

from . import config
from .cache import Cache
from .metrics import Obs, missing_counts
from .parsing import ParseError, parse_response


def record_to_obs(rec: dict) -> Obs:
    require_quotes = rec["condition"].startswith("context")
    try:
        p = parse_response(rec.get("text"), require_quotes=require_quotes)
    except ParseError:
        return Obs(None, status="unparseable")
    return Obs(p.answer, p.confidence, p.quotes)


def load_observations(sample: str | None = None, cache: Cache | None = None):
    """{(model, condition): {qid: [Obs, ...]}} from every cached record."""
    out: dict[tuple[str, str], dict[str, list[Obs]]] = defaultdict(lambda: defaultdict(list))
    for rec in (cache or Cache()).iter_records():
        if sample and rec.get("sample") != sample:
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
