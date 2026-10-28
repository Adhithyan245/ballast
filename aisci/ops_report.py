"""Operational summary of a run (calls, missing, parse failures, tokens, timing).

Deliberately reports nothing about answers or accuracy: this is used for the
pilot, whose results are not findings.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime

from . import config
from .cache import Cache
from .data import load_sample
from .parsing import ParseError, parse_response

SLEEP_GAP_S = 300  # gaps longer than this between consecutive calls are pauses, not work


def ops_summary(sample: str, planned_conditions: dict[str, list[str]],
                cache: Cache | None = None) -> list[dict]:
    n_q = len(load_sample(sample))
    by_model: dict[str, list[dict]] = defaultdict(list)
    for rec in (cache or Cache()).iter_records():
        if rec.get("sample") == sample:
            by_model[rec["key"]["model"]].append(rec)
    rows = []
    for model, conds in planned_conditions.items():
        recs = by_model.get(model, [])
        planned = n_q * config.RUNS_PER_ITEM * len(conds)
        status, modes = Counter(), Counter()
        for r in recs:
            try:
                p = parse_response(r.get("text"), require_quotes=r["condition"].startswith("context"))
                status["valid"] += 1
                modes[p.mode] += 1
            except ParseError:
                status["unparseable"] += 1
        times = sorted(datetime.fromisoformat(r["cached_at"]) for r in recs)
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        active = sum(g for g in gaps if g <= SLEEP_GAP_S)
        paused = sum(g for g in gaps if g > SLEEP_GAP_S)
        tin = [r.get("input_tokens") or 0 for r in recs]
        tout = [r.get("output_tokens") or 0 for r in recs]
        rows.append({
            "model": model,
            "conditions": conds,
            "planned_calls": planned,
            "cached_responses": len(recs),
            "not_yet_run_or_failed": planned - len(recs),
            "valid": status["valid"],
            "unparseable": status["unparseable"],
            "parse_failure_rate": round(status["unparseable"] / len(recs), 4) if recs else None,
            "parse_modes": dict(modes),
            "input_tokens": sum(tin),
            "output_tokens": sum(tout),
            "mean_tokens_per_call": round((sum(tin) + sum(tout)) / len(recs), 1) if recs else None,
            "median_latency_s": round(statistics.median(r["latency_s"] for r in recs), 2) if recs else None,
            "first_call_utc": times[0].isoformat() if times else None,
            "last_call_utc": times[-1].isoformat() if times else None,
            "active_minutes": round(active / 60, 1),
            "paused_minutes": round(paused / 60, 1),
            "calls_per_active_minute": round((len(recs) - 1) / (active / 60), 2) if active else None,
        })
    return rows


if __name__ == "__main__":
    import sys

    from .providers import load_study_models, make_provider, study_keys
    from .runner import conditions_for

    sample = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    planned = {}
    for key in study_keys():
        p = make_provider(key)
        planned[p.study_id] = conditions_for(p, include_t0=True)
    rows = ops_summary(sample, planned)
    out = config.ROOT / "results" / f"ops_{sample}.json"
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        print(json.dumps(r))
