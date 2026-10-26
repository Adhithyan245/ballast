"""Deterministic metrics. No LLM judge anywhere in this module.

A run observation is either a valid parsed answer or missing (no response /
unparseable). Missing runs are excluded from denominators and counted
separately; they are never treated as answers.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import config


@dataclass(frozen=True)
class Obs:
    answer: str | None              # None = missing
    confidence: float | None = None
    quotes: tuple[str, ...] | None = None
    status: str = "ok"              # ok | unparseable | no_response

    @property
    def valid(self) -> bool:
        return self.answer is not None


# --- quotes ---------------------------------------------------------------
def normalize(s: str) -> str:
    """Collapse all whitespace to single spaces and casefold."""
    return " ".join(s.split()).casefold()


def quote_found(quote: str, abstract: str) -> bool:
    q = normalize(quote)
    return bool(q) and q in normalize(abstract)


def quote_existence(runs: list[Obs], abstract: str) -> tuple[int, int]:
    """(quotes found, quotes given) over valid runs."""
    found = total = 0
    for o in runs:
        if o.valid and o.quotes is not None:
            for q in o.quotes:
                total += 1
                found += quote_found(q, abstract)
    return found, total


# --- consistency ------------------------------------------------------------
def valid_answers(runs: list[Obs]) -> list[str]:
    return [o.answer for o in runs if o.valid]


def flipped(runs: list[Obs], min_valid: int = config.RUNS_PER_ITEM) -> bool | None:
    """True if valid runs disagree; None if fewer than min_valid valid runs."""
    ans = valid_answers(runs)
    if len(ans) < max(2, min_valid):
        return None
    return len(set(ans)) > 1


def modal(runs: list[Obs]) -> tuple[str | None, int, int]:
    """(plurality answer or None on a tie, its count, number of valid runs)."""
    ans = valid_answers(runs)
    if not ans:
        return None, 0, 0
    counts = Counter(ans).most_common()
    top, k = counts[0]
    if len(counts) > 1 and counts[1][1] == k:
        return None, k, len(ans)
    return top, k, len(ans)


def agreement(runs: list[Obs]) -> float:
    _, k, n = modal(runs)
    return k / n if n else 0.0


# --- accuracy and confidence --------------------------------------------------
def run_accuracy(questions: dict[str, list[Obs]], labels: dict[str, str]) -> tuple[int, int]:
    k = n = 0
    for qid, runs in questions.items():
        for o in runs:
            if o.valid:
                n += 1
                k += o.answer == labels[qid]
    return k, n


def majority_correct(runs: list[Obs], label: str) -> bool | None:
    """Plurality answer vs. label. A tie counts as incorrect; no valid runs -> None."""
    top, _, n = modal(runs)
    if n == 0:
        return None
    return top == label


def majority_accuracy(questions: dict[str, list[Obs]], labels: dict[str, str]) -> tuple[int, int]:
    res = [majority_correct(r, labels[q]) for q, r in questions.items()]
    res = [x for x in res if x is not None]
    return sum(res), len(res)


def confident_wrong(questions: dict[str, list[Obs]], labels: dict[str, str],
                    threshold: float = config.CONFIDENT_THRESHOLD) -> dict[str, int]:
    """Counts of valid runs, wrong runs, and wrong runs with confidence >= threshold."""
    valid = wrong = cw = 0
    for qid, runs in questions.items():
        for o in runs:
            if not o.valid:
                continue
            valid += 1
            if o.answer != labels[qid]:
                wrong += 1
                cw += o.confidence is not None and o.confidence >= threshold
    return {"valid": valid, "wrong": wrong, "confident_wrong": cw}


def missing_counts(questions: dict[str, list[Obs]], planned_runs: int = config.RUNS_PER_ITEM,
                   planned_questions: int | None = None) -> dict[str, int]:
    c = Counter(o.status for runs in questions.values() for o in runs)
    planned = planned_runs * (planned_questions if planned_questions is not None else len(questions))
    observed = sum(c.values())
    return {"planned": planned, "valid": c["ok"], "unparseable": c["unparseable"],
            "no_response": c["no_response"] + (planned - observed)}


# --- uncertainty ------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """(proportion, lower, upper) Wilson score interval. n=0 -> (nan, nan, nan)."""
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_paired_diff(a, b, n_boot: int = 10_000, seed: int = config.SEED,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of (a - b) over paired units with a percentile bootstrap CI,
    resampling units (questions) with replacement."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("need equal-length, non-empty paired arrays")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_exact(a, b) -> dict:
    """Exact McNemar test on paired binary outcomes (two-sided binomial on discordant pairs)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    p = 1.0 if n == 0 else float(stats.binomtest(n10, n, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p_value": p}
