"""Abstention gate: answer with the plurality answer only when run agreement
reaches a threshold and (with-context) every quote is found in the abstract."""
from __future__ import annotations

from dataclasses import dataclass

from .metrics import Obs, modal, quote_found


@dataclass(frozen=True)
class Decision:
    answer: str | None   # None = abstain
    reason: str          # "answered" | "no_valid_runs" | "tie" | "low_agreement" | "quote_not_found" | "no_quotes"


def decide(runs: list[Obs], threshold: float, abstract: str | None = None,
           check_quotes: bool = False) -> Decision:
    top, k, n = modal(runs)
    if n == 0:
        return Decision(None, "no_valid_runs")
    if top is None:
        return Decision(None, "tie")
    if k / n < threshold:
        return Decision(None, "low_agreement")
    if check_quotes:
        if abstract is None:
            raise ValueError("quote check needs the abstract")
        for o in runs:
            if not o.valid:
                continue
            if not o.quotes:
                return Decision(None, "no_quotes")
            if not all(quote_found(q, abstract) for q in o.quotes):
                return Decision(None, "quote_not_found")
    return Decision(top, "answered")


def coverage_curve(questions: dict[str, list[Obs]], labels: dict[str, str],
                   thresholds: list[float], abstracts: dict[str, str] | None = None,
                   check_quotes: bool = False) -> list[dict]:
    """For each threshold: coverage (share answered), accuracy on answered,
    and the number of wrong answers given (vs. answering everything)."""
    out = []
    n_q = len(questions)
    for t in thresholds:
        answered = correct = 0
        reasons: dict[str, int] = {}
        for qid, runs in questions.items():
            d = decide(runs, t, (abstracts or {}).get(qid), check_quotes)
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
            if d.answer is not None:
                answered += 1
                correct += d.answer == labels[qid]
        out.append({
            "threshold": t, "questions": n_q, "answered": answered,
            "correct": correct, "wrong": answered - correct,
            "coverage": answered / n_q if n_q else float("nan"),
            "accuracy_answered": correct / answered if answered else float("nan"),
            "reasons": reasons,
        })
    return out
