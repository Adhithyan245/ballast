import pytest

from aisci.gate import coverage_curve, decide
from aisci.metrics import Obs

ABS = "The drug reduced mortality. Side effects were rare."


def o(a, quotes=None):
    return Obs(a, 90, quotes)


def test_decide_answers_when_unanimous():
    assert decide([o("yes")] * 5, 1.0).answer == "yes"


def test_decide_low_agreement_and_tie():
    rs = [o("yes")] * 3 + [o("no")] * 2
    assert decide(rs, 0.6).answer == "yes"
    assert decide(rs, 0.8).reason == "low_agreement"
    tie = [o("yes")] * 2 + [o("no")] * 2 + [o("maybe")]
    assert decide(tie, 0.0).reason == "tie"
    assert decide([Obs(None, status="unparseable")] * 5, 0.0).reason == "no_valid_runs"


def test_decide_quote_checks():
    good = o("yes", ("the drug reduced mortality.",))
    bad = o("yes", ("the drug cured everyone.",))
    empty = o("yes", ())
    assert decide([good] * 5, 1.0, ABS, check_quotes=True).answer == "yes"
    assert decide([good] * 4 + [bad], 0.8, ABS, check_quotes=True).reason == "quote_not_found"
    assert decide([good] * 4 + [empty], 0.8, ABS, check_quotes=True).reason == "no_quotes"
    # missing runs are skipped by the quote check
    assert decide([good] * 4 + [Obs(None, status="unparseable")], 1.0, ABS, True).answer == "yes"
    with pytest.raises(ValueError):
        decide([good], 0.0, None, check_quotes=True)


def test_coverage_curve_monotone_and_counts():
    qs = {
        "a": [o("yes")] * 5,                    # unanimous, correct
        "b": [o("no")] * 5,                     # unanimous, wrong
        "c": [o("yes")] * 3 + [o("no")] * 2,    # 0.6 agreement, correct
        "d": [o("no")] * 4 + [o("yes")],        # 0.8 agreement, correct
    }
    labels = {"a": "yes", "b": "yes", "c": "yes", "d": "no"}
    curve = coverage_curve(qs, labels, [0.0, 0.6, 0.8, 1.0])
    assert [r["answered"] for r in curve] == [4, 4, 3, 2]
    assert [r["wrong"] for r in curve] == [1, 1, 1, 1]
    assert curve[-1]["coverage"] == 0.5 and curve[-1]["accuracy_answered"] == 0.5
    assert curve[2]["reasons"] == {"answered": 3, "low_agreement": 1}
