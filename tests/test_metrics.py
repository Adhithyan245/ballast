import math

import numpy as np
import pytest

from aisci import metrics as M
from aisci.metrics import Obs

MISS = Obs(None, status="unparseable")


def runs(*answers, conf=90):
    return [Obs(a, conf) if a else MISS for a in answers]


def test_normalize_and_quote_found():
    abstract = "BACKGROUND: The   drug\nreduced mortality.\n\nRESULTS: n=40."
    assert M.quote_found("the drug reduced MORTALITY.", abstract)
    assert not M.quote_found("the drug increased mortality.", abstract)
    assert not M.quote_found("   ", abstract)  # empty quote never counts as found


def test_quote_existence_counts_only_valid_runs():
    abstract = "Alpha beta. Gamma delta."
    rs = [Obs("yes", 80, ("alpha beta.", "made up")), Obs("yes", 80, ("Gamma delta.",)),
          Obs(None, status="unparseable")]
    assert M.quote_existence(rs, abstract) == (2, 3)


def test_flipped():
    assert M.flipped(runs("yes", "yes", "yes", "yes", "yes")) is False
    assert M.flipped(runs("yes", "yes", "no", "yes", "yes")) is True
    assert M.flipped(runs("yes", "yes", "yes", "yes", None)) is None  # needs 5 valid by default
    assert M.flipped(runs("yes", "no", None, None, None), min_valid=2) is True


def test_modal_and_ties():
    assert M.modal(runs("yes", "yes", "no", "maybe", "yes")) == ("yes", 3, 5)
    assert M.modal(runs("yes", "yes", "no", "no", "maybe")) == (None, 2, 5)
    assert M.modal(runs(None, None)) == (None, 0, 0)
    assert M.agreement(runs("yes", "yes", "no", None, None)) == pytest.approx(2 / 3)


def test_run_and_majority_accuracy():
    qs = {"1": runs("yes", "yes", "no", "yes", "yes"),   # majority yes, label yes
          "2": runs("no", "no", "yes", "yes", None),     # tie -> incorrect
          "3": runs(None, None, None, None, None)}       # no valid runs -> excluded
    labels = {"1": "yes", "2": "no", "3": "yes"}
    assert M.run_accuracy(qs, labels) == (4 + 2, 5 + 4)
    assert M.majority_accuracy(qs, labels) == (1, 2)


def test_confident_wrong_threshold_is_inclusive():
    qs = {"1": [Obs("no", 80), Obs("no", 79.9), Obs("yes", 99), MISS]}
    assert M.confident_wrong(qs, {"1": "yes"}) == {"valid": 3, "wrong": 2, "confident_wrong": 1}


def test_missing_counts_include_never_returned_calls():
    qs = {"1": [Obs("yes", 50), MISS, Obs(None, status="no_response")]}
    m = M.missing_counts(qs, planned_runs=5, planned_questions=2)
    assert m == {"planned": 10, "valid": 1, "unparseable": 1, "no_response": 8}


def test_wilson_known_values():
    p, lo, hi = M.wilson(8, 10)
    assert p == 0.8
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert M.wilson(0, 10)[1] == 0.0
    assert all(math.isnan(x) for x in M.wilson(0, 0))


def test_bootstrap_paired_diff():
    a = np.ones(50)
    b = np.zeros(50)
    assert M.bootstrap_paired_diff(a, b) == (1.0, 1.0, 1.0)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 200)
    mean, lo, hi = M.bootstrap_paired_diff(x, x)
    assert mean == lo == hi == 0.0
    with pytest.raises(ValueError):
        M.bootstrap_paired_diff([1, 2], [1])


def test_bootstrap_is_seeded():
    rng = np.random.default_rng(1)
    a, b = rng.random(30), rng.random(30)
    assert M.bootstrap_paired_diff(a, b) == M.bootstrap_paired_diff(a, b)


def test_mcnemar_exact():
    a = [True] * 8 + [False] * 2 + [True] * 10
    b = [False] * 8 + [True] * 2 + [True] * 10
    r = M.mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (8, 2)
    assert r["p_value"] == pytest.approx(0.109375)  # 2 * P(X<=2 | n=10, p=.5)
    assert M.mcnemar_exact([True], [True])["p_value"] == 1.0
