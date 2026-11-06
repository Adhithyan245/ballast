import math

import numpy as np
import pytest

from aisci import analysis as A
from aisci.metrics import Obs, Pushback


def test_holm_matches_hand_computation():
    # p sorted: .01 (x4=.04), .02 (x3=.06), .03 (x2=.06 by monotonicity), .5 (x1=.5)
    assert A.holm([0.03, 0.01, 0.5, 0.02]) == pytest.approx([0.06, 0.04, 0.5, 0.06])
    assert A.holm([0.9, 0.9]) == [1.0, 1.0]  # 0.9 x 2 capped at 1, then monotone
    assert A.holm([0.04]) == [0.04]
    assert A.holm([0.6, 0.7]) == [1.0, 1.0]  # capped at 1


def test_bootstrap_p():
    assert A.bootstrap_p(np.array([0.1, 0.2, 0.3])) == 0.0
    assert A.bootstrap_p(np.array([-1.0, 1.0])) == 1.0
    assert A.bootstrap_p(np.array([0.1, -0.1, 0.2, 0.3])) == pytest.approx(0.5)
    assert math.isnan(A.bootstrap_p(np.array([math.nan])))


def test_clean_handles_nan_and_numpy():
    out = A.clean({"a": float("nan"), "b": np.int64(3), "c": [np.float64(0.5), np.bool_(True)]})
    assert out == {"a": None, "b": 3, "c": [0.5, True]}


QIDS = ["1", "2", "3", "4"]
LABELS = {"1": "yes", "2": "yes", "3": "no", "4": "maybe"}
ABS = {q: "The drug worked. Side effects were rare." for q in QIDS}


def o(a, c=90, quotes=("the drug worked.",)):
    return Obs(a, c, quotes)


def test_summarize_counts():
    qs = {"1": [o("yes")] * 5,                                     # stable, right
          "2": [o("no", 85)] * 3 + [o("yes")] * 2,                 # flips; 3 confident-wrong runs
          "3": [o("no", quotes=("made up",))] * 5,                 # quotes not found
          "4": [o("yes", 50)] * 4 + [Obs(None, status="unparseable")]}  # 4 valid -> excluded from flip
    s = A.summarize(qs, QIDS, LABELS, ABS, "context")
    assert s["runs_valid"] == 19 and s["runs_unparseable"] == 1 and s["runs_no_response"] == 0
    assert (s["flip_rate"]["k"], s["flip_rate"]["n"]) == (1, 3)
    assert (s["flip_rate_min2"]["k"], s["flip_rate_min2"]["n"]) == (1, 4)
    assert (s["run_accuracy"]["k"], s["run_accuracy"]["n"]) == (5 + 2 + 5, 19)
    assert (s["majority_accuracy"]["k"], s["majority_accuracy"]["n"]) == (2, 4)
    assert (s["confident_wrong_rate"]["k"], s["confident_wrong_rate"]["n"]) == (3, 19)
    assert (s["quote_existence"]["k"], s["quote_existence"]["n"]) == (14, 19)
    assert (s["runs_all_quotes_found"]["k"], s["runs_all_quotes_found"]["n"]) == (14, 19)


def test_gate_tests_counts_and_direction():
    runs = {"1": [o("yes")] * 5, "2": [o("no")] * 3 + [o("yes")] * 2,
            "3": [o("no")] * 5, "4": [o("no")] * 5}
    pairs = {"1": Pushback(o("yes"), o("yes"), "no"),      # held
             "3": Pushback(o("no"), o("yes"), "yes"),      # capitulated
             "4": Pushback(o("no"), o("no"), "yes")}       # held
    g = A.gate_tests(runs, QIDS, LABELS, ABS, "closed", pairs, np.random.default_rng(0))
    # consistency gate (tau .8): answers 1 (right), 3 (right), 4 (wrong); 2 has 0.6 agreement
    assert (g["consistency_gate"]["answered"], g["consistency_gate"]["correct"]) == (3, 2)
    # combined gate drops 3 (changed under pushback)
    assert (g["combined_gate"]["answered"], g["combined_gate"]["correct"]) == (2, 1)
    # ungated majority: 1 right, 2 wrong (no), 3 right, 4 wrong
    assert (g["ungated_majority"]["correct"], g["ungated_majority"]["answered"]) == (2, 4)
    assert g["H4_diff"]["est"] == pytest.approx(2 / 3 - 2 / 4)
    assert g["H7_diff"]["est"] == pytest.approx(1 / 2 - 2 / 3)
    assert 0 <= g["H4_diff"]["p"] <= 1


def test_expected_top_n_accuracy_handles_ties():
    scores = np.array([1.0, 1.0, 0.8, 0.8, 0.8, 0.0])
    correct = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert A.expected_top_n_accuracy(scores, correct, 2) == pytest.approx(0.5)
    # top 3: both 1.0s (1 correct) + 1 of the three 0.8s (mean 2/3) -> (1 + 2/3) / 3
    assert A.expected_top_n_accuracy(scores, correct, 3) == pytest.approx((1 + 2 / 3) / 3)
    assert A.expected_top_n_accuracy(scores, correct, 6) == pytest.approx(4 / 6)
    assert math.isnan(A.expected_top_n_accuracy(scores, correct, 0))


def test_matched_coverage_counts():
    runs = {"1": [o("yes")] * 5, "2": [o("no")] * 3 + [o("yes")] * 2,
            "3": [o("no")] * 5, "4": [o("no")] * 5}
    pairs = {"1": Pushback(o("yes"), o("yes"), "no"), "3": Pushback(o("no"), o("yes"), "yes"),
             "4": Pushback(o("no"), o("no"), "yes")}
    m = A.matched_coverage(runs, QIDS, LABELS, ABS, "closed", pairs, np.random.default_rng(0))
    assert m["answered"] == 2 and m["combined_accuracy"] == pytest.approx(0.5)
    # consistency ranking at n=2: three questions tied at 1.0 (1 right, 3 right, 4 wrong) -> 2/3
    assert m["consistency_matched_accuracy"] == pytest.approx(2 / 3)
    assert m["random_abstention_accuracy"] == pytest.approx(0.5)
