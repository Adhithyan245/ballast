import math

import numpy as np
import pytest

from aisci import scorer as S
from aisci.metrics import Obs, Pushback


def test_auroc_known_values():
    assert S.auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert S.auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert S.auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5      # ties count half
    assert math.isnan(S.auroc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_grouped_folds_keep_each_question_in_one_fold():
    qids = [str(q) for q in range(100, 140)] * 8      # every question appears 8 times
    folds = S.grouped_folds(qids)
    assert set(folds.values()) == set(range(S.N_FOLDS))
    assert len(folds) == 40                              # one fold per question, never split
    assert folds == S.grouped_folds(qids)                # seeded


def test_fit_recovers_signal_and_predicts_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + 0.2 * rng.normal(size=400) > 0).astype(int)   # only feature 0 matters
    m = S.fit(X, y)
    assert m["weights"][0] > 1.0 and abs(m["weights"][1]) < 0.5
    p = S.predict(m, X)
    assert ((p > 0) & (p < 1)).all() and S.auroc(y, p) > 0.95


def test_features_follow_the_gate_rules():
    abstract = "The drug worked. Side effects were rare."
    runs = [Obs("yes", 90, ("the drug worked.",))] * 4 + [Obs("no", 40, ("made up",))]
    pair = Pushback(Obs("yes", 90), Obs("no", 60), "no")
    f = S.features_for(runs, pair, "context", abstract)
    assert f["disagreement"] == pytest.approx(0.2)
    assert f["tie"] == 0.0 and f["quote_failed"] == 1.0 and f["with_abstract"] == 1.0
    assert f["pushback_changed"] == 1.0 and f["pushback_missing"] == 0.0
    assert f["low_confidence"] == pytest.approx(1 - 0.8)
    g = S.features_for([Obs("yes", 50)] * 2 + [Obs("no", 50)] * 2, None, "closed", abstract)
    assert g["tie"] == 1.0 and g["pushback_missing"] == 1.0 and g["quote_failed"] == 0.0


def test_product_scorer_excludes_the_pubmedqa_specific_feature():
    assert "answer_maybe" not in S.FEATURES and S.ALL_FEATURES[-1] == "answer_maybe"
