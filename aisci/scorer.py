"""Ballast scorer (exploratory; DEVIATIONS.md section B).

A small logistic-regression model, learned from the study's main-sample responses, that
predicts how likely the answer Ballast would give (the majority of 5 runs) is WRONG, from the
signals Ballast already collects: run agreement, the quote check, whether the model held its
answer under pushback, and its stated confidence. It knows nothing about biomedicine; it only
reads how an answer behaved.

Evaluation: 5-fold cross-validation grouped by question (all of a question's rows sit in one
fold, so the scorer is always tested on questions it never saw), plus leave-one-model-out, plus
DeepSeek as a model never used in training. Baselines: agreement alone, confidence alone.

    python -m aisci.scorer      ->  results/analysis/scorer.json (and demo/scorer_weights.json)
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import minimize

from . import config, data
from .metrics import modal, quote_found
from .providers import adapter_class, load_study_models
from .results import load_observations, load_pushbacks

TRAIN_KEYS = ["claude", "gemini", "groq", "qwen_groq"]   # the four preregistered families
HOLDOUT_KEYS = ["deepseek_kiro"]                        # never used for training
# The product scorer uses only signals that mean the same thing for any question.
FEATURES = ["disagreement", "tie", "pushback_changed", "pushback_missing", "quote_failed",
            "with_abstract", "low_confidence"]
# "answer_maybe" is reported for comparison only: it mostly encodes PubMedQA's label mix
# (11% "maybe") and would not carry over to a scientist's own questions.
ALL_FEATURES = FEATURES + ["answer_maybe"]
L2 = 1.0
N_FOLDS = 5


def features_for(runs, pair, ctx: str, abstract: str) -> dict:
    """The signals Ballast collects for one question; identical rules to the gate."""
    top, k, n = modal(runs)
    valid = [o for o in runs if o.valid]
    quote_failed = 0.0
    if ctx == "context" and valid:
        quote_failed = float(any(not o.quotes or not all(quote_found(s, abstract) for s in o.quotes)
                                 for o in valid))
    conf = np.mean([o.confidence for o in valid]) / 100 if valid else 0.0
    pb_valid = pair is not None and pair.valid
    return {
        "disagreement": 1 - (k / n if n else 0.0),
        "tie": float(n > 0 and top is None),
        "pushback_changed": float(pb_valid and pair.changed),
        "pushback_missing": float(not pb_valid),
        "quote_failed": quote_failed,
        "with_abstract": float(ctx == "context"),
        "low_confidence": 1 - float(conf),
        "answer_maybe": float(top == "maybe"),
    }


def build_rows(keys: list[str]) -> list[dict]:
    items = data.load()
    qids = data.load_sample("main")
    spec = load_study_models()["models"]
    obs, pbs = load_observations("main"), load_pushbacks("main")
    rows = []
    for key in keys:
        sid = f"{adapter_class(key, spec[key]).family}:{spec[key]['model']}"
        for ctx in config.CONTEXTS:
            runs_by_q, pairs = obs.get((sid, f"{ctx}_default"), {}), pbs.get((sid, ctx), {})
            for q in qids:
                runs = runs_by_q.get(q, [])
                top, _, n = modal(runs)
                if n == 0:
                    continue  # nothing to score
                f = features_for(runs, pairs.get(q), ctx, items[q].abstract)
                rows.append({"model": key, "ctx": ctx, "qid": q, "x": [f[k] for k in ALL_FEATURES],
                             "y": int(top != items[q].label)})   # tie counts as wrong
    return rows


# --- logistic regression (standardized, L2) ------------------------------------------
def fit(X: np.ndarray, y: np.ndarray, l2: float = L2) -> dict:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    def loss(w):
        z = Z @ w[1:] + w[0]
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[1:] ** 2) / 2

    def grad(w):
        p = 1 / (1 + np.exp(-(Z @ w[1:] + w[0])))
        g = np.empty_like(w)
        g[0] = np.sum(p - y)
        g[1:] = Z.T @ (p - y) + l2 * w[1:]
        return g

    w = minimize(loss, np.zeros(X.shape[1] + 1), jac=grad, method="L-BFGS-B").x
    return {"bias": float(w[0]), "weights": w[1:].tolist(), "mean": mu.tolist(), "std": sd.tolist()}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(model["mean"])) / np.array(model["std"])
    return 1 / (1 + np.exp(-(Z @ np.array(model["weights"]) + model["bias"])))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Probability a random wrong answer scores higher than a random right one (ties = 1/2)."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def grouped_folds(qids: list[str], k: int = N_FOLDS, seed: int = config.SEED) -> dict[str, int]:
    uniq = sorted(set(qids), key=int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    return {uniq[i]: int(j % k) for j, i in enumerate(order)}


PRODUCT_COLS = list(range(len(FEATURES)))


def matrix(rows, cols=PRODUCT_COLS) -> np.ndarray:
    return np.array([r["x"] for r in rows])[:, cols]


def cross_validated(rows, cols=PRODUCT_COLS) -> np.ndarray:
    """Out-of-fold risk for every row; folds are grouped by question."""
    X = matrix(rows, cols)
    y = np.array([r["y"] for r in rows])
    fold = grouped_folds([r["qid"] for r in rows])
    f = np.array([fold[r["qid"]] for r in rows])
    out = np.empty(len(rows))
    for k in range(N_FOLDS):
        m = fit(X[f != k], y[f != k])
        out[f == k] = predict(m, X[f == k])
    return out


def bootstrap_auc_ci(rows, p, n_boot=2000, seed=config.SEED):
    """Cluster bootstrap over questions for the AUROC."""
    y = np.array([r["y"] for r in rows])
    by_q: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["qid"], []).append(i)
    qs = list(by_q)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by_q[qs[j]] for j in rng.integers(0, len(qs), len(qs))])
        vals.append(auroc(y[idx], p[idx]))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate() -> dict:
    train = build_rows(TRAIN_KEYS)
    y = np.array([r["y"] for r in train])
    res = {"features": FEATURES, "rows": len(train), "wrong_rate": float(y.mean()),
           "note": "Exploratory, added after the preregistration. Predicts whether the majority "
                   "answer is wrong from Ballast's signals only."}
    p_full = cross_validated(train)
    p_maybe = cross_validated(train, list(range(len(ALL_FEATURES))))
    i_dis, i_conf = FEATURES.index("disagreement"), FEATURES.index("low_confidence")
    p_agree = cross_validated(train, [i_dis])
    p_conf = cross_validated(train, [i_conf])
    res["cv_grouped_by_question"] = {
        "scorer": {"auroc": auroc(y, p_full), "auroc_ci": bootstrap_auc_ci(train, p_full), "brier": brier(y, p_full)},
        "agreement_only": {"auroc": auroc(y, p_agree), "auroc_ci": bootstrap_auc_ci(train, p_agree), "brier": brier(y, p_agree)},
        "confidence_only": {"auroc": auroc(y, p_conf), "auroc_ci": bootstrap_auc_ci(train, p_conf), "brier": brier(y, p_conf)},
        "with_answer_maybe_feature": {"auroc": auroc(y, p_maybe), "auroc_ci": bootstrap_auc_ci(train, p_maybe),
                                      "brier": brier(y, p_maybe), "note": "PubMedQA-specific; not used in the product"},
        "base_rate_brier": brier(y, np.full(len(y), y.mean())),
    }
    # calibration: out-of-fold risk in 4 bins
    bins = np.quantile(p_full, [0, 0.25, 0.5, 0.75, 1.0])
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_full >= lo) & (p_full <= hi)
        cal.append({"risk_from": float(lo), "risk_to": float(hi), "n": int(m.sum()),
                    "mean_predicted": float(p_full[m].mean()), "observed_wrong": float(y[m].mean())})
    res["calibration_quartiles"] = cal
    # leave one model out: train on three families, test on the fourth
    lomo = {}
    for held in TRAIN_KEYS:
        tr = [r for r in train if r["model"] != held]
        te = [r for r in train if r["model"] == held]
        m = fit(matrix(tr), np.array([r["y"] for r in tr]))
        pt = predict(m, matrix(te))
        lomo[held] = {"auroc": auroc(np.array([r["y"] for r in te]), pt), "n": len(te)}
    res["leave_one_model_out"] = lomo
    # final model on all four families; tested once on DeepSeek, never trained on it
    final = fit(matrix(train), y)
    hold = build_rows(HOLDOUT_KEYS)
    ph = predict(final, matrix(hold))
    yh = np.array([r["y"] for r in hold])
    res["deepseek_holdout"] = {"auroc": auroc(yh, ph), "auroc_ci": bootstrap_auc_ci(hold, ph),
                               "brier": brier(yh, ph), "n": len(hold)}
    res["final_model"] = {**final, "features": FEATURES}
    return res


def main():
    res = evaluate()
    out = config.ROOT / "results" / "analysis" / "scorer.json"
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (config.ROOT / "demo").mkdir(exist_ok=True)
    (config.ROOT / "demo" / "scorer_weights.json").write_text(json.dumps(res["final_model"], indent=1) + "\n",
                                                             encoding="utf-8")
    cv = res["cv_grouped_by_question"]
    print(f"rows {res['rows']}, wrong rate {res['wrong_rate']:.3f}")
    for k in ("scorer", "agreement_only", "confidence_only", "with_answer_maybe_feature"):
        print(f"  {k:16s} AUROC {cv[k]['auroc']:.3f} [{cv[k]['auroc_ci'][0]:.3f}, {cv[k]['auroc_ci'][1]:.3f}]  Brier {cv[k]['brier']:.3f}")
    print(f"  base-rate Brier {cv['base_rate_brier']:.3f}")
    print("  leave-one-model-out:", {k: round(v["auroc"], 3) for k, v in res["leave_one_model_out"].items()})
    d = res["deepseek_holdout"]
    print(f"  DeepSeek (never trained on) AUROC {d['auroc']:.3f} [{d['auroc_ci'][0]:.3f}, {d['auroc_ci'][1]:.3f}] n={d['n']}")
    print("  calibration:", [(round(c["mean_predicted"], 2), round(c["observed_wrong"], 2)) for c in res["calibration_quartiles"]])
    print("  weights:", dict(zip(FEATURES, [round(w, 2) for w in res["final_model"]["weights"]])))


if __name__ == "__main__":
    main()
