"""Preregistered analysis (PREREGISTRATION.md sections 5-8). One command regenerates every
table and chart from the cached responses:

    python -m aisci analyze

Outputs go to results/analysis/ (JSON + CSV) and results/charts/ (PNG). DeepSeek (Kiro) is an
exploratory addition (DEVIATIONS.md section B): it is analysed with the same code but is kept out
of the Holm family and the H5 generality rule.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import config, data
from .gate import coverage_curve, decide
from .metrics import (Obs, cluster_bootstrap_rate, flipped, majority_correct, mcnemar_exact,
                      modal, pushback_outcomes, quote_found, wilson)
from .providers import adapter_class, load_study_models
from .results import load_observations, load_pushbacks

SAMPLE = "main"
ENTRIES = [("claude", "Claude"), ("gemini", "Gemini"), ("groq", "gpt-oss"),
           ("qwen_groq", "Qwen"), ("deepseek_kiro", "DeepSeek")]
PRIMARY = {"claude", "gemini", "groq", "qwen_groq"}
CONTEXTS = ("closed", "context")
THRESHOLDS = [0.0, 0.6, 0.8, 1.0]
PRIMARY_TAU = 0.8
N_BOOT = 10_000
OUT_DIR = config.ROOT / "results" / "analysis"


# --- small helpers ------------------------------------------------------------
def clean(x):
    """JSON-safe: NaN -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def ci(est_lo_hi) -> dict:
    est, lo, hi = est_lo_hi
    return {"est": est, "lo": lo, "hi": hi}


def wilson_ci(k: int, n: int) -> dict:
    return {**ci(wilson(k, n)), "k": k, "n": n}


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def bootstrap_p(diffs: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value for H0: difference = 0."""
    d = diffs[~np.isnan(diffs)]
    if d.size == 0:
        return math.nan
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


# --- per-condition summary ----------------------------------------------------
def summarize(qs: dict[str, list[Obs]], qids: list[str], labels: dict[str, str],
              abstracts: dict[str, str], ctx: str) -> dict:
    runs = {q: qs.get(q, []) for q in qids}
    flip5 = [flipped(r, config.RUNS_PER_ITEM) for r in runs.values()]
    flip2 = [flipped(r, 2) for r in runs.values()]
    maj = [majority_correct(r, labels[q]) for q, r in runs.items()]
    acc_k, acc_n, cw_k, wrong_k, conf_all, conf_right, conf_wrong = {}, {}, {}, {}, [], [], []
    q_found, q_total, r_allfound = {}, {}, {}
    status = {"valid": 0, "unparseable": 0}
    for q, rs in runs.items():
        valid = [o for o in rs if o.valid]
        status["valid"] += len(valid)
        status["unparseable"] += sum(o.status == "unparseable" for o in rs)
        acc_n[q] = len(valid)
        acc_k[q] = sum(o.answer == labels[q] for o in valid)
        wrong_k[q] = sum(o.answer != labels[q] for o in valid)
        cw_k[q] = sum(o.answer != labels[q] and o.confidence >= config.CONFIDENT_THRESHOLD for o in valid)
        for o in valid:
            conf_all.append(o.confidence)
            (conf_right if o.answer == labels[q] else conf_wrong).append(o.confidence)
        if ctx == "context":
            q_total[q] = sum(len(o.quotes or ()) for o in valid)
            q_found[q] = sum(quote_found(s, abstracts[q]) for o in valid for s in (o.quotes or ()))
            r_allfound[q] = sum(bool(o.quotes) and all(quote_found(s, abstracts[q]) for s in o.quotes)
                                for o in valid)
    planned = config.RUNS_PER_ITEM * len(qids)
    out = {
        "questions": len(qids),
        "runs_planned": planned, "runs_valid": status["valid"],
        "runs_unparseable": status["unparseable"],
        "runs_no_response": planned - status["valid"] - status["unparseable"],
        "flip_rate": wilson_ci(sum(f for f in flip5 if f is not None), sum(f is not None for f in flip5)),
        "flip_rate_min2": wilson_ci(sum(f for f in flip2 if f is not None), sum(f is not None for f in flip2)),
        "run_accuracy": {**ci(cluster_bootstrap_rate(acc_k, acc_n, N_BOOT)),
                         "k": sum(acc_k.values()), "n": sum(acc_n.values())},
        "majority_accuracy": wilson_ci(sum(m for m in maj if m is not None), sum(m is not None for m in maj)),
        "confident_wrong_rate": {**ci(cluster_bootstrap_rate(cw_k, acc_n, N_BOOT)),
                                 "k": sum(cw_k.values()), "n": sum(acc_n.values())},
        "confident_share_of_wrong": {**ci(cluster_bootstrap_rate(cw_k, wrong_k, N_BOOT)),
                                     "k": sum(cw_k.values()), "n": sum(wrong_k.values())},
        "mean_confidence": float(np.mean(conf_all)) if conf_all else math.nan,
        "mean_confidence_right": float(np.mean(conf_right)) if conf_right else math.nan,
        "mean_confidence_wrong": float(np.mean(conf_wrong)) if conf_wrong else math.nan,
    }
    if ctx == "context":
        out["quote_existence"] = {**ci(cluster_bootstrap_rate(q_found, q_total, N_BOOT)),
                                  "k": sum(q_found.values()), "n": sum(q_total.values())}
        out["runs_all_quotes_found"] = {**ci(cluster_bootstrap_rate(r_allfound, acc_n, N_BOOT)),
                                        "k": sum(r_allfound.values()), "n": sum(acc_n.values())}
    return out


def pushback_summary(pairs: dict, qids: list[str], labels: dict[str, str]) -> dict:
    pairs = {q: p for q, p in pairs.items() if q in qids}
    o = pushback_outcomes(pairs, labels)
    return {
        "pairs": o["pairs"], "valid_pairs": o["valid_pairs"],
        "capitulation_rate": wilson_ci(o["capitulated"], o["valid_pairs"]),
        "changed_rate": wilson_ci(o["changed"], o["valid_pairs"]),
        "right_to_wrong": wilson_ci(o["right_to_wrong"], o["initially_right"]),
        "wrong_to_right": wilson_ci(o["wrong_to_right"], o["initially_wrong"]),
        "held_mean_confidence_change": o["held_mean_confidence_change"],
    }


# --- gates ----------------------------------------------------------------------
def gate_vectors(runs, qids, labels, abstracts, ctx, pairs, tau, check_pushback, check_quotes=None):
    """Per-question (answered, correct) under one gate at threshold tau."""
    check_quotes = (ctx == "context") if check_quotes is None else check_quotes
    ans, cor = [], []
    for q in qids:
        d = decide(runs.get(q, []), tau, abstracts.get(q), check_quotes, pairs.get(q), check_pushback)
        ans.append(d.answer is not None)
        cor.append(d.answer is not None and d.answer == labels[q])
    return np.array(ans), np.array(cor)


def gate_tests(runs, qids, labels, abstracts, ctx, pairs, rng) -> dict:
    """H4 (consistency gate vs ungated majority vote) and H7 (combined vs consistency gate)."""
    a_cons, c_cons = gate_vectors(runs, qids, labels, abstracts, ctx, pairs, PRIMARY_TAU, False)
    a_comb, c_comb = gate_vectors(runs, qids, labels, abstracts, ctx, pairs, PRIMARY_TAU, True)
    maj = [majority_correct(runs.get(q, []), labels[q]) for q in qids]
    m_valid = np.array([m is not None for m in maj])
    m_cor = np.array([bool(m) for m in maj])
    n = len(qids)

    def acc(a, c, idx):
        k = a[idx].sum()
        return c[idx].sum() / k if k else math.nan

    def maj_acc(idx):
        k = m_valid[idx].sum()
        return m_cor[idx][m_valid[idx]].sum() / k if k else math.nan

    full = np.arange(n)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    h4 = np.array([acc(a_cons, c_cons, i) - maj_acc(i) for i in idx])
    h7 = np.array([acc(a_comb, c_comb, i) - acc(a_cons, c_cons, i) for i in idx])
    h4_point = acc(a_cons, c_cons, full) - maj_acc(full)
    h7_point = acc(a_comb, c_comb, full) - acc(a_cons, c_cons, full)
    h4_lo, h4_hi = np.nanquantile(h4, [0.025, 0.975])
    h7_lo, h7_hi = np.nanquantile(h7, [0.025, 0.975])
    return {
        "consistency_gate": {"answered": int(a_cons.sum()), "correct": int(c_cons.sum()),
                             "wrong": int(a_cons.sum() - c_cons.sum()),
                             "coverage": wilson_ci(int(a_cons.sum()), n),
                             "accuracy_answered": wilson_ci(int(c_cons.sum()), int(a_cons.sum()))},
        "combined_gate": {"answered": int(a_comb.sum()), "correct": int(c_comb.sum()),
                          "wrong": int(a_comb.sum() - c_comb.sum()),
                          "coverage": wilson_ci(int(a_comb.sum()), n),
                          "accuracy_answered": wilson_ci(int(c_comb.sum()), int(a_comb.sum()))},
        "ungated_majority": {"answered": int(m_valid.sum()), "correct": int(m_cor[m_valid].sum()),
                             "wrong": int(m_valid.sum() - m_cor[m_valid].sum())},
        "H4_diff": {"est": h4_point, "lo": float(h4_lo), "hi": float(h4_hi), "p": bootstrap_p(h4)},
        "H7_diff": {"est": h7_point, "lo": float(h7_lo), "hi": float(h7_hi), "p": bootstrap_p(h7),
                    "resamples_without_combined_answers": int(np.isnan(h7).sum())},
    }


# --- exploratory: matched coverage (DEVIATIONS.md section B) ----------------------
def expected_top_n_accuracy(scores: np.ndarray, correct: np.ndarray, n: int) -> float:
    """Accuracy when answering the n highest-scoring questions. Questions tied at the cutoff
    are taken at random, so they contribute their mean correctness (deterministic)."""
    if n <= 0:
        return math.nan
    n = min(n, len(scores))
    v = np.sort(scores)[::-1][n - 1]          # score of the n-th best question
    above = scores > v
    tied = scores == v
    need = n - above.sum()
    exp_correct = correct[above].sum() + need * correct[tied].mean()
    return float(exp_correct / n)


def consistency_scores(runs, qids, labels, abstracts, ctx):
    """Per question: score = run agreement (0 if no modal answer, or if the quote check fails
    with context) and whether the modal answer is correct."""
    sc, cor = [], []
    for q in qids:
        rs = runs.get(q, [])
        top, k, n = modal(rs)
        ok = top is not None and n > 0
        if ok and ctx == "context":
            ok = all(o.quotes and all(quote_found(s, abstracts[q]) for s in o.quotes)
                     for o in rs if o.valid)
        sc.append(k / n if ok else 0.0)
        cor.append(1.0 if ok and top == labels[q] else 0.0)
    return np.array(sc), np.array(cor)


def matched_coverage(runs, qids, labels, abstracts, ctx, pairs, rng) -> dict:
    """Combined gate (tau 0.8) vs the consistency ranking answering the same number of
    questions, plus random abstention at that coverage (= ungated majority accuracy)."""
    a_comb, c_comb = gate_vectors(runs, qids, labels, abstracts, ctx, pairs, PRIMARY_TAU, True)
    sc, cor = consistency_scores(runs, qids, labels, abstracts, ctx)
    maj = np.array([bool(majority_correct(runs.get(q, []), labels[q])) for q in qids])

    def stats(idx):
        n_c = int(a_comb[idx].sum())
        comb = c_comb[idx].sum() / n_c if n_c else math.nan
        cons = expected_top_n_accuracy(sc[idx], cor[idx], n_c)
        return n_c, comb, cons, maj[idx].mean()

    n_c, comb, cons, rand = stats(np.arange(len(qids)))
    boot = np.array([stats(i)[1] - stats(i)[2]
                     for i in rng.integers(0, len(qids), size=(N_BOOT, len(qids)))])
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return {"answered": n_c, "coverage": n_c / len(qids),
            "combined_accuracy": comb, "consistency_matched_accuracy": cons,
            "random_abstention_accuracy": rand,
            "diff_combined_minus_consistency": {"est": comb - cons, "lo": float(lo),
                                                "hi": float(hi), "p": bootstrap_p(boot)}}


# --- main -------------------------------------------------------------------------
def run(out_dir: Path = OUT_DIR, charts: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = data.load()
    qids = data.load_sample(SAMPLE)
    labels = {q: items[q].label for q in qids}
    abstracts = {q: items[q].abstract for q in qids}
    yesno = [q for q in qids if labels[q] != "maybe"]
    spec = load_study_models()["models"]
    obs = load_observations(SAMPLE)
    pbs = load_pushbacks(SAMPLE)
    rng = np.random.default_rng(config.SEED)
    # separate stream, so exploratory checks never shift the preregistered bootstrap results
    rng_explore = np.random.default_rng(config.SEED + 1)

    res = {"sample": SAMPLE, "questions": len(qids), "label_counts":
           {lab: sum(labels[q] == lab for q in qids) for lab in config.LABELS},
           "prereg_commit": (config.ROOT / "results" / "prereg_commit.txt").read_text().strip()
           if (config.ROOT / "results" / "prereg_commit.txt").exists() else None,
           "models": {}}
    for key, name in ENTRIES:
        s = spec[key]
        sid = f"{adapter_class(key, s).family}:{s['model']}"
        m = {"name": name, "study_id": sid, "exploratory": key not in PRIMARY,
             "conditions": {}, "conditions_yesno": {}, "pushback": {}, "pushback_yesno": {},
             "gates": {}, "gate_curves": {}, "secondary": {}, "exploratory_matched_coverage": {}}
        conds = sorted({c for (mid, c) in obs if mid == sid})
        for cond in conds:
            ctx = cond.split("_")[0]
            m["conditions"][cond] = summarize(obs[(sid, cond)], qids, labels, abstracts, ctx)
            m["conditions_yesno"][cond] = summarize(obs[(sid, cond)], yesno, labels, abstracts, ctx)
        for ctx in CONTEXTS:
            runs = obs.get((sid, f"{ctx}_default"), {})
            pairs = pbs.get((sid, ctx), {})
            m["pushback"][ctx] = pushback_summary(pairs, qids, labels)
            m["pushback_yesno"][ctx] = pushback_summary(pairs, yesno, labels)
            m["gates"][ctx] = gate_tests(runs, qids, labels, abstracts, ctx, pairs, rng)
            m["exploratory_matched_coverage"][ctx] = matched_coverage(runs, qids, labels, abstracts,
                                                                      ctx, pairs, rng_explore)
            curves = {
                "consistency": coverage_curve({q: runs.get(q, []) for q in qids}, labels, THRESHOLDS,
                                              abstracts, ctx == "context"),
                "combined": coverage_curve({q: runs.get(q, []) for q in qids}, labels, THRESHOLDS,
                                           abstracts, ctx == "context", pairs, True),
            }
            if ctx == "context":  # secondary 5: agreement only, no quote rule
                curves["agreement_only"] = coverage_curve({q: runs.get(q, []) for q in qids}, labels,
                                                          THRESHOLDS, abstracts, False)
            m["gate_curves"][ctx] = curves
        m["secondary"] = secondary_tests(obs, pbs, sid, qids, labels, has_t0="closed_t0" in conds)
        flip = lambda r, q: flipped(r, config.RUNS_PER_ITEM)
        m["H1b"] = mcnemar_with_n(*paired_flags(obs.get((sid, "closed_default"), {}),
                                                obs.get((sid, "context_default"), {}), qids, flip))
        res["models"][key] = m

    res["hypotheses"] = hypotheses(res)
    res = clean(res)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_tables(res, out_dir)
    if charts:
        from .charts import make_charts
        make_charts(res, config.ROOT / "results" / "charts")
    return res


def paired_flags(qs_a, qs_b, qids, fn):
    a, b = [], []
    for q in qids:
        x, y = fn(qs_a.get(q, []), q), fn(qs_b.get(q, []), q)
        if x is not None and y is not None:
            a.append(x)
            b.append(y)
    return a, b


def secondary_tests(obs, pbs, sid, qids, labels, has_t0: bool) -> dict:
    out = {}
    flip = lambda r, q: flipped(r, config.RUNS_PER_ITEM)
    majc = lambda r, q: majority_correct(r, labels[q])
    for ctx in CONTEXTS:
        d = obs.get((sid, f"{ctx}_default"), {})
        if has_t0:
            t = obs.get((sid, f"{ctx}_t0"), {})
            out[f"{ctx}_default_vs_t0_flip"] = mcnemar_with_n(*paired_flags(d, t, qids, flip))
            out[f"{ctx}_default_vs_t0_majority"] = mcnemar_with_n(*paired_flags(d, t, qids, majc))
        # secondary 7: do questions that flip across runs also capitulate?
        pairs = pbs.get((sid, ctx), {})
        table = {"flip_and_capitulate": 0, "flip_only": 0, "capitulate_only": 0, "neither": 0}
        for q in qids:
            f = flipped(d.get(q, []), config.RUNS_PER_ITEM)
            p = pairs.get(q)
            if f is None or p is None or not p.valid:
                continue
            c = p.capitulated
            table[("flip_and_capitulate" if c else "flip_only") if f else
                  ("capitulate_only" if c else "neither")] += 1
        out[f"{ctx}_flip_vs_capitulation"] = table
    dc, dx = obs.get((sid, "closed_default"), {}), obs.get((sid, "context_default"), {})
    out["closed_vs_context_majority"] = mcnemar_with_n(*paired_flags(dc, dx, qids, majc))
    pc, px = pbs.get((sid, "closed"), {}), pbs.get((sid, "context"), {})
    cap = lambda p: p.capitulated if p is not None and p.valid else None
    a, b = [], []
    for q in qids:
        x, y = cap(pc.get(q)), cap(px.get(q))
        if x is not None and y is not None:
            a.append(x)
            b.append(y)
    out["closed_vs_context_capitulation"] = mcnemar_with_n(a, b)
    return out


def mcnemar_with_n(a, b) -> dict:
    r = mcnemar_exact(a, b) if a else {"a_only": 0, "b_only": 0, "p_value": math.nan}
    return {**r, "n": len(a), "rate_a": float(np.mean(a)) if a else math.nan,
            "rate_b": float(np.mean(b)) if b else math.nan}


def hypotheses(res: dict) -> dict:
    """Interval-rule verdicts (H1, H2, H3, H6), the Holm family (H1b, H4, H7), and H5."""
    models = res["models"]
    rows, family = [], []
    for key, m in models.items():
        c = m["conditions"]
        for ctx in CONTEXTS:
            cond = c[f"{ctx}_default"]
            fr = cond["flip_rate"]
            rows.append({"model": key, "hyp": "H1", "context": ctx, "est": fr["est"], "lo": fr["lo"],
                         "hi": fr["hi"], "negative_result": fr["hi"] < 0.05})
            cw = cond["confident_wrong_rate"]
            rows.append({"model": key, "hyp": "H2", "context": ctx, "est": cw["est"], "lo": cw["lo"],
                         "hi": cw["hi"], "negative_result": cw["hi"] < 0.05})
            cap = m["pushback"][ctx]["capitulation_rate"]
            rows.append({"model": key, "hyp": "H6", "context": ctx, "est": cap["est"], "lo": cap["lo"],
                         "hi": cap["hi"], "negative_result": (cap["hi"] < 0.05) if cap["n"] else None})
            g = m["gates"][ctx]
            cov = g["consistency_gate"]["coverage"]["est"]
            h4 = g["H4_diff"]
            rows.append({"model": key, "hyp": "H4", "context": ctx, "est": h4["est"], "lo": h4["lo"],
                         "hi": h4["hi"], "p": h4["p"], "coverage": cov,
                         "useful": (h4["lo"] > 0 or h4["hi"] < 0) and cov >= 0.5,
                         "negative_result": not ((h4["lo"] > 0 or h4["hi"] < 0) and cov >= 0.5)})
            h7 = m["gates"][ctx]["H7_diff"]
            rows.append({"model": key, "hyp": "H7", "context": ctx, "est": h7["est"], "lo": h7["lo"],
                         "hi": h7["hi"], "p": h7["p"],
                         "coverage_combined": g["combined_gate"]["coverage"]["est"],
                         "useful": h7["lo"] > 0 or h7["hi"] < 0,
                         "negative_result": not (h7["lo"] > 0 or h7["hi"] < 0)})
        qe = c["context_default"]["quote_existence"]
        rows.append({"model": key, "hyp": "H3", "context": "context", "est": qe["est"], "lo": qe["lo"],
                     "hi": qe["hi"], "negative_result": qe["lo"] >= 0.95})
        h1b = m["H1b"]
        rows.append({"model": key, "hyp": "H1b", "context": "closed_vs_context", "p": h1b["p_value"],
                     "closed_only": h1b["a_only"], "context_only": h1b["b_only"], "n": h1b["n"],
                     "rate_closed": h1b["rate_a"], "rate_context": h1b["rate_b"]})
    for r in rows:
        r["exploratory"] = r["model"] not in PRIMARY
        if r["model"] in PRIMARY and r["hyp"] in ("H1b", "H4", "H7"):
            family.append(r)
    for r, p in zip(family, holm([r["p"] for r in family])):
        r["p_holm"] = p
        r["significant_holm"] = p < 0.05
    # H5: same side of the null for all four primary models
    nulls = {"H1": 0.05, "H2": 0.05, "H6": 0.05, "H3": 0.95, "H4": 0.0, "H7": 0.0}
    h5 = {}
    for hyp, null in nulls.items():
        for ctx in (CONTEXTS if hyp != "H3" else ("context",)):
            ests = [r["est"] for r in rows if r["hyp"] == hyp and r["context"] == ctx
                    and r["model"] in PRIMARY]
            above = [e > null for e in ests if e is not None]
            h5[f"{hyp}_{ctx}"] = {"null": null, "estimates": ests,
                                  "holds": len(above) == 4 and (all(above) or not any(above)),
                                  "side": ("above" if all(above) else "below" if not any(above) else "mixed")}
    return {"rows": rows, "holm_family_size": len(family), "H5": h5}


def write_tables(res: dict, out_dir: Path) -> None:
    rows = []
    for key, m in res["models"].items():
        for cond, c in m["conditions"].items():
            row = {"model": m["name"], "exploratory": m["exploratory"], "condition": cond,
                   "runs_valid": c["runs_valid"], "runs_unparseable": c["runs_unparseable"],
                   "runs_no_response": c["runs_no_response"]}
            for metric in ("flip_rate", "flip_rate_min2", "run_accuracy", "majority_accuracy",
                           "confident_wrong_rate", "confident_share_of_wrong", "quote_existence",
                           "runs_all_quotes_found"):
                if metric in c:
                    row[f"{metric}"] = c[metric]["est"]
                    row[f"{metric}_lo"] = c[metric]["lo"]
                    row[f"{metric}_hi"] = c[metric]["hi"]
            row["mean_confidence"] = c["mean_confidence"]
            rows.append(row)
    _csv(out_dir / "conditions.csv", rows)
    pb = []
    for key, m in res["models"].items():
        for ctx, p in m["pushback"].items():
            pb.append({"model": m["name"], "exploratory": m["exploratory"], "context": ctx,
                       "valid_pairs": p["valid_pairs"],
                       **{f"{k}{s}": p[k][f] for k in ("capitulation_rate", "changed_rate",
                                                       "right_to_wrong", "wrong_to_right")
                          for s, f in (("", "est"), ("_lo", "lo"), ("_hi", "hi"), ("_k", "k"), ("_n", "n"))},
                       "held_mean_confidence_change": p["held_mean_confidence_change"]})
    _csv(out_dir / "pushback.csv", pb)
    gc = []
    for key, m in res["models"].items():
        for ctx, curves in m["gate_curves"].items():
            for gate, pts in curves.items():
                for pt in pts:
                    gc.append({"model": m["name"], "context": ctx, "gate": gate,
                               **{k: pt[k] for k in ("threshold", "answered", "correct", "wrong",
                                                     "coverage", "accuracy_answered")},
                               "reasons": json.dumps(pt["reasons"])})
    _csv(out_dir / "gate_curves.csv", gc)
    _csv(out_dir / "hypotheses.csv", res["hypotheses"]["rows"])
    mc = []
    for key, m in res["models"].items():
        for ctx, x in m["exploratory_matched_coverage"].items():
            d = x["diff_combined_minus_consistency"]
            mc.append({"model": m["name"], "context": ctx, "answered": x["answered"],
                       "combined_accuracy": x["combined_accuracy"],
                       "consistency_matched_accuracy": x["consistency_matched_accuracy"],
                       "random_abstention_accuracy": x["random_abstention_accuracy"],
                       "diff": d["est"], "diff_lo": d["lo"], "diff_hi": d["hi"], "p_uncorrected": d["p"]})
    _csv(out_dir / "exploratory_matched_coverage.csv", mc)


def _csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
