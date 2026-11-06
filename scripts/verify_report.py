"""Check that the numbers quoted in REPORT.md match results/analysis/summary.json.

Each check formats a value from the results file the way the report writes it and asserts that
exact string appears in REPORT.md. Run: python scripts/verify_report.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
res = json.loads((ROOT / "results" / "analysis" / "summary.json").read_text(encoding="utf-8"))
M = res["models"]
order = ["claude", "gemini", "groq", "qwen_groq", "deepseek_kiro"]
primary = order[:4]


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def pci(c):
    return f"{pct(c['est'])} [{c['lo'] * 100:.1f}, {c['hi'] * 100:.1f}]"


checks = []


def need(s, what):
    checks.append((s in report, s, what))


for k in order:
    m, c = M[k], M[k]["conditions"]
    for ctx in ("closed", "context"):
        need(pci(c[f"{ctx}_default"]["flip_rate"]), f"{k} {ctx} flip")
        need(pct(c[f"{ctx}_default"]["majority_accuracy"]["est"]), f"{k} {ctx} majority acc")
        need(pci(c[f"{ctx}_default"]["confident_wrong_rate"]), f"{k} {ctx} confident-wrong")
        p = m["pushback"][ctx]
        need(pci(p["capitulation_rate"]), f"{k} {ctx} capitulation")
        rw = p["right_to_wrong"]
        need(f"{rw['k']}/{rw['n']}", f"{k} {ctx} right->wrong counts")
        g = m["gates"][ctx]
        cg, cb = g["consistency_gate"], g["combined_gate"]
        need(f"{cg['coverage']['est'] * 100:.0f}%, {pct(cg['accuracy_answered']['est'])}", f"{k} {ctx} consistency gate")
        need(f"{cb['coverage']['est'] * 100:.0f}%, {pct(cb['accuracy_answered']['est'])}", f"{k} {ctx} combined gate")
        um = g["ungated_majority"]
        need(f"{um['correct']} / {um['wrong']}", f"{k} {ctx} ungated majority")
    q = c["context_default"]["quote_existence"]
    need(f"{pct(q['est'])} ({q['k']}/{q['n']})", f"{k} quote existence")
    wr = M[k]["pushback"]["closed"]["wrong_to_right"]
    need(f"{wr['k']}/{wr['n']}", f"{k} closed wrong->right counts")
    if k in primary or k == "deepseek_kiro":
        x = M[k]["exploratory_matched_coverage"]
        for ctx in ("closed", "context"):
            if k == "deepseek_kiro":
                continue
            e = x[ctx]
            need(f"{pct(e['combined_accuracy'])} | {pct(e['consistency_matched_accuracy'])} | {pct(e['random_abstention_accuracy'])}",
                 f"{k} {ctx} matched coverage")

rows = {(r["model"], r["hyp"], r["context"]): r for r in res["hypotheses"]["rows"]}
for key, ctx, txt in [("claude", "closed", "p_holm = 0.015"), ("claude", "context", "p_holm = 0.008"),
                      ("qwen_groq", "context", "p_holm = 0.037")]:
    r = rows[(key, "H7", ctx)]
    need(txt, f"{key} {ctx} H7 holm")
    assert f"p_holm = {r['p_holm']:.3f}" == txt, (key, ctx, r["p_holm"])
need(f"p_holm = {rows[('qwen_groq', 'H1b', 'closed_vs_context')]['p_holm']:.3f}", "Qwen H1b holm")

failed = [c for c in checks if not c[0]]
for ok, s, what in checks:
    if not ok:
        print(f"MISSING  {what}: {s!r}")
print(f"{len(checks) - len(failed)}/{len(checks)} report numbers match results/analysis/summary.json")
sys.exit(1 if failed else 0)
