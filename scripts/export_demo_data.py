"""Export the main-study responses the interactive Ballast page needs, as compact JSON.

Per question: text, abstract, expert label. Per model x context: the 5 default-sampling runs
(answer, confidence, whether every quote was found, run-1 quotes for highlighting) and the
pushback pair. Everything comes from results/cache through the same parser and quote rule as
the analysis. Run: python scripts/export_demo_data.py  ->  demo/data.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aisci import data  # noqa: E402
from aisci.metrics import quote_found  # noqa: E402
from aisci.providers import adapter_class, load_study_models  # noqa: E402
from aisci.results import load_observations, load_pushbacks  # noqa: E402

ENTRIES = [("claude", "Claude", "claude-sonnet-5", False), ("gemini", "Gemini", "gemini-3.5-flash-lite", False),
           ("groq", "gpt-oss", "gpt-oss-120b", False), ("qwen_groq", "Qwen", "qwen3.8-27b", False),
           ("deepseek_kiro", "DeepSeek", "deepseek-3.2", True)]
CODE = {"yes": "y", "no": "n", "maybe": "m", None: None}


def main():
    items = data.load()
    qids = data.load_sample("main")
    spec = load_study_models()["models"]
    obs = load_observations("main")
    pbs = load_pushbacks("main")
    out = {"questions": [], "models": []}
    for q in qids:
        it = items[q]
        out["questions"].append({"id": q, "q": it.question, "label": CODE[it.label],
                                 "abstract": [[lab, txt] for lab, txt in zip(it.context_labels, it.contexts)]})
    for key, name, short, exploratory in ENTRIES:
        sid = f"{adapter_class(key, spec[key]).family}:{spec[key]['model']}"
        m = {"key": key, "name": name, "model": short, "exploratory": exploratory, "ctx": {}}
        for ctx in ("closed", "context"):
            runs_by_q = obs.get((sid, f"{ctx}_default"), {})
            pairs = pbs.get((sid, ctx), {})
            per_q = []
            for q in qids:
                abstract = items[q].abstract
                runs = []
                for o in runs_by_q.get(q, []):
                    if not o.valid:
                        runs.append(None)
                        continue
                    r = [CODE[o.answer], round(o.confidence)]
                    if ctx == "context":
                        r.append(1 if o.quotes and all(quote_found(s, abstract) for s in o.quotes) else 0)
                    runs.append(r)
                p = pairs.get(q)
                pb = None
                if p is not None:
                    pb = {"s": CODE[p.suggested], "i": CODE[p.initial.answer],
                          "f": CODE[p.final.answer] if p.final is not None else None,
                          "fc": round(p.final.confidence) if p.final is not None and p.final.valid else None}
                entry = {"r": runs, "p": pb}
                if ctx == "context":
                    first = next((o for o in runs_by_q.get(q, []) if o.valid), None)
                    entry["qt"] = [[s, 1 if quote_found(s, abstract) else 0] for s in (first.quotes or ())] if first else []
                per_q.append(entry)
            m["ctx"][ctx] = per_q
        out["models"].append(m)
    dest = ROOT / "demo" / "data.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
