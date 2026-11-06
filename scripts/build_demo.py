"""Build demo/ballast.html: the template with the gate logic, the exported study data and the
summary numbers inlined, so the page is one self-contained file.

    python scripts/export_demo_data.py && python scripts/build_demo.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"


def summary_subset() -> dict:
    s = json.loads((ROOT / "results" / "analysis" / "summary.json").read_text(encoding="utf-8"))
    pick = lambda c: {k: c[k] for k in ("est", "lo", "hi")}
    out = {}
    for key, m in s["models"].items():
        c = m["conditions"]
        out[key] = {
            "flip": {ctx: pick(c[f"{ctx}_default"]["flip_rate"]) for ctx in ("closed", "context")},
            "cap": {ctx: pick(m["pushback"][ctx]["capitulation_rate"]) for ctx in ("closed", "context")},
            "conf": {ctx: pick(c[f"{ctx}_default"]["confident_wrong_rate"]) for ctx in ("closed", "context")},
            "quotes": pick(c["context_default"]["quote_existence"]),
        }
    return out


def main():
    html = (DEMO / "template.html").read_text(encoding="utf-8")
    core = (DEMO / "ballast_core.js").read_text(encoding="utf-8")
    data = (DEMO / "data.json").read_text(encoding="utf-8")
    summary = json.dumps(summary_subset(), separators=(",", ":"))
    for marker, value in (("/*__CORE__*/", core), ("/*__DATA__*/", data), ("/*__SUMMARY__*/", summary)):
        assert html.count(marker) == 1, marker
        html = html.replace(marker, value.replace("</script", "<\\/script"))
    out = DEMO / "ballast.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
