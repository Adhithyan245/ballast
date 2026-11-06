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
    # Standalone copy for GitHub Pages (docs/index.html): the same page inside a full document.
    head = "\n".join([
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="description" content="Ballast: a reliability check for AI research assistants. '
        'Interactive demo of a preregistered study of 8,998 LLM answers on PubMedQA.">',
        "<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);"
        "padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}"
        "[hidden]{display:none!important}</style>",
        "</head>",
        "<body>",
    ])
    page = head + "\n" + html + "\n</body>\n</html>\n"
    site = ROOT / "docs" / "index.html"
    site.parent.mkdir(exist_ok=True)
    site.write_text(page, encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {site.relative_to(ROOT)} for GitHub Pages")


if __name__ == "__main__":
    main()
