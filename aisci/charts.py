"""Static PNG charts for REPORT.md, drawn from results/analysis/summary.json.

Palette: categorical slots 1-2 of the reference palette (blue #2a78d6, orange #eb6834),
validated for CVD separation on the light surface. Identity is never color-only: every chart
has a legend, and condition is also encoded by marker shape.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CTX_STYLE = {"closed": (BLUE, "o", "Closed-book"), "context": (ORANGE, "s", "With abstract")}
FOOT = "† DeepSeek is exploratory (added after the preregistration). Bars: 95% intervals."


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)


def _names(res):
    return [m["name"] + ("†" if m["exploratory"] else "") for m in res["models"].values()]


def _dot_chart(res, getter, title, ylabel, path, contexts=("closed", "context"), ylim=(0, 1)):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    keys = list(res["models"])
    offsets = {c: (i - (len(contexts) - 1) / 2) * 0.18 for i, c in enumerate(contexts)}
    for ctx in contexts:
        color, marker, label = CTX_STYLE[ctx]
        xs, ys, lo, hi = [], [], [], []
        for i, k in enumerate(keys):
            v = getter(res["models"][k], ctx)
            if v is None or v.get("est") is None:
                continue
            xs.append(i + offsets[ctx])
            ys.append(v["est"])
            lo.append(v["est"] - v["lo"])
            hi.append(v["hi"] - v["est"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=marker, color=color, ecolor=color, elinewidth=2,
                    capsize=0, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(_names(res), color=TEXT)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, title, ylabel)
    if len(contexts) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT2, loc="upper right")
    fig.text(0.01, 0.01, FOOT, color=TEXT2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _gate_curves(res, path):
    keys = list(res["models"])
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 6.2), dpi=160,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    gate_style = {"consistency": (BLUE, "o", "Consistency gate"),
                  "combined": (ORANGE, "s", "Combined gate (+ pushback)")}
    for row, ctx in enumerate(("closed", "context")):
        for col, k in enumerate(keys):
            ax = axes[row][col]
            m = res["models"][k]
            for gate, (color, marker, label) in gate_style.items():
                pts = [p for p in m["gate_curves"][ctx][gate] if p["accuracy_answered"] is not None]
                ax.plot([p["coverage"] for p in pts], [p["accuracy_answered"] for p in pts],
                        color=color, linewidth=2, marker=marker, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
                for p in pts:
                    if abs(p["threshold"] - 0.8) < 1e-9:
                        ax.plot(p["coverage"], p["accuracy_answered"], marker=marker, markersize=11,
                                markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.5)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0.3, 1.0)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            _style(ax, f"{m['name']}{'†' if m['exploratory'] else ''}" if row == 0 else "", "")
            if col == 0:
                ax.set_ylabel(("Closed-book" if ctx == "closed" else "With abstract")
                              + "\naccuracy on answered", color=TEXT2, fontsize=9)
            if row == 1:
                ax.set_xlabel("coverage (share answered)", color=TEXT2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT2, ncol=2)
    fig.suptitle("Abstention gates: coverage vs accuracy (thresholds 0, 0.6, 0.8, 1.0; ring = 0.8)",
                 x=0.01, ha="left", color=TEXT, fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, FOOT.split(" Bars")[0], color=TEXT2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def make_charts(res: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "flip": out_dir / "flip_rate.png",
        "pushback": out_dir / "capitulation.png",
        "confident": out_dir / "confident_wrong.png",
        "quotes": out_dir / "quote_existence.png",
        "gates": out_dir / "gate_curves.png",
    }
    _dot_chart(res, lambda m, c: m["conditions"][f"{c}_default"]["flip_rate"],
               "Flip rate: share of questions whose 5 answers were not all the same",
               "flip rate (default sampling)", paths["flip"])
    _dot_chart(res, lambda m, c: m["pushback"][c]["capitulation_rate"],
               "Capitulation: switched to the user's suggested answer after pushback",
               "capitulation rate", paths["pushback"])
    _dot_chart(res, lambda m, c: m["conditions"][f"{c}_default"]["confident_wrong_rate"],
               "Confident errors: wrong answers stated with confidence ≥ 80",
               "share of all valid answers", paths["confident"], ylim=(0, 0.6))
    _dot_chart(res, lambda m, c: m["conditions"]["context_default"].get("quote_existence"),
               "Quote existence: quotes found verbatim in the abstract (whitespace/case normalized)",
               "share of quotes found", paths["quotes"], contexts=("context",))
    _gate_curves(res, paths["gates"])
    return list(paths.values())
