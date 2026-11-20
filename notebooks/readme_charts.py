"""Generate the polished, dark, insight-driven charts for the README / RESULTS page.

Design (per the `data-visualization` skill):
- Title states the *insight*, not a label; subtitle carries context.
- Highlight the story: the model is a bright blue, baselines recede into slate.
- No grid / clean spines (chart-junk removal); direct value labels over legend.
- Sort by the size of the model's win (meaningful order); dim the few stats it loses.
- One amber callout on the single biggest win to anchor the eye.
- Generous whitespace; colorblind-safe blue (model) vs neutral (baselines).

Reads:  models/eval_report.json, models/holdout_score.json
Writes: docs/mae_vs_baselines.png, docs/holdout_2026.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# --- refined dark, editorial palette --------------------------------------
BG      = "#0a0f1e"   # near-black navy
MODEL   = "#4d9fff"   # bright confident blue — the hero
TR3     = "#3b4a68"   # muted slate (tr3 baseline)
LAST    = "#293249"   # darker slate (last baseline)
DIM     = "#222c42"   # losing stats — recede
AMBER   = "#f6b545"   # single callout accent (colorblind-safe vs blue)
INK     = "#f2f6fc"   # near-white (titles, key labels)
MUTED   = "#8b99b3"   # axis labels, secondary text
FAINT   = "#5b6a86"   # footnotes
LINE    = "#243149"   # hairline
plt.rcParams["figure.facecolor"] = BG
plt.rcParams["axes.facecolor"]   = BG
plt.rcParams["savefig.facecolor"]= BG
plt.rcParams["savefig.edgecolor"]= BG
plt.rcParams["text.color"]       = INK
plt.rcParams["axes.labelcolor"]  = MUTED
plt.rcParams["xtick.color"]      = MUTED
plt.rcParams["ytick.color"]      = MUTED
plt.rcParams["legend.labelcolor"] = MUTED


def _clean(ax, x_axis: bool = False, y_axis: bool = False):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if x_axis:
        ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(LINE)
    if y_axis:
        ax.spines["left"].set_visible(True);  ax.spines["left"].set_color(LINE)
    ax.grid(False)


def _head(fig, title, subtitle):
    fig.text(0.035, 0.965, title, fontsize=19, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.035, 0.915, subtitle, fontsize=10.5, color=MUTED, va="top")
    fig.text(0.965, 0.018, "Winning-NFL-Fantasy-Model  ·  clean holdout  ·  "
             "lower MAE / higher ρ = better", fontsize=7.5, color=FAINT,
             va="bottom", ha="right")


# ======================================================================
# Chart 1 — 2025 clean holdout: model vs both naive baselines
# ======================================================================
def chart_holdout(report: dict):
    res = report["results"]
    rows = []
    for key, r in res.items():
        m_mae = float(r["test"]["mae"])
        t_mae = float(r["baseline_test"]["trailing3"]["mae"])
        l_mae = float(r["baseline_test"]["last_game"]["mae"])
        best = min(t_mae, l_mae)
        rows.append({
            "position": r["position"], "stat": r["target"],
            "m_mae": m_mae, "t_mae": t_mae, "l_mae": l_mae,
            "m_rho": float(r.get("test_spearman") or 0.0),
            "beats": bool(r.get("beats_baseline", False)),
            "win": best - m_mae,                                  # >0 = model better
            "winpct": (best - m_mae) / best * 100 if best > 0 else 0.0,
        })

    # sort by size of the win (biggest first); dim the ones the model loses
    rows = sorted(rows, key=lambda r: r["win"], reverse=True)
    n = len(rows)
    labels = [f"{r['position']}  {r['stat']}" for r in rows]
    y = np.arange(n)[::-1]

    fig = plt.figure(figsize=(12.5, 8.2))
    _head(fig, f"Beats both naive baselines on {sum(1 for r in rows if r['beats'])} of {n} stats",
          "Clean 2025 holdout — the model was never fit on 2025.  "
          "MAE (lower is better), sorted by the size of the edge.")
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.985, top=0.85, bottom=0.10,
                          wspace=0.22, width_ratios=[1.35, 1.0])

    # ---- left: MAE comparison -------------------------------------------
    ax = fig.add_subplot(gs[0])
    h = 0.26
    off = {"Last year's same week": -h, "Trailing 3 (same week)": 0.0, "this model": +h}
    for name, col, key in (("Last year's same week", LAST, "l_mae"),
                           ("Trailing 3 (same week)", TR3, "t_mae"),
                           ("this model", MODEL, "m_mae")):
        vals = [r[key] for r in rows]
        dim = [not r["beats"] for r in rows]
        ax.barh([yy - off[name] for yy in y], vals, height=h*0.92,
                color=[DIM if d else col for d, _ in zip(dim, vals)],
                edgecolor="none", zorder=3)
        # direct labels on the model series only (precision, no legend clutter)
        if name == "this model":
            for i, (r, yy) in enumerate(zip(rows, y)):
                ax.text(r[key] + max(vals)*0.01, yy + h, f"{r[key]:.1f}",
                        va="center", ha="left", fontsize=8.2,
                        color=INK if r["beats"] else MUTED,
                        fontweight="bold" if r["beats"] else "normal")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.3)
    ax.set_xlim(0, max(r["l_mae"] for r in rows)*1.16)
    ax.set_xlabel("MAE (fantasy-stat units, lower = better)", fontsize=8.5)
    ax.xaxis.set_visible(False)
    _clean(ax, x_axis=True)
    # headline callout in the top headroom — no arrow crossing the bars
    ax.set_ylim(y[-1]-1.5, y[0]+2.5)
    top = rows[0]
    ax.text(ax.get_xlim()[1], y[0]+1.5,
            f"biggest edge    {top['position']} {top['stat'].replace('_',' ')}    −{top['winpct']:.0f}% MAE",
            ha="right", va="center", fontsize=9.5, color=AMBER, fontweight="bold")
    ax.text(ax.get_xlim()[0], y[-1]-0.7, "dimmed = model does not beat baseline (sparse TD/INT)",
            ha="left", va="top", fontsize=7.3, color=FAINT)

    # ---- right: ranking quality (Spearman) -------------------------------
    ax2 = fig.add_subplot(gs[1])
    rhos = [r["m_rho"] for r in rows]
    colors = [MODEL if r["beats"] else DIM for r in rows]
    ax2.barh(y, rhos, height=h*0.92, color=colors, zorder=3)
    ref = 0.5
    ax2.axvline(ref, color=AMBER, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax2.text(ref, y[0]+1.1, f"ρ = {ref:.2f}", fontsize=7.6, color=AMBER, ha="center")
    for i, (r, yy) in enumerate(zip(rows, y)):
        ax2.text(max(r["m_rho"], 0)+0.015, yy, f"{r['m_rho']:+.2f}", va="center",
                 ha="left", fontsize=8.2, color=INK if r["beats"] else MUTED)
    ax2.set_yticks([]); ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Spearman ρ — ranking quality (0 = random)", fontsize=8.5)
    _clean(ax2, x_axis=True)
    ax2.set_title("ranking the field", fontsize=10, color=MUTED, pad=10, loc="left")

    ax.set_title("MAE — accuracy", fontsize=10, color=MUTED, pad=10, loc="left")
    fig.savefig(DOCS / "mae_vs_baselines.png", dpi=150, facecolor=BG)
    plt.close(fig)


# ======================================================================
# Chart 2 — 2026 live holdout: the real test, week by week
# ======================================================================
def chart_live(holdout: dict):
    weeks = holdout["weeks"]
    items = []
    for key, w in weeks.items():
        agg = w["aggregate"]
        items.append({
            "week": w["week"],
            "mae": agg["model"]["mae"],
            "tr3": agg["trailing3"]["mae"],
            "last": agg["last_game"]["mae"],
            "beats": agg["model"]["mae"] < min(agg["trailing3"]["mae"], agg["last_game"]["mae"]),
        })
    items = sorted(items, key=lambda r: r["week"])
    weeks = [f"W{r['week']}" for r in items]
    mae = [r["mae"] for r in items]
    tr3 = [r["tr3"] for r in items]
    last = [r["last"] for r in items]

    fig = plt.figure(figsize=(11.5, 6.6))
    n_beat = sum(1 for r in items if r["beats"])
    _head(fig, f"2026 live holdout — beats both baselines {n_beat} of {len(items)} weeks",
          "The real test: predictions written to disk before kickoff, scored after games.  "
          "Pooled MAE (lower = better).")
    gs = fig.add_gridspec(1, 1, left=0.10, right=0.96, top=0.80, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(weeks)); w = 0.26
    ax.bar(x - w, last,  width=w, color=LAST,  label="Last year's same week", zorder=3)
    ax.bar(x,     tr3,   width=w, color=TR3,   label="Trailing 3 (same week)", zorder=3)
    ax.bar(x + w, mae,   width=w, color=MODEL, label="this model", zorder=3)
    # mark the weeks the model wins
    for i, r in enumerate(items):
        if r["beats"]:
            ax.scatter([i + w], [mae[i]], marker="_", s=0, color=MODEL)
            ax.text(i + w, mae[i] + max(mae)*0.02, "✓", ha="center", va="bottom",
                    fontsize=9, color=MODEL, fontweight="bold")
        else:
            ax.text(i + w, mae[i] + max(mae)*0.02, "—", ha="center", va="bottom",
                    fontsize=9, color=FAINT)
    ax.set_xticks(x); ax.set_xticklabels(weeks, fontsize=9)
    ax.set_ylabel("Pooled MAE (lower = better)", fontsize=9)
    ax.yaxis.set_visible(False)
    ax.set_ylim(0, max(max(mae), max(tr3), max(last))*1.22)
    _clean(ax, x_axis=True)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3,
              bbox_to_anchor=(0.0, 1.16))
    fig.savefig(DOCS / "holdout_2026.png", dpi=150, facecolor=BG)
    plt.close(fig)


def main():
    report = json.loads((ROOT / "models" / "eval_report.json").read_text(encoding="utf-8"))
    holdout = json.loads((ROOT / "models" / "holdout_score.json").read_text(encoding="utf-8"))
    chart_holdout(report)
    chart_live(holdout)
    print("wrote:", (DOCS / "mae_vs_baselines.png").resolve())
    print("wrote:", (DOCS / "holdout_2026.png").resolve())


if __name__ == "__main__":
    main()
