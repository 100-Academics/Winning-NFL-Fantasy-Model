"""Generate the README charts from the CURRENT model artifacts.

The older ``notebooks/charts/*.png`` pre-date the mean-fit fix and the live
2026 holdout, so they are stale. These charts are regenerated from:
  * models/eval_report.json    (clean 2025 holdout: model vs naive baselines)
  * models/holdout_score.json  (2026 LIVE holdout, logged pre-kickoff)

Run:  uv run python notebooks/readme_charts.py
Writes: docs/mae_vs_baselines.png, docs/holdout_2026.png
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
OUT = REPO / "docs"
OUT.mkdir(exist_ok=True)

MODEL = "#2563eb"
TR3 = "#9ca3af"
LAST = "#d1d5db"
INK = "#111827"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
    "axes.titlesize": 13, "axes.titleweight": "bold", "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.labelcolor": INK, "xtick.color": "#475569",
    "ytick.color": "#475569",
})


def _stats(d: dict, k: str) -> tuple[float, float]:
    v = d.get(k, {})
    return v.get("mae", float("nan")), v.get("spearman", float("nan"))


# --------------------------------------------------------------------------- #
# Chart 1 — clean 2025 holdout: model vs naive baselines
# --------------------------------------------------------------------------- #
def chart_2025_holdout() -> None:
    ev = json.loads((MODELS / "eval_report.json").read_text())
    res = ev["results"]
    keys = list(res.keys())
    labels = [k.replace("/", " ") for k in keys]
    mae_m = [res[k]["test"]["mae"] for k in keys]
    mae_last = [res[k]["baseline_test"]["last_game"]["mae"] for k in keys]
    mae_tr3 = [res[k]["baseline_test"]["trailing3"]["mae"] for k in keys]
    rho = [res[k]["test_spearman"] for k in keys]
    n_beat = ev["n_beat_baseline"]
    n_tot = ev["n_models"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 6.4),
                                  gridspec_kw={"width_ratios": [1.55, 1.0]})
    y = np.arange(len(keys))[::-1]
    h = 0.27
    ax.barh(y + h, mae_last, height=h, color=LAST, label="last game")
    ax.barh(y, mae_tr3, height=h, color=TR3, label="trailing-3 avg")
    ax.barh(y - h, mae_m, height=h, color=MODEL, label="this model")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Test MAE (lower is better) — clean 2025 holdout")
    ax.set_title(f"Model vs naive baselines  ({n_beat}/{n_tot} position-stats beat both)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.02)

    # Spearman panel
    colors = [MODEL if res[k]["beats_baseline"] else "#93c5fd" for k in keys]
    ax2.barh(y, rho, height=0.62, color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Spearman rank corr (higher = better ranking)")
    ax2.set_title("2025 holdout — ranking quality")
    ax2.set_xlim(0, 1)
    ax2.grid(axis="y", visible=False)
    for yi, r in zip(y, rho):
        ax2.text(r + 0.015, yi, f"{r:+.2f}", va="center", fontsize=8, color=INK)

    fig.suptitle("Winning-NFL-Fantasy-Model — held-out accuracy", y=1.00,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = OUT / "mae_vs_baselines.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# --------------------------------------------------------------------------- #
# Chart 2 — 2026 LIVE holdout progression (the one clean test)
# --------------------------------------------------------------------------- #
def chart_2026_live() -> None:
    hs = json.loads((MODELS / "holdout_score.json").read_text())
    weeks = list(hs["weeks"].keys())
    wlab = [w.split("_w")[1] for w in weeks]
    weeks_int = [int(w.split("_w")[1]) for w in weeks]

    rho_m, rho_t, rho_l, mae_m, mae_t, mae_l = [], [], [], [], [], []
    for w in weeks:
        a = hs["weeks"][w].get("aggregate", {})
        mm, mr = _stats(a, "model")
        tm, tr = _stats(a, "trailing3")
        lm, lr = _stats(a, "last_game")
        mae_m.append(mm); mae_t.append(tm); mae_l.append(lm)
        rho_m.append(mr if mr == mr else None)
        rho_t.append(tr if tr == tr else None)
        rho_l.append(lr if (lr == lr) else None)

    x = np.arange(len(weeks))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.4))

    ax.plot(x, rho_m, "-o", color=MODEL, lw=2.4, ms=6, label="this model", zorder=5)
    ax.plot(x, rho_t, "--s", color=TR3, lw=1.8, ms=5, label="trailing-3 avg")
    ax.plot(x, rho_l, ":^", color=LAST, lw=1.8, ms=5, label="last game")
    ax.set_xticks(x); ax.set_xticklabels(["W" + w for w in wlab], fontsize=9)
    ax.set_ylabel("Spearman rank corr")
    ax.set_title("2026 live holdout — ranking, by week")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0.45, 0.85)
    ax.axhline(0.5, color="#e2e8f0", lw=0.8, ls=":")
    for xi, r in zip(x, rho_m):
        if r is not None:
            ax.annotate(f"{r:+.3f}", (xi, r), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, color=MODEL)
    # mark the skipped week gap
    gaps = [i for i, w in enumerate(weeks_int)
            if i and w - weeks_int[i - 1] > 1]
    for i in gaps:
        ax.axvspan(i - 1.5, i - 0.5, color="#f1f5f9", zorder=0)
        ax.text(i - 1, 0.475, "W6\nskipped", ha="center", fontsize=7.5, color="#94a3b8")

    ax2.plot(x, mae_m, "-o", color=MODEL, lw=2.4, ms=6, label="this model", zorder=5)
    ax2.plot(x, mae_t, "--s", color=TR3, lw=1.8, ms=5, label="trailing-3 avg")
    ax2.plot(x, mae_l, ":^", color=LAST, lw=1.8, ms=5, label="last game")
    ax2.set_xticks(x); ax2.set_xticklabels(["W" + w for w in wlab], fontsize=9)
    ax2.set_ylabel("MAE (ESPN Std pts, lower = better)")
    ax2.set_title("2026 live holdout — error, by week")
    ax2.legend(frameon=False, fontsize=9)
    for i in gaps:
        ax2.axvspan(i - 1.5, i - 0.5, color="#f1f5f9", zorder=0)

    fig.suptitle("The live season is the only clean test — logged before kickoff",
                 y=1.00, fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = OUT / "holdout_2026.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    chart_2025_holdout()
    chart_2026_live()
    print("done.")
