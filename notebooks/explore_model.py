"""Phase 6 — model evaluation charts.

Loads `models/eval_report.json` + the saved Phase 4 models and produces:
  * per-target test MAE: model vs naive baselines (grouped bar chart)
  * predicted-vs-actual scatter for headline stats (QB passing_yards,
    WR receiving_yards) on the test split
  * test-set ranking quality (Spearman) per target

Run:
    uv run python notebooks/explore_model.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import joblib

REPO = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "models"
CHARTS_DIR = REPO / "notebooks" / "charts"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_report() -> dict:
    return json.loads((MODELS_DIR / "eval_report.json").read_text())


def chart_mae(report: dict) -> None:
    keys = [k for k in report["results"] if "/" in k]
    model = [report["results"][k]["test"]["mae"] for k in keys]
    last = [report["results"][k]["baseline_test"]["last_game"]["mae"] for k in keys]
    tr3 = [report["results"][k]["baseline_test"]["trailing3"]["mae"] for k in keys]

    x = np.arange(len(keys))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - w, model, w, label="model", color="#1f77b4")
    ax.bar(x, last, w, label="last game", color="#ff7f0e", alpha=0.7)
    ax.bar(x + w, tr3, w, label="trailing 3-avg", color="#2ca02c", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("test MAE (2024-2025)")
    ax.set_title("Test MAE: model vs naive baselines (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "mae_vs_baselines.png", dpi=130)
    plt.close(fig)


def chart_scatter(report: dict) -> None:
    payload = joblib.load(MODELS_DIR / "models.joblib")
    feat_cols = payload["meta"]["feature_columns"]
    feat = pl.read_parquet(REPO / "data" / "processed" / "features.parquet")
    test_seasons = report["split"]["test"]

    picks = [("QB", "passing_yards"), ("WR", "receiving_yards")]
    fig, axes = plt.subplots(1, len(picks), figsize=(11, 5))
    for ax, (pos, target) in zip(axes, picks):
        sub = feat.filter(
            (pl.col("position") == pos) & pl.col("season").is_in(test_seasons)
        )
        X = sub.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
        y = sub.select(target).to_series().to_numpy().astype(np.float64)
        pred = payload["models"][f"{pos}/{target}"].predict(X)
        rho = report["results"][f"{pos}/{target}"]["test_spearman"]
        ax.scatter(pred, y, s=6, alpha=0.25)
        lim = max(np.percentile(y, 99.5), np.percentile(pred, 99.5))
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(f"{pos} {target}\nSpearman rho={rho:+.3f}", fontsize=9)
    fig.suptitle("Predicted vs actual — test split (2024-2025)", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "pred_vs_actual.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_ranking(report: dict) -> None:
    keys = [k for k in report["results"] if "/" in k]
    rho = [report["results"][k]["test_spearman"] for k in keys]
    fig, ax = plt.subplots(figsize=(13, 4))
    colors = ["#2ca02c" if (r == r and r > 0.3) else "#7f7f7f" for r in rho]
    ax.bar(range(len(keys)), [r if r == r else 0 for r in rho], color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("test Spearman rho")
    ax.set_title("Ranking quality per target (green = rho > 0.3; gray = sparse/NaN)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ranking_quality.png", dpi=130)
    plt.close(fig)


def main() -> int:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    report = _load_report()
    chart_mae(report)
    chart_scatter(report)
    chart_ranking(report)
    for p in sorted(CHARTS_DIR.glob("*.png")):
        print(f"chart -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
