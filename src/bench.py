"""Bench — compare this model vs the naive baselines in FANTASY-POINTS space.

NOTES.md says the baselines to beat / sanity-check are FantasyPros consensus,
ESPN, Yahoo, Sleeper. Those sites publish *standard PPR weekly fantasy points*,
not raw stat components — so the honest comparison metric is PPR points, not
per-stat MAE. This bench computes, on the held-out 2024-2025 test split (the
same split eval_report.json uses):

  * our model   — the saved HGBR per (position, stat) predictions -> PPR points
  * last_game   — each stat's own value in the immediately prior week -> PPR
  * trailing3   — each stat's own 3-week trailing avg (shift 1) -> PPR
  * actual      — the real week's stat components -> PPR (the target)

and reports MAE / RMSE / Spearman (rank) per position and aggregate. A model
that "beats the baselines" in this space is one that ranks/estimates weekly
PPR better than "last week" or "3-week average" — which is exactly what the
public projection boards compete on.

Standard 1-PPR scoring (the most common league default):
  yards (rush/pass/rec)   1 pt / 10 yds
  TD (rush/pass/rec)       6 pts each
  receptions               1 pt each
  interception            -2 pts
  fumble lost             -2 pts

Run:  uv run python -m src.bench
"""
from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.clean import PROCESSED_DIR
from src.model import POS_TARGETS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
TEST_SEASONS = [2024, 2025]

# Standard 1-PPR point values.
SCORE = {
    "passing_yards": 0.1, "rushing_yards": 0.1, "receiving_yards": 0.1,
    "passing_tds": 6.0, "rushing_tds": 6.0, "receiving_tds": 6.0,
    "receptions": 1.0,
    "passing_interceptions": -2.0,
}


def _ppr(vals: dict[str, float]) -> float:
    """Convert a dict of stat components -> standard PPR points."""
    return sum(SCORE.get(s, 0.0) * (v or 0.0) for s, v in vals.items())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _stats(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = np.abs(pred - actual)
    return {
        "n": int(len(actual)),
        "mae": round(float(err.mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "spearman": round(_spearman(pred, actual), 4),
    }


def run() -> dict:
    payload = joblib.load(MODELS_DIR / "models.joblib")
    feat_cols = payload["meta"]["feature_columns"]
    models = payload["models"]

    feat = (pl.read_parquet(PROCESSED_DIR / "features.parquet")
            .sort(["player_id", "season", "week"])
            .filter(pl.col("season").is_in(TEST_SEASONS)))

    # Pre-compute per-stat baseline columns (last game, trailing3) via shift.
    stat_names = sorted({t for ts in POS_TARGETS.values() for t in ts})
    for s in stat_names:
        if s in feat.columns:
            feat = feat.with_columns(
                pl.col(s).shift(1).over(["player_id", "season"]).alias(s + "_bg"),
                (pl.col(s).shift(1).over(["player_id", "season"])
                 .rolling_mean(3, min_samples=1)
                 .over(["player_id", "season"]).alias(s + "_bg3")),
            )

    rows = feat.to_dicts()

    # Per-position accumulators.
    per_pos: dict[str, dict[str, list[float]]] = {}
    agg: dict[str, list[float]] = {"model": [], "actual": [],
                                   "last_game": [], "trailing3": []}

    for r in rows:
        pos = r["position"]
        targets = POS_TARGETS.get(pos, [])
        if not targets:
            continue

        # --- actuals ---
        actual_pts = _ppr({s: float(r[s] or 0.0) for s in targets})

        # --- our model ---
        X = np.array([[0.0 if r.get(c) is None else float(r.get(c)) for c in feat_cols]],
                     dtype=np.float32)
        model_vals = {}
        for s in targets:
            m = models.get(f"{pos}/{s}")
            if m is not None:
                model_vals[s] = float(m.predict(X)[0])
        model_pts = _ppr(model_vals)

        # --- baselines ---
        last_vals = {s: float(r.get(s + "_bg") or 0.0) for s in targets}
        tr3_vals = {s: float(r.get(s + "_bg3") or 0.0) for s in targets}
        last_pts = _ppr(last_vals)
        tr3_pts = _ppr(tr3_vals)

        for name, val in (("model", model_pts), ("actual", actual_pts),
                          ("last_game", last_pts), ("trailing3", tr3_pts)):
            agg[name].append(val)
            per_pos.setdefault(pos, {}).setdefault(name, []).append(val)

    # Summarize.
    out = {"scoring": "standard 1-PPR",
           "test_seasons": TEST_SEASONS,
           "positions": {}, "aggregate": {}}
    for pos in POS_TARGETS:
        if pos not in per_pos:
            continue
        d = per_pos[pos]
        out["positions"][pos] = {
            "n": len(d["actual"]),
            "actual_mean_pts": round(float(np.mean(d["actual"])), 2),
            "model": _stats(np.array(d["model"]), np.array(d["actual"])),
            "last_game": _stats(np.array(d["last_game"]), np.array(d["actual"])),
            "trailing3": _stats(np.array(d["trailing3"]), np.array(d["actual"])),
        }
    out["aggregate"] = {
        "n": len(agg["actual"]),
        "actual_mean_pts": round(float(np.mean(agg["actual"])), 2),
        "model": _stats(np.array(agg["model"]), np.array(agg["actual"])),
        "last_game": _stats(np.array(agg["last_game"]), np.array(agg["actual"])),
        "trailing3": _stats(np.array(agg["trailing3"]), np.array(agg["actual"])),
    }
    return out


def main() -> None:
    out = run()
    print(json.dumps(out, indent=2))
    out_path = MODELS_DIR / "bench_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
