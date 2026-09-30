"""Phase 4 — model training, evaluation, and saving.

Per (position, target-stat) we train a gradient-boosting regressor
(`sklearn.ensemble.HistGradientBoostingRegressor` with quantile loss) and
compare it against two naive baselines:
  * last_game  — the player's own value in the immediately previous week
  * trailing3  — the player's own 3-game trailing average (a feature we already build)

The model must BEAT these on the test set or it adds noise, not signal.

Split (chronological, never random):
    train 2016-2022   |   val 2023   |   test 2024-2025

Feature rule (leak-free): use every feature column EXCEPT the identifier keys
and the 14 *current-week* target columns. A target's own trailing form
(`passing_yards_tr3`, `_s2d`, ...) is kept — it is a strictly-prior-week
predictor, not a leak.

Loss choice (calibration-correct):
  * CENTRAL estimate  -> ``loss="squared_error"`` (the conditional MEAN).
    Stat targets are right-skewed and zero-inflated, so their MEDIAN
    (``loss="quantile", quantile=0.5`` / ``absolute_error``) sits below the
    mean and systematically UNDER-projects (measured: RB −40%, TE −33%,
    WR −30% of mean; TDs collapsed to 0). Because fantasy value is a LINEAR
    combination of the stats — E[points] = Σ scoreᵢ·E[statᵢ] — the only
    consistent central estimate is the mean; a median model breaks that
    linearity (median of a sum ≠ sum of medians) and distorts cross-position
    (flex) rankings even when per-stat MAE looks fine. Fitting the mean puts
    per-stat bias at +0…+9% while preserving Spearman (verified 2024-25 test).
  * BANDS (p10/p90)  -> ``loss="quantile"`` at 0.10 / 0.90, unchanged. Quantile
    regression is the right tool for the uncertainty band; it is NOT the
    right tool for the point estimate.

Run:
    uv run python -m src.model            # train, eval, save to models/
    uv run python -m src.model --quick    # smaller grid, fewer iters (fast smoke test)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

from src.clean import PROCESSED_DIR, TARGET_COLS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# --------------------------------------------------------------------------- #
# Splits & targets
# --------------------------------------------------------------------------- #
TRAIN_SEASONS = list(range(2016, 2023))   # 2016..2022
VAL_SEASONS   = [2023]
TEST_SEASONS  = [2024, 2025]

# Identifier / key columns — never used as model features.
KEY_COLS = [
    "player_id", "player_display_name", "season", "week", "team",
    "opponent_team", "game_id", "position", "position_group",
]

# Per-position target stats (the component outcomes we predict). K is excluded
# (no clean component target in the panel; its value is in a separate project).
POS_TARGETS: dict[str, list[str]] = {
    "QB": ["passing_yards", "passing_tds", "passing_interceptions"],
    "RB": ["rushing_yards", "rushing_tds", "receptions",
           "receiving_yards", "receiving_tds"],
    "WR": ["receptions", "receiving_yards", "receiving_tds"],
    "TE": ["receptions", "receiving_yards", "receiving_tds"],
}

# Headline stats worth a p10/p90 confidence band in the final output.
HEADLINE = {
    "QB": {"passing_yards", "passing_tds"},
    "RB": {"rushing_yards", "rushing_tds", "receiving_yards", "receptions"},
    "WR": {"receiving_yards", "receptions", "receiving_tds"},
    "TE": {"receiving_yards", "receptions"},
}


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #
def load_features(path: pathlib.Path = PROCESSED_DIR / "features.parquet") -> pl.DataFrame:
    return pl.read_parquet(path).sort(["position", "player_id", "season", "week"])


def feature_columns(feat: pl.DataFrame) -> list[str]:
    """All model-usable columns: everything except keys and current-week targets."""
    drop = set(KEY_COLS) | set(TARGET_COLS)
    return [c for c in feat.columns if c not in drop]


def _xy(sub: pl.DataFrame, feat_cols: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    X = sub.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y = sub.select(target).to_series().to_numpy().astype(np.float64)
    return X, y


def _split(feat: pl.DataFrame, pos: str) -> dict[str, pl.DataFrame]:
    sub = feat.filter(pl.col("position") == pos)
    return {
        "train": sub.filter(pl.col("season").is_in(TRAIN_SEASONS)),
        "val":   sub.filter(pl.col("season").is_in(VAL_SEASONS)),
        "test":  sub.filter(pl.col("season").is_in(TEST_SEASONS)),
    }


def _baselines(sub: pl.DataFrame, target: str) -> dict[str, np.ndarray]:
    """Naive baselines for a subset (test): last-game and trailing-3 of the
    player's own target, computed from the target's own prior weeks."""
    s = sub.sort(["player_id", "season", "week"])
    last_game = s.with_columns(
        pl.col(target).shift(1).over(["player_id", "season"]).alias("_bg")
    ).select("_bg").to_series().to_numpy()
    tr3 = s.with_columns(
        pl.col(target).shift(1).over(["player_id", "season"])
        .rolling_mean(3, min_samples=1).over(["player_id", "season"]).alias("_b3")
    ).select("_b3").to_series().to_numpy()
    return {"last_game": last_game, "trailing3": tr3}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(pred - actual).mean())


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(((pred - actual) ** 2).mean()))


def _spearman(pred: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation (higher = better ranking). Uses average ranks to
    handle ties without scipy."""
    from scipy.stats import rankdata
    p = rankdata(pred)
    a = rankdata(actual)
    if np.std(p) == 0 or np.std(a) == 0:
        return float("nan")
    return float(np.corrcoef(p, a)[0, 1])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _grid(quick: bool) -> list[dict]:
    if quick:
        return [
            dict(max_iter=100, learning_rate=0.06, max_depth=4,
                 l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=50),
        ]
    import itertools
    return [
        dict(max_iter=mi, learning_rate=lr, max_depth=depth,
             l2_regularization=l2, max_leaf_nodes=31, min_samples_leaf=50)
        for mi, lr, depth, l2 in itertools.product(
            (150, 300, 500), (0.03, 0.06), (3, 5), (1.0, 5.0)
        )
    ]


def _fit(HGBR_cls, params: dict, X, y, quantile=0.5):
    m = HGBR_cls(loss="quantile", quantile=quantile, random_state=0, **params)
    m.fit(X, y)
    return m


def train_one(pos: str, target: str, feat: pl.DataFrame,
              feat_cols: list[str], quick: bool) -> dict:
    splits = _split(feat, pos)
    Xtr, ytr = _xy(splits["train"], feat_cols, target)
    Xva, yva = _xy(splits["val"], feat_cols, target)
    Xte, yte = _xy(splits["test"], feat_cols, target)
    base = _baselines(splits["test"], target)

    # Baseline test metrics (fill NaN baselines with 0 = "no prior game")
    def _m(b):
        b = np.nan_to_num(b, nan=0.0)
        return {"mae": _mae(b, yte), "rmse": _rmse(b, yte)}
    base_metrics = {"last_game": _m(base["last_game"]),
                    "trailing3": _m(base["trailing3"])}

    # Grid search on validation (squared_error = conditional MEAN, the
    # calibration-correct central estimate; see module docstring). Pick best,
    # evaluate on test.
    best = None
    for params in _grid(quick):
        m = HGBR(loss="squared_error", random_state=0, **params)
        m.fit(Xtr, ytr)
        va = np.abs(m.predict(Xva) - yva).mean()
        if best is None or va < best["val_mae"]:
            best = {"val_mae": va, "params": params}

    m = HGBR(loss="squared_error", random_state=0, **best["params"])
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    test_metrics = {"mae": _mae(pred, yte), "rmse": _rmse(pred, yte),
                    "signed_bias": float(pred.mean() - yte.mean())}
    ranking = _spearman(pred, yte)
    beats = test_metrics["mae"] < min(base_metrics["last_game"]["mae"],
                                      base_metrics["trailing3"]["mae"])

    record = {
        "position": pos, "target": target,
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "best_params": best["params"], "val_mae": best["val_mae"],
        "test": test_metrics, "baseline_test": base_metrics,
        "test_spearman": ranking, "beats_baseline": bool(beats),
        "model": m,
    }

    # Headline stats: p10 / p90 bands (refit best params at each quantile).
    if target in HEADLINE.get(pos, set()):
        lo = _fit(HGBR, best["params"], Xtr, ytr, quantile=0.10).predict(Xte)
        hi = _fit(HGBR, best["params"], Xtr, ytr, quantile=0.90).predict(Xte)
        record["test_p10"] = float(lo.mean())
        record["test_p90"] = float(hi.mean())
        record["band_models"] = {
            "p10": _fit(HGBR, best["params"], Xtr, ytr, quantile=0.10),
            "p90": _fit(HGBR, best["params"], Xtr, ytr, quantile=0.90),
        }
    return record


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(quick: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feat = load_features()
    feat_cols = feature_columns(feat)
    print(f"features: {feat.height:,} rows, {len(feat_cols)} model features")
    print(f"positions/targets: "
          + ", ".join(f"{p}({len(t)})" for p, t in POS_TARGETS.items()))
    print(f"split: train {TRAIN_SEASONS} | val {VAL_SEASONS} | test {TEST_SEASONS}\n")

    results: dict[str, dict] = {}
    n_beats = n_models = 0
    for pos, targets in POS_TARGETS.items():
        for target in targets:
            rec = train_one(pos, target, feat, feat_cols, quick)
            results[f"{pos}/{target}"] = rec
            n_models += 1
            n_beats += int(rec["beats_baseline"])
            print(f"  {pos:<3} {target:<22} test MAE {rec['test']['mae']:7.2f} "
                  f"(last {rec['baseline_test']['last_game']['mae']:7.2f}, "
                  f"tr3 {rec['baseline_test']['trailing3']['mae']:7.2f}) "
                  f"rho {rec['test_spearman']:+.3f} "
                  f"{'BEATS' if rec['beats_baseline'] else 'no   '}")

    # Strip unserializable models for the JSON report.
    def _clean(rec):
        r = {k: v for k, v in rec.items() if k not in ("model", "band_models")}
        return r
    report = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "quick": quick,
        "split": {"train": TRAIN_SEASONS, "val": VAL_SEASONS, "test": TEST_SEASONS},
        "n_features": len(feat_cols),
        "n_models": n_models,
        "n_beat_baseline": n_beats,
        "results": {k: _clean(v) for k, v in results.items()},
    }
    report_path = MODELS_DIR / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    # Save the trained models (joblib) + the report.
    import joblib
    models_payload = {
        "meta": {
            "generated": report["generated"],
            "split": report["split"],
            "n_features": len(feat_cols),
            "feature_columns": feat_cols,
            "pos_targets": POS_TARGETS,
        },
        "models": {k: v["model"] for k, v in results.items()},
        "bands": {k: v["band_models"] for k, v in results.items() if "band_models" in v},
    }
    models_path = MODELS_DIR / "models.joblib"
    joblib.dump(models_payload, models_path)

    print(f"\n{'='*64}")
    print(f"models beating baseline: {n_beats}/{n_models}")
    print(f"report -> {report_path}")
    print(f"models -> {models_path}")
    print(f"{'='*64}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="smaller grid / fewer iters (fast smoke test)")
    args = ap.parse_args(argv)
    run(quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
