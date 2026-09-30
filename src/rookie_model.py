"""Phase 5.5 — rookie prior model.

The main model (Phase 4) predicts a player-week's component stats from
TRAILING FORM. A true rookie in week 1 has NO trailing form (all zeros), so the
main model degenerates to a league-average guess. The rookie prior fills that
gap: it predicts a rookie's expected per-week output from *static, pre-season*
inputs that are known before their first NFL game:

  * draft capital : round / pick / UDFA
  * pre_draft_grade (CFBD)
  * age at draft
  * combine athleticism (40 / vertical / broad / cone / shuttle / bench)
  * college production (rate stats YPA/YPC/YPR/PCT + volume, CFBD)
  * college usage shares (dominator-style, CFBD)

Training data = FIRST-SEASON player-weeks, pooled across 2016-2025. For each
(rookie, week) row the target is that week's component stat (same column the
main model predicts) and the features are the rookie's static prior features
(constant across the rookie's season). Same scale as the main model => we can
blend them with shrinkage in the predict CLI:

    final = w * main + (1 - w) * prior,   w = games_played / (games_played + k)

So week 1 leans on the prior (kills the "lucky first week" problem) and the
real NFL output takes over fast — with k=0.5 (tuned on the 2023 rookie season:
uniform k=0.5 beat k=4 on both 2024 and 2025 test, 6.37 vs 6.89 MAE) the main
model leads by ~4-5 games.

Run:
    uv run python -m src.rookie_model
        -> models/rookie_prior.joblib + models/rookie_prior_report.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

from src.model import POS_TARGETS, HEADLINE
from src.clean import PROCESSED_DIR
from src.rookie_features import COMBINE_COLS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# Splits (chronological). Rookies are rare, so we pool aggressively:
# train on 2016-2022 first-seasons, val 2023, test 2024-2025.
TRAIN_SEASONS = list(range(2016, 2023))
VAL_SEASONS = [2023]
TEST_SEASONS = [2024, 2025]

# Non-feature columns in rookie_features (true identifiers / labels / strings).
# NOTE: draft_round, draft_pick, pre_draft_grade, udfa, was_drafted, age_at_draft
# are all features (draft capital + age are the prior).
NON_FEATURE = {
    "player_id", "display_name", "position", "draft_team", "college_team",
    "college_conference", "draft_year",
}


def prior_feature_columns(df: pl.DataFrame, prior_table: pl.DataFrame | None = None) -> list[str]:
    """Model-usable prior features: the static per-player columns that exist in
    BOTH the training frame and the rookie_features table. This excludes the
    target stats and per-game context (season/week/spread/home/...) that leak in
    from the joined features frame."""
    if prior_table is not None:
        table_cols = set(prior_table.columns)
    else:
        table_cols = set(df.columns)
    out = []
    for c in df.columns:
        if c in NON_FEATURE or c not in table_cols:
            continue
        try:
            df[c].cast(pl.Float64)
        except Exception:
            continue
        out.append(c)
    return out


def load_rookie_first_seasons() -> pl.DataFrame:
    """Join the main features (per-week targets) with the static rookie prior,
    keeping only each player's FIRST season (their rookie season)."""
    feat = pl.read_parquet(PROCESSED_DIR / "features.parquet")
    prior = pl.read_parquet(PROCESSED_DIR / "rookie_features.parquet")

    # first regular-season year per player
    first = (feat.group_by("player_id")
             .agg(pl.col("season").min().alias("first_season")))

    # rookie-weeks = feature rows in the player's first season
    rw = feat.join(first, on="player_id", how="left")
    rw = rw.filter(pl.col("season") == pl.col("first_season"))
    rw = rw.filter(pl.col("position").is_in(list(POS_TARGETS)))

    # attach static prior features (flag whether the player is in the prior table)
    prior_feats = [c for c in prior.columns if c not in NON_FEATURE]
    rw = rw.join(prior.select(["player_id", *prior_feats]),
                 on="player_id", how="left")
    rw = rw.with_columns(pl.col("age_at_draft").is_not_null().alias("_in_prior"))
    # keep only players present in the rookie prior (drafted or UDFA in our window)
    rw = rw.filter(pl.col("_in_prior")).drop("_in_prior")
    return rw


def _xy(sub: pl.DataFrame, feat_cols: list[str], target: str):
    X = sub.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y = sub.select(target).to_series().to_numpy().astype(np.float64)
    return X, y


def _mae(p, a): return float(np.abs(p - a).mean())
def _rmse(p, a): return float(np.sqrt(((p - a) ** 2).mean()))


def _grid(quick: bool):
    if quick:
        return [dict(max_iter=150, learning_rate=0.06, max_depth=4,
                     l2_regularization=5.0, min_samples_leaf=20)]
    import itertools
    return [dict(max_iter=mi, learning_rate=lr, max_depth=d,
                 l2_regularization=l2, min_samples_leaf=ms)
            for mi, lr, d, l2, ms in itertools.product(
                (150, 300), (0.04, 0.08), (3, 4), (1.0, 5.0, 10.0), (10, 20, 40))]


def train_one(pos, target, data, feat_cols, quick):
    tr = data.filter((pl.col("position") == pos) & pl.col("season").is_in(TRAIN_SEASONS))
    va = data.filter((pl.col("position") == pos) & pl.col("season").is_in(VAL_SEASONS))
    te = data.filter((pl.col("position") == pos) & pl.col("season").is_in(TEST_SEASONS))
    if tr.height < 30 or te.height < 10:
        return None
    Xtr, ytr = _xy(tr, feat_cols, target)
    Xva, yva = _xy(va, feat_cols, target) if va.height else (None, None)
    Xte, yte = _xy(te, feat_cols, target)

    # grid search on val (squared_error = conditional MEAN — the
    # calibration-correct central estimate; the median under-projects
    # zero-inflated stats, see src.model docstring). Fall back to train error.
    best = None
    for params in _grid(quick):
        m = HGBR(loss="squared_error", random_state=0, **params)
        m.fit(Xtr, ytr)
        if Xva is not None and yva is not None and len(yva):
            score = float(np.abs(m.predict(Xva) - yva).mean())
        else:
            score = _mae(m.predict(Xtr), ytr)
        if best is None or score < best[0]:
            best = (score, params)
    m = HGBR(loss="squared_error", random_state=0, **best[1])
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    return {
        "position": pos, "target": target,
        "n_train": len(ytr), "n_val": len(yva) if yva is not None else 0, "n_test": len(yte),
        "best_params": best[1], "val_mae": best[0],
        "test_mae": _mae(pred, yte), "test_rmse": _rmse(pred, yte),
        "test_signed_bias": float(pred.mean() - yte.mean()),
        "model": m,
    }


def run(quick=False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    from src.rookie_features import PROCESSED_DIR as _PD
    import polars as _pl
    prior_table = _pl.read_parquet(_PD / "rookie_features.parquet")
    data = load_rookie_first_seasons()
    feat_cols = prior_feature_columns(data, prior_table)
    # drop any feature that is 100% null (e.g. a combine metric never present)
    feat_cols = [c for c in feat_cols if data[c].null_count() < data.height]
    print(f"rookie first-season rows: {data.height:,} across {data['player_id'].unique().len():,} rookies")
    print(f"prior features ({len(feat_cols)}): {feat_cols}")

    results, n = {}, 0
    for pos, targets in POS_TARGETS.items():
        for target in targets:
            rec = train_one(pos, target, data, feat_cols, quick)
            if rec is None:
                print(f"  {pos} {target}: skipped (too few rows)")
                continue
            results[f"{pos}/{target}"] = rec
            n += 1
            print(f"  {pos:<3} {target:<22} test MAE {rec['test_mae']:7.2f} "
                  f"(n_train {rec['n_train']}, n_test {rec['n_test']})")

    def _clean(r): return {k: v for k, v in r.items() if k != "model"}
    report = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "quick": quick,
        "split": {"train": TRAIN_SEASONS, "val": VAL_SEASONS, "test": TEST_SEASONS},
        "n_features": len(feat_cols), "feature_columns": feat_cols, "n_models": n,
        "results": {k: _clean(v) for k, v in results.items()},
    }
    (MODELS_DIR / "rookie_prior_report.json").write_text(json.dumps(report, indent=2, default=str))
    import joblib
    joblib.dump({
        "meta": {"generated": report["generated"], "feature_columns": feat_cols,
                 "pos_targets": POS_TARGETS, "split": report["split"]},
        "models": {k: v["model"] for k, v in results.items()},
    }, MODELS_DIR / "rookie_prior.joblib")
    print(f"\n{n} rookie prior models -> {MODELS_DIR / 'rookie_prior.joblib'}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    run(quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
