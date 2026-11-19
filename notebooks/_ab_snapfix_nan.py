"""A/B: the audit's live-path correctness fixes + the NaN-vs-zero-fill question.

Findings being tested (2026-11-19):
  B1   snap_pct_* were silently 0.0 on the LIVE board (no dummy row).  FIXED.
  B2   prev_season_* were silently 0.0 on the LIVE board (panel filtered to
       the current season only).  FIXED. (Live-only; the trained model already
       had correct bridge values from the played-week path.)
  SNAP-ORDER  `_attach_snap_shares` computed its trailing over a SCRAMBLED
       physical order (polars `unique()` after `.sort()` in polars 1.44), so
       snap_pct_tr3/s2d windowed over arbitrary rows, not the prior weeks.
       FIXED (re-sort before the trailing). This ALSO changes the played-week
       features -> it changes what the model trains on, so it is A/B'd here.
  B3   `.fill_null(0.0)` before HGBR turns nulls into 0.0 (dome -> 0F, missing
       NGS -> 0% separation, week-1 -> 0-yard game). HGBR handles NaN natively.
       Does keeping NaN beat zero-fill on the clean 2025 holdout?

Three arms, all scored on the SAME clean 2025 test rows (apples-to-apples):
  V0  OLD snap values (scrambled order) + zero-fill  == current shipped model
  V1  CORRECTED snap values (week order) + zero-fill
  V2  CORRECTED snap values (week order) + NaN

Cross-position NGS columns (e.g. QB receiving-NGS, RB passing-NGS) are structurally
all-null for a given position: constant-0 in the zero-fill arm, all-NaN in the
NaN arm. HGBR can't bin a column with <2 distinct values, so the NaN arm drops
those per-position columns; we assert they're also empty in the 2025 test set,
so the drop is a no-op and the arms stay comparable.

Decision rule: ship only if it beats V0 on the 2025 holdout; if basically the
same, keep the simpler thing.

Run:  uv run python notebooks/_ab_snapfix_nan.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

from src import features as F
from src.model import (
    POS_TARGETS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    feature_columns, _split, _mae, _rmse, _spearman, _grid,
)
from src.clean import PROCESSED_DIR, RAW_DIR

REPO = pathlib.Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
GRID = _grid(quick=False)  # the same 24-config grid the shipped model used
SNAP = ["snap_pct_tr3", "snap_pct_s2d"]
print(f"grid: {len(GRID)} configs | split train {TRAIN_SEASONS} val {VAL_SEASONS} test {TEST_SEASONS}")

feat_new = pl.read_parquet(PROCESSED_DIR / "features.parquet")   # CORRECTED snap
fc = feature_columns(feat_new)
print(f"features: {feat_new.height} rows x {len(fc)} cols")

# --- reconstruct the OLD (scrambled-order) snap values for the V0 baseline ---
snaps_raw = F.load_snap_counts(RAW_DIR)   # ends with .sort().unique() -> scrambled
g_old = snaps_raw.with_columns([
    F._trailing(["player_display_name"], "offense_pct", 3).alias("snap_pct_tr3"),
    F._s2d_rate(["player_display_name"], "offense_pct").alias("snap_pct_s2d"),
]).select(["season", "week", "player_display_name", *SNAP])
feat_old = (feat_new.drop(SNAP)
            .join(g_old, on=["season", "week", "player_display_name"], how="left"))
diff_cols = [c for c in fc if not feat_old.select(c).equals(feat_new.select(c))]
print("cols that differ between old/new feature frames:", diff_cols)


def _active_cols(splits: dict, fill: bool) -> list[str]:
    """Feature cols with >=2 distinct non-null values in the train set.

    In the NaN arm, an all-null train column (cross-position NGS) crashes HGBR's
    binning (no threshold to find). In the zero-fill arm the same column is a
    constant 0 (never split on). Dropping it from BOTH arms is therefore a
    no-op for the zero-fill arms and a fix for the NaN arm -> comparable.
    We assert the dropped cols are also empty in the 2025 test set.
    """
    tr, te = splits["train"], splits["test"]
    active = []
    dropped = []
    for c in fc:
        if tr[c].drop_nulls().unique().len() >= 2:
            active.append(c)
        else:
            if te[c].drop_nulls().len():
                raise AssertionError(
                    f"{c}: all-null in train but {te[c].drop_nulls().len()} "
                    f"non-null in test — dropping would lose test signal")
            dropped.append(c)
    return active


def make_X(sub: pl.DataFrame, cols: list[str], fill: bool) -> np.ndarray:
    if fill:
        return sub.select(cols).fill_null(0.0).to_numpy().astype(np.float32)
    return sub.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy().astype(np.float32)


def train_arm(feat: pl.DataFrame, fill: bool) -> dict:
    out = {}
    for pos, targets in POS_TARGETS.items():
        splits = _split(feat, pos)
        cols = _active_cols(splits, fill)
        Xtr = make_X(splits["train"], cols, fill)
        Xva = make_X(splits["val"], cols, fill)
        Xte = make_X(splits["test"], cols, fill)
        for target in targets:
            ytr = splits["train"].select(target).to_series().to_numpy().astype(np.float64)
            yva = splits["val"].select(target).to_series().to_numpy().astype(np.float64)
            yte = splits["test"].select(target).to_series().to_numpy().astype(np.float64)
            best = None
            for p in GRID:
                m = HGBR(loss="squared_error", random_state=0, **p)
                m.fit(Xtr, ytr)
                va = float(np.abs(m.predict(Xva) - yva).mean())
                if best is None or va < best[0]:
                    best = (va, p)
            m = HGBR(loss="squared_error", random_state=0, **best[1])
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            out[f"{pos}/{target}"] = {"mae": round(_mae(pred, yte), 3),
                                      "rmse": round(_rmse(pred, yte), 3),
                                      "rho": round(_spearman(pred, yte), 4),
                                      "n_cols": len(cols),
                                      "pred": pred, "actual": yte}
    return out


print("\n[V0] old snap + zero-fill  ...")
V0 = train_arm(feat_old, fill=True)
print("[V1] corrected snap + zero-fill ...")
V1 = train_arm(feat_new, fill=True)
print("[V2] corrected snap + NaN ...")
V2 = train_arm(feat_new, fill=False)


def pool(arms):
    pred = np.concatenate([arms[k]["pred"] for k in arms])
    act = np.concatenate([arms[k]["actual"] for k in arms])
    return round(_mae(pred, act), 3), round(_spearman(pred, act), 4)


keys = [f"{p}/{t}" for p, ts in POS_TARGETS.items() for t in ts]
print("\n" + "=" * 84)
print(f"{'pos/target':<26}{'V0 old':>9}{'V1 corr':>9}{'V2 nan':>9}   |  dMAE V1-V0  dMAE V2-V0")
print("-" * 84)
for k in keys:
    print(f"{k:<26}{V0[k]['mae']:>9.2f}{V1[k]['mae']:>9.2f}{V2[k]['mae']:>9.2f}   |  "
          f"{V1[k]['mae']-V0[k]['mae']:+8.2f}  {V2[k]['mae']-V0[k]['mae']:+8.2f}")
print("-" * 84)
p0 = pool(V0); p1 = pool(V1); p2 = pool(V2)
print(f"{'POOLED MAE':<26}{p0[0]:>9.2f}{p1[0]:>9.2f}{p2[0]:>9.2f}   |  {p1[0]-p0[0]:+8.2f}  {p2[0]-p0[0]:+8.2f}")
print(f"{'POOLED rho':<26}{p0[1]:>9.4f}{p1[1]:>9.4f}{p2[1]:>9.4f}   |  {p1[1]-p0[1]:+8.4f}  {p2[1]-p0[1]:+8.4f}")
print("=" * 84)

print("\nPer-stat Spearman  (V0 -> V1 -> V2):")
for k in keys:
    print(f"  {k:<26}{V0[k]['rho']:+.3f}  {V1[k]['rho']:+.3f}  {V2[k]['rho']:+.3f}")


def wins(arm):
    dm = sum(1 for k in keys if arm[k]["mae"] < V0[k]["mae"])
    dr = sum(1 for k in keys if arm[k]["rho"] > V0[k]["rho"])
    return dm, dr
print(f"\nstats with MAE < V0 : V1 {wins(V1)[0]}/16   V2 {wins(V2)[0]}/16")
print(f"stats with rho > V0 : V1 {wins(V1)[1]}/16   V2 {wins(V2)[1]}/16")
_qb = next(k for k in keys if k.startswith("QB/"))
_rb = next(k for k in keys if k.startswith("RB/"))
print(f"feature cols used (V2/{_qb}): {V2[_qb]['n_cols']}  "
      f"(V0/{_qb}: {V0[_qb]['n_cols']})")

report = {
    "note": "A/B on clean 2025 holdout. V0=old scrambled-snap+zero (== shipped model), "
            "V1=corrected-snap+zero, V2=corrected-snap+NaN. B1/B2 are live-only fixes "
            "(verified exact on the live panel) and do not change trained features. "
            "Cross-position all-null-in-train NGS cols dropped per-position (asserted "
            "empty in test; no-op for the zero-fill arms).",
    "split": {"train": TRAIN_SEASONS, "val": VAL_SEASONS, "test": TEST_SEASONS},
    "V0_old_zero": {k: {x: V0[k][x] for x in ("mae", "rmse", "rho", "n_cols")} for k in keys},
    "V1_corr_zero": {k: {x: V1[k][x] for x in ("mae", "rmse", "rho", "n_cols")} for k in keys},
    "V2_corr_nan": {k: {x: V2[k][x] for x in ("mae", "rmse", "rho", "n_cols")} for k in keys},
    "pooled": {"V0": list(p0), "V1": list(p1), "V2": list(p2)},
    "wins_vs_V0": {"V1": wins(V1), "V2": wins(V2)},
}
(MODELS / "ab_snapfix_nan_report.json").write_text(json.dumps(report, indent=2))
print(f"\n-> {MODELS / 'ab_snapfix_nan_report.json'}")
