"""A/B: third-party audit proposals (2026-11-17).

Arms on the clean 2025 holdout (train 2016-2023, test 2025):
  V0  baseline (shipped 80 features)
  V1  + snap-share: offense_pct tr3/s2d (snap_counts.parquet, never wired before)
  V2  + vacated-opportunity: team-level trailing volume of same-position
      teammates designated Out/Doubtful in the official injury report (week W)
  V3  both

Shipped hyperparameters per (pos, target); squared_error; internal early
stopping (validation_fraction 0.1, n_iter_no_change 10) — same as models.joblib.
Decision rule: apply an arm only if pooled Spearman beats V0 by > +0.01.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

REPO = Path(__file__).resolve().parent.parent
T0 = time.time()

feat = pl.read_parquet(REPO / "data/processed/features.parquet")
payload = joblib.load(REPO / "models/models.joblib")
BASE = payload["meta"]["feature_columns"]
POS_TARGETS: dict[str, list[str]] = payload["meta"]["pos_targets"]
if not isinstance(POS_TARGETS, dict):
    raise SystemExit(f"unexpected pos_targets shape: {type(POS_TARGETS)}")
POS = ["QB", "RB", "WR", "TE"]
assert all(p in feat.columns for p in ["season", "week", "position", "team"])

# ------------------------------------------------------------ new features
# --- V1: snap share (offense_pct), strictly-prior tr3 / s2d ---
sc = (pl.read_parquet(REPO / "data/raw/snap_counts.parquet")
      .cast({"season": pl.Int32, "week": pl.Int32}))
snaps = (
    sc.filter(pl.col("position").is_in(POS))
      .select(["season", "week", "player", "offense_pct"])
      .rename({"player": "player_display_name"})
      .sort(["player_display_name", "season", "week"])
      .unique(subset=["season", "week", "player_display_name"])
)
snaps = snaps.with_columns(
    pl.col("offense_pct").shift(1).over(["player_display_name", "season"])
    .rolling_mean(3, min_samples=1).over(["player_display_name", "season"])
    .alias("snap_pct_tr3"),
    pl.col("offense_pct").shift(1).over(["player_display_name", "season"])
    .rolling_mean(20, min_samples=1).over(["player_display_name", "season"])
    .fill_null(0.0).alias("snap_pct_s2d"),
).select(["season", "week", "player_display_name", "snap_pct_tr3", "snap_pct_s2d"])
n_snap = (feat.filter(pl.col("season") == 2025)
          .join(snaps, on=["season", "week", "player_display_name"], how="left")
          .filter(pl.col("snap_pct_tr3").is_not_null()).height)
n_2025 = feat.filter((pl.col("season") == 2025) & (pl.col("position").is_in(POS))).height
print(f"snap join: {n_snap}/{n_2025} 2025 rows matched "
      f"({100*n_snap/max(1,n_2025):.0f}%)")

# --- V2: vacated opportunity (Out/Doubtful same-position teammates, week W) ---
inj = (pl.read_parquet(REPO / "data/raw/injuries.parquet")
       .filter(pl.col("report_status").is_in(["Out", "Doubtful"]))
       .select(["season", "week", "team", "position", "gsis_id"])
       .cast({"season": pl.Int32, "week": pl.Int32})
       .unique())
vol = (feat.select(["season", "week", "player_id", "targets_tr3", "carries_tr3"])
       .unique().sort(["player_id", "season", "week"])
       .with_columns(pl.col(["targets_tr3", "carries_tr3"])
                     .cast(pl.Float64).fill_null(0.0)))
# Injured players often have no stats row in week W (they sat out). Backward
# asof: last available row at or before week W, same season — still
# strictly-prior information for the week-W decision.
j = (inj.rename({"gsis_id": "player_id"})
     .sort(["player_id", "season", "week"])
     .join_asof(vol, on="week", by=["player_id", "season"],
                strategy="backward"))
vac = (j.filter(pl.col("player_id").is_not_null()
               & pl.col("targets_tr3").is_not_null())
       .group_by(["season", "week", "team", "position"], maintain_order=True)
       .agg((pl.col("targets_tr3").sum().alias("vac_targets_tr3"),
             pl.col("carries_tr3").sum().alias("vac_carries_tr3"),
             pl.len().alias("n_vacated"))))
n_vac_2025 = vac.filter(pl.col("season") == 2025).height
print(f"vacated: {vac.height} non-zero (season,week,team,pos) groups total; "
      f"2025: {n_vac_2025}")

SNAP = ["snap_pct_tr3", "snap_pct_s2d"]
VAC = ["vac_targets_tr3", "vac_carries_tr3", "n_vacated"]

# ------------------------------------------------------------ split + arms
TGT = []
for _ts in POS_TARGETS.values():
    for _t in _ts:
        if _t not in TGT and _t not in BASE:
            TGT.append(_t)
df = (feat.filter(pl.col("position").is_in(POS))
      .select(["season", "week", "position", "player_id", "player_display_name", "team"]
              + BASE + TGT)
      .join(snaps, on=["season", "week", "player_display_name"], how="left")
      .join(vac, on=["season", "week", "team", "position"], how="left")
      .with_columns([pl.col(c).fill_null(0.0) for c in SNAP + VAC]))
train = df.filter(pl.col("season").is_in(list(range(2016, 2024))))
test = df.filter(pl.col("season") == 2025)
early = test.filter(pl.col("week") <= 4)
print(f"rows: train={train.height} test={test.height} early={early.height}")

def spear(p: np.ndarray, a: np.ndarray) -> float:
    if np.std(p) == 0 or np.std(a) == 0:
        return 0.0
    return float(spearmanr(p, a).statistic)

ARMS = {"V0": BASE,
        "V1": BASE + SNAP,
        "V2": BASE + VAC,
        "V3": BASE + SNAP + VAC}

results = []
for pos in POS:
    for target in POS_TARGETS[pos]:
        yte = test[target].to_numpy().astype(float)
        ye = early[target].to_numpy().astype(float)
        row = {"pos": pos, "target": target, "n": int(test.height)}
        for arm, cols in ARMS.items():
            Xtr = train.select(cols).to_numpy().astype(np.float32)
            Xte = test.select(cols).to_numpy().astype(np.float32)
            Xe = early.select(cols).to_numpy().astype(np.float32)
            p0 = payload["models"][f"{pos}/{target}"].get_params()
            m = HGBR(loss="squared_error", random_state=0,
                     n_iter_no_change=10, validation_fraction=0.1,
                     **{k: v for k, v in p0.items()
                        if k not in ("loss", "random_state", "n_iter_no_change",
                                     "validation_fraction", "quantile")})
            m.fit(Xtr, train[target].to_numpy().astype(float))
            p = m.predict(Xte)
            row[f"{arm}_mae"] = float(np.mean(np.abs(p - yte)))
            row[f"{arm}_rho"] = spear(p, yte)
            row[f"{arm}_rho_early"] = spear(m.predict(Xe), ye)
        results.append(row)

# ------------------------------------------------------------ report
def pooled(rows, arm, key):
    return np.mean([r[f"{arm}_{key}"] for r in rows])

print(f"\n{'arm':4} {'pooled MAE':>12} {'pooled rho':>12} {'early rho':>12}")
for arm in ARMS:
    print(f"{arm:4} {pooled(results, arm, 'mae'):>12.4f} "
          f"{pooled(results, arm, 'rho'):>12.4f} "
          f"{pooled(results, arm, 'rho_early'):>12.4f}")

print(f"\nper-stat (rho, V0 -> best arm):")
for r in results:
    arms = {a: r[f"{a}_rho"] for a in ARMS}
    best = max(arms, key=arms.get)
    wins = sum(1 for a in ("V1", "V2", "V3") if arms[a] > arms["V0"])
    print(f"  {r['pos']}/{r['target']:22} "
          f"V0={arms['V0']:+.4f} V1={arms['V1']:+.4f} "
          f"V2={arms['V2']:+.4f} V3={arms['V3']:+.4f}  best={best} "
          f"{'**' if wins else ''}")

out = {"generated": time.strftime("%Y-%m-%d %H:%M"),
       "split": "train 2016-23 | test 2025 (early = w1-4)",
       "arms": {a: len(c) for a, c in ARMS.items()},
       "snap_match_2025": [n_snap, n_2025],
       "vacated_groups_2025": n_vac_2025,
       "pooled": {a: {"mae": pooled(results, a, "mae"),
                      "rho": pooled(results, a, "rho"),
                      "rho_early": pooled(results, a, "rho_early")}
                  for a in ARMS},
       "per_stat": results}
(REPO / "models/ab_audit_report.json").write_text(json.dumps(out, indent=1),
                                                   encoding="utf-8")
print(f"\nwrote models/ab_audit_report.json  ({time.time()-T0:.0f}s)")
