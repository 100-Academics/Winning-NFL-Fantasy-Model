"""A/B: prior-season bridge (regression fix) on top of the current feature set.

The bridge was documented in NOTES.md but its columns were silently dropped
before the join (see src/features.py _prev_season_bridge). Now fixed; this
tests whether adding the 7 bridge columns to the current 85-feature set
helps on the clean 2025 holdout (train 2016-2023, same hyperparameters as
the shipped models). Arms:
  A  current 85 features (snap share + vacated opportunity, no bridge)
  B  + prev_season_* bridge (7 cols)
Decision rule: adopt B only if pooled Spearman beats A by > +0.01.
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

import src.features as F

panel = F.build_panel(F.RAW_DIR)
team = F.load_team_stats(F.RAW_DIR)
ngs = F.load_nextgen(F.RAW_DIR)
usage = F.load_usage(F.RAW_DIR)
snaps = F.load_snap_counts(F.RAW_DIR)
inj = F.load_injury_reports(F.RAW_DIR)

BRIDGE = [f"prev_season_{s}_pg" for s in F.PREV_SEASON_STATS] + ["prev_season_games"]
payload = joblib.load(REPO / "models/models.joblib")
POS_TARGETS: dict[str, list[str]] = payload["meta"]["pos_targets"]
F_KEY_COLS = ["player_id", "player_display_name", "season", "week", "team",
              "opponent_team", "game_id", "position", "position_group"]
ALL_TARGETS = sorted({t for ts in POS_TARGETS.values() for t in ts})
POS = ["QB", "RB", "WR", "TE"]

feat = F.build_features_from(panel, team, ngs, usage, snaps=snaps, injuries=inj)
BASE = [c for c in feat.columns
        if c not in F_KEY_COLS and c not in ALL_TARGETS]

train = feat.filter(pl.col("season").is_in(list(range(2016, 2024)))
                    & pl.col("position").is_in(POS))
test = feat.filter((pl.col("season") == 2025) & pl.col("position").is_in(POS))
early = test.filter(pl.col("week") <= 4)
print(f"rows: train={train.height} test={test.height} early={early.height} "
      f"feats A={len(BASE)} B={len(BASE)+len(BRIDGE)}")


def spear(p, a) -> float:
    if np.std(p) == 0 or np.std(a) == 0:
        return 0.0
    return float(spearmanr(p, a).statistic)


rows = []
for pos in POS:
    for target in POS_TARGETS[pos]:
        yte = test[target].to_numpy().astype(float)
        ye = early[target].to_numpy().astype(float)
        p0 = payload["models"][f"{pos}/{target}"].get_params()
        common = dict(loss="squared_error", random_state=0, n_iter_no_change=10,
                      validation_fraction=0.1,
                      **{k: v for k, v in p0.items()
                         if k not in ("loss", "random_state", "n_iter_no_change",
                                      "validation_fraction", "quantile")})
        out = {"pos": pos, "target": target}
        no_bridge = [c for c in BASE if c not in BRIDGE]
        for arm, cols in (("A", no_bridge), ("B", BASE)):
            m = HGBR(**common)
            m.fit(train.select(cols).to_numpy().astype(np.float32),
                  train[target].to_numpy().astype(float))
            out[f"{arm}_mae"] = float(np.mean(np.abs(m.predict(
                test.select(cols).to_numpy().astype(np.float32)) - yte)))
            out[f"{arm}_rho"] = spear(m.predict(
                test.select(cols).to_numpy().astype(np.float32)), yte)
            out[f"{arm}_rho_early"] = spear(m.predict(
                early.select(cols).to_numpy().astype(np.float32)), ye)
        rows.append(out)

def pooled(key):
    return float(np.mean([r[key] for r in rows]))

print(f"\n{'arm':4} {'pooled MAE':>12} {'pooled rho':>12} {'early rho':>12}")
for arm in ("A", "B"):
    print(f"{arm:4} {pooled(arm+'_mae'):>12.4f} {pooled(arm+'_rho'):>12.4f} "
          f"{pooled(arm+'_rho_early'):>12.4f}")
print(f"\nper-stat rho (A -> B):")
for r in rows:
    d = r["B_rho"] - r["A_rho"]
    print(f"  {r['pos']}/{r['target']:22} A={r['A_rho']:+.4f} B={r['B_rho']:+.4f}  d={d:+.4f}")

(REPO / "models/ab_bridge_report.json").write_text(json.dumps(
    {"generated": time.strftime("%Y-%m-%d %H:%M"),
     "arms": {"A": len(BASE) - len(BRIDGE), "B": len(BASE)},
     "pooled": {a: {"mae": pooled(a + "_mae"), "rho": pooled(a + "_rho"),
                    "rho_early": pooled(a + "_rho_early")} for a in ("A", "B")},
     "per_stat": rows}, indent=1), encoding="utf-8")
print(f"\nwrote models/ab_bridge_report.json  ({time.time()-T0:.0f}s)")
