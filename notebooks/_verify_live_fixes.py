"""Verify the B1 (snap) + B2 (prior-season bridge) fixes on the LIVE scheduled
panel path.

Before the fix, on any unplayed target week:
  * snap_pct_tr3 / snap_pct_s2d   -> 100% null (read as 0.0)
  * prev_season_* (7 columns)     -> 100% null (read as 0.0)
because _build_scheduled_panel passed `snaps` raw and filtered the panel to the
current season only (so _prev_season_bridge had no prior-season rows to
aggregate).

After the fix both should be populated for the players who have the underlying
history. Run:  uv run python notebooks/_verify_live_fixes.py
"""
from __future__ import annotations

import polars as pl

from src.holdout import _build_scheduled_panel
from src.clean import PROCESSED_DIR

SEASON, WEEK = 2026, 12   # an unplayed week -> scheduled-panel path

panel = _build_scheduled_panel(SEASON, WEEK)
print(f"scheduled panel {SEASON} w{WEEK}: {panel.height} rows x {panel.width} cols")

snap_cols = ["snap_pct_tr3", "snap_pct_s2d"]
prev_cols = [c for c in panel.columns if c.startswith("prev_season_")]
print("prev_season cols present:", prev_cols)

def report(cols, label):
    print(f"\n=== {label} ===")
    for c in cols:
        if c not in panel.columns:
            print(f"  {c:<22} MISSING")
            continue
        s = panel[c]
        n_null = s.is_null().sum()
        n_zero = (s == 0.0).sum()          # nulls compare null -> not counted
        nnz = s.is_not_null().sum() - n_zero
        print(f"  {c:<22} null={n_null:>5}  zero={n_zero:>5}  non-zero={nnz:>5}")

report(snap_cols, "SNAP SHARES (B1)")
report(prev_cols, "PRIOR-SEASON BRIDGE (B2)")

# Cross-check: compute the EXPECTED values straight from the raw tables and
# confirm the live panel matches them (not just "non-zero"). Pick a QB who has
# BOTH a live snap share and a live prior-season passing value.
from src.clean import RAW_DIR
cand = (panel.filter((pl.col("position") == "QB")
                     & pl.col("snap_pct_tr3").is_not_null()
                     & pl.col("prev_season_passing_yards_pg").is_not_null()))
if cand.height:
    name = cand["player_display_name"][0]
    live = cand.to_dicts()[0]
    sc = (pl.read_parquet(RAW_DIR / "snap_counts.parquet")
          .filter((pl.col("game_type") == "REG") & (pl.col("season") == 2026)
                  & (pl.col("player") == name)))
    pw_raw = (pl.read_parquet(RAW_DIR / "player_weekly_stats.parquet")
              .filter((pl.col("season") == 2025) & (pl.col("player_display_name") == name)))
    recent = sc.filter(pl.col("week") < 12).sort("week").tail(3)["offense_pct"].to_list()
    exp_snap = sum(recent) / len(recent)
    exp_prev = float(pw_raw["passing_yards"].mean())
    print(f"\nCross-check ({name}, a veteran):")
    print(f"  snap_pct_tr3          live={live['snap_pct_tr3']:.6f}   expected={exp_snap:.6f}  "
          f"match={abs(live['snap_pct_tr3']-exp_snap) < 1e-6}")
    print(f"  prev_season_passing_yds_pg  live={live['prev_season_passing_yards_pg']:.4f}   "
          f"expected={exp_prev:.4f}  match={abs(live['prev_season_passing_yards_pg']-exp_prev) < 1e-6}")
