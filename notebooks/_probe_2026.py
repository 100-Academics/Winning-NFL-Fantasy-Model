"""Probe 2026 availability in the nflverse source (non-destructive: temp dir).

Tells us which 2026 weeks are published so we know what the clean holdout can
target, WITHOUT clobbering the existing data/raw/ files.
"""
from pathlib import Path
import tempfile

import nflreadpy as nfl
import polars as pl

for label, fn, kw in [
    ("player_weekly", nfl.load_player_stats, {"summary_level": "week", "seasons": [2026]}),
    ("schedules",     nfl.load_schedules,     {"seasons": [2026]}),
    ("team_weekly",   nfl.load_team_stats,   {"summary_level": "week", "seasons": [2026]}),
]:
    try:
        df = fn(**kw)
        if "week" in df.columns:
            wk = sorted(df["week"].unique().to_list())
        else:
            wk = None
        st = df.get_column("season_type").unique().to_list() if "season_type" in df.columns else None
        print(f"{label:16s} rows={df.height:>7,}  weeks={wk}  season_type={st}")
    except Exception as e:
        print(f"{label:16s} ERR {e!r}")

# Also: what's the LATEST season the source has at all?
try:
    s = nfl.load_schedules(seasons=[2024, 2025, 2026])
    print("\nSchedules seasons present:", sorted(s["season"].unique().to_list()))
    for yr in [2025, 2026]:
        sub = s.filter(pl.col("season") == yr)
        print(f"  {yr}: weeks={sorted(sub['week'].unique().to_list())}  games={sub.height}")
except Exception as e:
    print("schedules multi ERR", repr(e))
