"""One-off schema exploration for Phase 3 (cleaning + feature design).

Run:  uv run python notebooks/explore_phase3.py
Prints the shape/keys/distributions of every raw table we'll join on, so the
feature design is grounded in the real columns rather than assumptions.
"""
from __future__ import annotations
import polars as pl
import pathlib

RAW = pathlib.Path("data/raw")

def head(name: str, *cols: str) -> None:
    df = pl.read_parquet(RAW / f"{name}.parquet")
    print("=" * 78)
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    if cols:
        show = [c for c in cols if c in df.columns]
        print(df.select(show).head(3))
    print("columns:", df.columns)
    if "season" in df.columns:
        print("seasons:", sorted(df["season"].unique().to_list()))
    print()

# ---- core player panel ----------------------------------------------------
p = pl.read_parquet(RAW / "player_weekly_stats.parquet")
print("=" * 78)
print("player_weekly_stats:", p.shape, "seasons", sorted(p["season"].unique().to_list()))
print("week range:", p["week"].min(), "-", p["week"].max())
print("season_type values:", p["season_type"].unique().to_list() if "season_type" in p.columns else "n/a")
print("position values:", p["position"].unique().to_list())
print("position_group values:", p["position_group"].unique().to_list() if "position_group" in p.columns else "n/a")
print("player_id nulls:", p["player_id"].is_null().sum())
print("team nulls:", p["team"].is_null().sum())
print("opponent_team nulls:", p["opponent_team"].is_null().sum())
# duplicate (player,season,week)?
print("dup (player_id,season,week) count:", p.group_by(["player_id", "season", "week"]).len()
      .filter(pl.col("len") > 1).height)
# how many weeks per season for a QB (byes / multi-game weeks?)
print("weeks/season (all):")
print(p.group_by("season").agg(pl.col("week").max()).sort("season"))
print("max week per season:", p.group_by("season").agg(pl.col("week").max()).sort("season"))
# byes: a given player's weeks in a season (should be ~17)
mah = p.filter(pl.col("player_name").str.contains("Mahomes", literal=True) & (pl.col("season") == 2024))
print("Mahomes 2024 weeks present:", sorted(mah["week"].unique().to_list()), "count", mah.height)

# ---- team weekly stats (for opponent defense + team strength) -------------
t = pl.read_parquet(RAW / "team_weekly_stats.parquet")
print("=" * 78)
print("team_weekly_stats:", t.shape)
print("has 'opponent' col?", [c for c in t.columns if "opp" in c.lower()])
print("cols sample:", t.columns[:40])

# ---- schedules (Vegas / weather / rest) ------------------------------------
s = pl.read_parquet(RAW / "schedules.parquet")
print("=" * 78)
print("schedules:", s.shape, "seasons", sorted(s["season"].unique().to_list()))
print("week range:", s["week"].min(), "-", s["week"].max())
print("sample 2025 wk1:")
print(s.filter((pl.col("season") == 2025) & (pl.col("week") == 1)).head(3).select(
    ["game_id", "away_team", "home_team", "away_spread_line" if "away_spread_line" in s.columns else "spread_line",
     "total_line", "away_rest", "home_rest", "temp", "wind", "roof"]))
print("spread_line nulls:", s["spread_line"].is_null().sum(), "of", s.height)
print("total_line nulls:", s["total_line"].is_null().sum(), "of", s.height)
print("temp nulls:", s["temp"].is_null().sum(), "wind nulls:", s["wind"].is_null().sum())

# ---- snap counts ----------------------------------------------------------
sc = pl.read_parquet(RAW / "snap_counts.parquet")
print("=" * 78)
print("snap_counts:", sc.shape)
print("cols:", sc.columns)
print(sc.head(2))

# ---- injuries -------------------------------------------------------------
inj = pl.read_parquet(RAW / "injuries.parquet")
print("=" * 78)
print("injuries:", inj.shape)
print("cols:", inj.columns)
print(inj.head(2))

# ---- participation --------------------------------------------------------
par = pl.read_parquet(RAW / "participation.parquet")
print("=" * 78)
print("participation:", par.shape)
print("cols:", par.columns)

# ---- nextgen --------------------------------------------------------------
for ng in ["nextgen_passing", "nextgen_receiving", "nextgen_rushing"]:
    g = pl.read_parquet(RAW / f"{ng}.parquet")
    print(f"{ng}: {g.shape} cols={g.columns}")

print("DONE")
