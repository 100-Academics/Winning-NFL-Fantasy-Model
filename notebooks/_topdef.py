"""Week-4 (2026) top-defense (D/ST) board — ESPN STANDARD scoring.

METHOD (chosen by A/B, see notebooks/_def_ab.py, _def_grid.py, _def_truth.py):
  Rank D/ST by the OPPONENT's Vegas implied team total, LOWEST first.
  Points-allowed is the dominant term of ESPN D/ST scoring and is best
  predicted by how much the opponent is expected to score (Vegas). Own-defense
  trailing stats do NOT improve the ranking (best tuned model 0.055 Spearman
  << 0.345 baseline, on 2024+25 actuals), so they are not used for ranking.

  Point values shown = ESPN Standard:
    points-allowed tier on projected allowed (= opponent implied total)
      0-6:+10  7-13:+7  14-17:+4  18-23:+1  24-30:0  31+:-1
    + a flat league-mean expected sack count (~2.36, +1/sack) and a hard-shrunk
      turnover expectation (~1.0, +2/TO) for a realistic total. These are near-
      constants and do not change the RANKING (verified), only the absolute value.

Usage:  uv run python notebooks/_topdef.py [season] [week]   (default 2026 4)
"""
import sys
from pathlib import Path
import polars as pl
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
WEEK = int(sys.argv[2]) if len(sys.argv) > 2 else 4

def tier(a):
    if a <= 6: return 10
    if a <= 13: return 7
    if a <= 17: return 4
    if a <= 23: return 1
    if a <= 30: return 0
    return -1

t = pl.read_parquet(REPO / "data/raw/team_weekly_stats.parquet")
sched = pl.read_parquet(REPO / "data/raw/schedules.parquet")

# league/game sack + turnover means (2016-25) for the flat components
TO_COLS = ["def_interceptions", "def_fumbles", "def_safeties",
           "def_punt_blocks", "def_pat_blocks", "def_fg_blocks"]
hist = t.filter(pl.col("season").is_in(range(2016, 2026)))
for c in TO_COLS:
    hist = hist.with_columns(pl.col(c).fill_null(0.0))
SACKS_MEAN = float(hist["def_sacks"].fill_null(0.0).mean())
TO_MEAN = float(hist.select(pl.sum_horizontal(TO_COLS).alias("to"))["to"].mean())
print(f"league means used for flat components: sacks/game={SACKS_MEAN:.2f} TO/game={TO_MEAN:.2f}")

sched = sched.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["spread_line", "total_line"]])
sched = sched.with_columns(
    ((pl.col("total_line") - pl.col("spread_line")) / 2).alias("away_imp"),
    ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("home_imp"))
oi = pl.concat([
    sched.select(["season", "week", "away_team", "home_imp"]).rename({"away_team": "team", "home_imp": "proj_allowed"}),
    sched.select(["season", "week", "home_team", "away_imp"]).rename({"home_team": "team", "away_imp": "proj_allowed"}),
], how="vertical")

week = oi.filter((pl.col("season") == SEASON) & (pl.col("week") == WEEK) & pl.col("proj_allowed").is_not_null())
if len(week) == 0:
    print(f"No Vegas data for {SEASON} week {WEEK}.")
    sys.exit(1)

rows = []
for r in week.to_dicts():
    allowed = r["proj_allowed"]
    base = tier(allowed)
    total = base + 1.0 * SACKS_MEAN + 2.0 * TO_MEAN  # flat, ranking-invariant
    rows.append(dict(team=r["team"], proj_allowed=round(allowed, 1),
                     allowed_pts=base, d_st=round(total, 1)))
rows.sort(key=lambda x: (-x["d_st"], x["proj_allowed"]))

print(f"\n=== {SEASON} WEEK {WEEK}  D/ST board  (rank by opponent Vegas implied total, lowest=best) ===")
print(f"{'RK':>3} {'DEF':5} {'projAllwd':>10} {'allowedPts':>11} {'D/ST':>7}")
print("-" * 42)
for i, r in enumerate(rows, 1):
    mark = "  <= top pick" if i <= 5 else ""
    print(f"{i:>3} {r['team']:5} {r['proj_allowed']:>10} {r['allowed_pts']:>11} {r['d_st']:>7}{mark}")

# --- sanity check: correlation of this board's ranking vs actual D/ST on 2024+25 ---
sc = sched.select(["season", "week", "away_team", "away_score", "home_team", "home_score"])
sc = sc.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["away_score", "home_score"]])
allowed_act = pl.concat([
    sc.select(["season", "week", "away_team", "home_score"]).rename({"away_team": "team", "home_score": "allowed"}),
    sc.select(["season", "week", "home_team", "away_score"]).rename({"home_team": "team", "away_score": "allowed"}),
], how="vertical")
t2 = t.select(["season", "week", "team"] + TO_COLS + ["def_sacks", "def_tds"])
for c in TO_COLS + ["def_sacks", "def_tds"]:
    t2 = t2.with_columns(pl.col(c).fill_null(0.0))
t2 = t2.with_columns(pl.sum_horizontal(TO_COLS).alias("act_to"))
act = (t2.join(allowed_act, on=["season", "week", "team"], how="left")
       .with_columns((tier_expr := (pl.when(pl.col("allowed") <= 6).then(10).when(pl.col("allowed") <= 13).then(7)
                              .when(pl.col("allowed") <= 17).then(4).when(pl.col("allowed") <= 23).then(1)
                              .when(pl.col("allowed") <= 30).then(0).otherwise(-1))).alias("tier"))
       .with_columns((pl.col("tier") + 2 * pl.col("act_to") + pl.col("def_sacks") + 6 * pl.col("def_tds")).alias("act_dst")))
oi2 = pl.concat([
    sched.select(["season", "week", "away_team", "home_imp"]).rename({"away_team": "team", "home_imp": "oi"}),
    sched.select(["season", "week", "home_team", "away_imp"]).rename({"home_team": "team", "away_imp": "oi"}),
], how="vertical")
chk = act.join(oi2, on=["season", "week", "team"], how="left").filter(
    pl.col("season").is_in([2024, 2025]) & pl.col("act_dst").is_not_null() & pl.col("oi").is_not_null())
rho = spearmanr(-chk["oi"].to_numpy(), chk["act_dst"].to_numpy())[0]
print(f"\nA/B sanity (2024+25, n={len(chk)}): Spearman(opponent implied total, lowest first vs actual D/ST) = {rho:+.3f}")
print("  (this is the number that must beat any own-defense model; see _def_grid.py)")
