"""A/B: does adding a PRIOR-SEASON defense-quality prior beat the pure
opponent-implied-total (matchup) baseline for ranking D/ST?

- Baseline (B): rank by opponent's Vegas implied team total, lowest first.
- Blend (M): rank by  matchup + w * (prior-season defense points-allowed rank).
  prior season = 2023 for 2024, 2024 for 2025 (no lookahead).
- Score both vs ACTUAL D/ST points, per year + pooled, on 2024 & 2025.

If M does NOT clearly beat B, ship B (the expert's "HOU/DEN are elite" is a
team-quality prior the data says is a weak predictor vs the matchup).
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sched = pl.read_parquet(REPO / "data/raw/schedules.parquet")
t = pl.read_parquet(REPO / "data/raw/team_weekly_stats.parquet")
TO = ["def_interceptions", "def_fumbles", "def_safeties", "def_punt_blocks", "def_pat_blocks", "def_fg_blocks"]

t = t.select(["season", "week", "team"] + TO + ["def_sacks", "def_tds"])
for c in TO + ["def_sacks", "def_tds"]:
    t = t.with_columns(pl.col(c).fill_null(0.0))
t = t.with_columns(pl.sum_horizontal(TO).alias("act_to"))

sc = sched.select(["season", "week", "away_team", "away_score", "home_team", "home_score"]).with_columns(
    [pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["away_score", "home_score"]])
al = pl.concat([
    sc.select(["season", "week", "away_team", "home_score"]).rename({"away_team": "team", "home_score": "allowed"}),
    sc.select(["season", "week", "home_team", "away_score"]).rename({"home_team": "team", "away_score": "allowed"}),
], how="vertical")
sched = sched.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["spread_line", "total_line"]])
sched = sched.with_columns(((pl.col("total_line") - pl.col("spread_line")) / 2).alias("a"),
                           ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("h"))
oi = pl.concat([
    sched.select(["season", "week", "away_team", "h"]).rename({"away_team": "team", "h": "oi"}),
    sched.select(["season", "week", "home_team", "a"]).rename({"home_team": "team", "a": "oi"}),
], how="vertical")

def tier(a):
    return (pl.when(a <= 6).then(10).when(a <= 13).then(7).when(a <= 17).then(4)
            .when(a <= 23).then(1).when(a <= 30).then(0).otherwise(-1))
p = t.join(al, on=["season", "week", "team"], how="left").join(oi, on=["season", "week", "team"], how="left")
p = p.with_columns((tier(pl.col("allowed")) + 2 * pl.col("act_to") + pl.col("def_sacks") + 6 * pl.col("def_tds")).alias("act_dst"))

# prior-season defense points-allowed (mean per game) for each team
al_full = (al.select(["season", "team", "allowed"])
           .group_by(["season", "team"]).agg(pl.col("allowed").mean().alias("allowed_prior")))
print("prior-season defense quality available for:", al_full.select(pl.col("season").unique().sort()).to_series().to_list())

def prior_rank(season):
    # lower allowed = better defense -> higher "quality" score (1..N)
    d = al_full.filter(pl.col("season") == season).sort("allowed_prior")
    n = len(d)
    return dict(zip(d["team"], range(n, 0, -1)))  # best= n, worst = 1

P = {2024: prior_rank(2023), 2025: prior_rank(2024)}

def z(x):
    x = np.asarray(x, float)
    s = np.std(x)
    return (x - x.mean()) / s if s else np.zeros_like(x)

print(f"\n{'year':6}{'n':>6} | {'BASE(-oi)':>12} | blend w=0.25  0.5  0.75  1.0")
pooled = {"B": [], "M": []}
for yr in [2024, 2025]:
    s = p.filter((pl.col("season") == yr) & pl.col("act_dst").is_not_null() & pl.col("oi").is_not_null())
    act = s["act_dst"].to_numpy()
    oi_ = s["oi"].to_numpy()
    pr = np.array([P[yr].get(tm, 16) for tm in s["team"]])
    rB = spearmanr(-oi_, act)[0]
    pooled["B"].append(rB)
    line = f"{yr:6}{len(s):>6} | {rB:>12.3f} |"
    for w in [0.25, 0.5, 0.75, 1.0]:
        score = -z(oi_) + w * z(pr)
        rM = spearmanr(score, act)[0]
        pooled["M"].append(rM)
        line += f" {rM:>+7.3f}"
    print(line)
print(f"\nPOOLED: BASELINE = {np.mean(pooled['B']):+.3f}   (best blend shown above per year)")

# Which teams does the expert want? show their baseline rank on 2026 W4 is not in this
# (this is 2024/25). Instead show: on 2025, where do HOU/BAL/DEN rank by baseline vs by blend?
print("\n--- 2025: expert teams (HOU BAL DEN PHI) vs baseline rank ---")
s = p.filter((pl.col("season") == 2025) & pl.col("act_dst").is_not_null() & pl.col("oi").is_not_null())
act = s["act_dst"].to_numpy(); oi_ = s["oi"].to_numpy()
pr = np.array([P[2025].get(tm, 16) for tm in s["team"]])
order_B = np.argsort(oi_)  # lowest oi first = best baseline
order_M = np.argsort(-(-z(oi_) + 0.5 * z(pr)))  # blend w=0.5
names = s["team"].to_list()
for tm in ["HOU", "BAL", "DEN", "PHI", "MIN", "ARI", "NO", "CHI", "PIT"]:
    if tm in names:
        i = names.index(tm)
        print(f"  {tm}: baseline_rank={order_B.tolist().index(i)+1:2d}  blend_rank={order_M.tolist().index(i)+1:2d}  act_dst={act[i]:.0f}")
