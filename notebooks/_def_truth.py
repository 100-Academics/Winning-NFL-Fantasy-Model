"""Minimal ground-truth check: what is the ACTUAL correlation between
(opponent Vegas implied team total) and ACTUAL D/ST fantasy points, 2024+2025?
Also dump a few concrete boards so I can eyeball whether the baseline is sane.
"""
import polars as pl
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sched = pl.read_parquet(REPO / "data/raw/schedules.parquet")
t = pl.read_parquet(REPO / "data/raw/team_weekly_stats.parquet")

TO_COLS = ["def_interceptions","def_fumbles","def_safeties","def_punt_blocks","def_pat_blocks","def_fg_blocks"]
t = t.select(["season","week","team"]+TO_COLS+["def_sacks","def_tds"])
for c in TO_COLS+["def_sacks","def_tds"]:
    t = t.with_columns(pl.col(c).fill_null(0.0))
t = t.with_columns((sum(pl.col(c) for c in TO_COLS)).alias("act_to"))

sc = sched.select(["season","week","away_team","away_score","home_team","home_score"])
sc = sc.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["away_score","home_score"]])
allowed = pl.concat([
    sc.select(["season","week","away_team","home_score"]).rename({"away_team":"team","home_score":"allowed"}),
    sc.select(["season","week","home_team","away_score"]).rename({"home_team":"team","away_score":"allowed"}),
], how="vertical")

sched = sched.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["spread_line","total_line"]])
sched = sched.with_columns(
    ((pl.col("total_line")-pl.col("spread_line"))/2).alias("away_imp"),
    ((pl.col("total_line")+pl.col("spread_line"))/2).alias("home_imp"))
oi = pl.concat([
    sched.select(["season","week","away_team","home_imp"]).rename({"away_team":"team","home_imp":"oi"}),
    sched.select(["season","week","home_team","away_imp"]).rename({"home_team":"team","away_imp":"oi"}),
], how="vertical")

def tier(a):
    return (pl.when(a<=6).then(10).when(a<=13).then(7).when(a<=17).then(4)
            .when(a<=23).then(1).when(a<=30).then(0).otherwise(-1))

panel = (t.join(allowed, on=["season","week","team"], how="left")
         .join(oi, on=["season","week","team"], how="left"))
panel = panel.with_columns(tier(pl.col("allowed")).alias("allowed_tier"))
panel = panel.with_columns(
    (pl.col("allowed_tier")+2*pl.col("act_to")+pl.col("def_sacks")+6*pl.col("def_tds")).alias("act_dst"))
panel = panel.filter(pl.col("season").is_in([2024,2025]) & pl.col("act_dst").is_not_null() & pl.col("oi").is_not_null())

print(f"\nn team-weeks = {len(panel)}")
for yr in [2024, 2025]:
    s = panel.filter(pl.col("season") == yr)
    print(f"  {yr}: n={len(s)}  Spearman(-oi vs act_dst) = {spearmanr(-s['oi'].to_numpy(), s['act_dst'].to_numpy())[0]:+.3f}")
print(f"corr(oi, act_dst)        = {spearmanr(panel['oi'], panel['act_dst'])[0]:+.3f}")
print(f"corr(-oi, act_dst)       = {spearmanr(-np.array(panel['oi']), np.array(panel['act_dst']))[0]:+.3f}")
print(f"corr(oi, allowed)        = {spearmanr(panel['oi'], panel['allowed'])[0]:+.3f}")
print(f"corr(oi, act_to)         = {spearmanr(panel['oi'], panel['act_to'])[0]:+.3f}")
print(f"corr(oi, def_sacks)      = {spearmanr(panel['oi'], panel['def_sacks'])[0]:+.3f}")

# mean D/ST points
print(f"\nmean act_dst = {panel['act_dst'].mean():.2f}  (min {panel['act_dst'].min()} max {panel['act_dst'].max()})")
print(f"mean oi      = {panel['oi'].mean():.2f}")

# eyeball a real week: 2025 week 4, sort by -oi (baseline best first)
w = panel.filter((pl.col("season")==2025)&(pl.col("week")==4)).sort("oi")
print("\n2025 W4 baseline board (lowest opp implied = 'best D' first):")
print(f"{'team':5}{'oi':>6}{'allowed':>8}{'to':>4}{'sack':>5}{'dTD':>4} | {'act_dst':>8}")
for r in w.to_dicts():
    print(f"{r['team']:5}{r['oi']:>6.1f}{r['allowed']:>8}{r['act_to']:>4.0f}{r['def_sacks']:>5.0f}{r['def_tds']:>4.0f} | {r['act_dst']:>8.0f}")
