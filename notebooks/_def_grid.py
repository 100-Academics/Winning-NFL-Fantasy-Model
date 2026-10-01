"""Grid-search the D/ST shrinkage weights, then compare the best model to the
Vegas-implied-team-total baseline on 2024+2025 actual D/ST points.

Goal: pick weights that maximize pooled ranking quality (Spearman), then decide
whether the model beats the baseline. If not, ship the baseline.
"""
import sys
from pathlib import Path
import itertools
import numpy as np, polars as pl
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TO_COLS = ["def_interceptions", "def_fumbles", "def_safeties",
           "def_punt_blocks", "def_pat_blocks", "def_fg_blocks"]

def tier_scalar(a):
    if a <= 6: return 10
    if a <= 13: return 7
    if a <= 17: return 4
    if a <= 23: return 1
    if a <= 30: return 0
    return -1

t = pl.read_parquet(REPO / "data/raw/team_weekly_stats.parquet")
sched = pl.read_parquet(REPO / "data/raw/schedules.parquet")
t = t.select(["season", "week", "team"] + TO_COLS + ["def_sacks", "def_tds"])
for c in TO_COLS + ["def_sacks", "def_tds"]:
    t = t.with_columns(pl.col(c).fill_null(0.0))
t = t.with_columns((pl.col(TO_COLS[0]) + pl.col(TO_COLS[1]) + pl.col(TO_COLS[2]) +
                   pl.col(TO_COLS[3]) + pl.col(TO_COLS[4]) + pl.col(TO_COLS[5])).alias("act_to"))
sc = sched.select(["season", "week", "away_team", "away_score", "home_team", "home_score"])
sc = sc.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["away_score", "home_score"]])
allowed = pl.concat([
    sc.select(["season", "week", "away_team", "home_score"]).rename({"away_team": "team", "home_score": "pts"}),
    sc.select(["season", "week", "home_team", "away_score"]).rename({"home_team": "team", "away_score": "pts"}),
], how="vertical")

sched = sched.select(["season", "week", "away_team", "home_team", "spread_line", "total_line"])
sched = sched.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["spread_line", "total_line"]])
opp_imp = pl.concat([
    # team = AWAY -> opponent is HOME, whose implied total is (total + spread)/2
    (sched.with_columns(((pl.col("total_line") + pl.col("spread_line")) / 2).alias("home_implied"))
         .select(["season", "week", "away_team", "home_implied"]).rename({"away_team": "team", "home_implied": "oi"})),
    # team = HOME -> opponent is AWAY, whose implied total is (total - spread)/2
    (sched.with_columns(((pl.col("total_line") - pl.col("spread_line")) / 2).alias("away_implied"))
         .select(["season", "week", "home_team", "away_implied"]).rename({"home_team": "team", "away_implied": "oi"})),
], how="vertical")

def trail(df, key, col):
    df = df.sort([key, "season", "week"])
    return df.with_columns(pl.col(col).shift(1).over([key, "season"]).rolling_mean(3, min_samples=1).over([key, "season"]).alias(col + "_tr"))

panels = []
panels.append(trail(allowed.rename({"pts": "own_allowed"}), "team", "own_allowed").select(["season","week","team","own_allowed_tr"]))
panels.append(trail(t.select(["season","week","team","act_to"]).rename({"act_to":"own_to"}), "team", "own_to").select(["season","week","team","own_to_tr"]))
panels.append(trail(t.select(["season","week","team","def_sacks"]).rename({"def_sacks":"own_sacks"}), "team", "own_sacks").select(["season","week","team","own_sacks_tr"]))
panels.append(trail(t.select(["season","week","team","def_tds"]).rename({"def_tds":"own_dtd"}), "team", "own_dtd").select(["season","week","team","own_dtd_tr"]))
base = panels[0]
for p in panels[1:]:
    base = base.join(p.select(["season","week","team", [c for c in p.columns if c.endswith("_tr")][0]]), on=["season","week","team"], how="left")

hist = allowed.join(t, on=["season","week","team"], how="left").filter(pl.col("season").is_in(range(2016,2026)))
LEAGUE = {
    "to": float(t.filter(pl.col("season").is_in(range(2016,2026)))["act_to"].mean()),
    "sacks": float(t.filter(pl.col("season").is_in(range(2016,2026)))["def_sacks"].mean()),
    "td": float(t.filter(pl.col("season").is_in(range(2016,2026)))["def_tds"].mean()),
    "allowed": float(allowed.filter(pl.col("season").is_in(range(2016,2026)))["pts"].mean()),
}
print(f"league/game: TO={LEAGUE['to']:.3f} sacks={LEAGUE['sacks']:.3f} dTD={LEAGUE['td']:.4f} allowed={LEAGUE['allowed']:.2f}")

panel = base.join(opp_imp, on=["season","week","team"], how="left").join(
    allowed.select(["season","week","team","pts"]).rename({"pts":"act_allowed"}), on=["season","week","team"], how="left").join(
    t.select(["season","week","team","act_to","def_sacks","def_tds"]), on=["season","week","team"], how="left")
for yr in [2024, 2025]:
    panel = panel.filter(pl.col("season") != yr) if False else panel
panel = panel.filter(pl.col("season").is_in([2024, 2025]) & pl.col("act_allowed").is_not_null())
print(f"panel rows: {len(panel)}")

rows = panel.to_dicts()

def project(r, lam):
    oi = r["oi"] if r["oi"] is not None else LEAGUE["allowed"]
    ow = lambda col, lg: (r[col] if r[col] is not None else lg)
    pa = oi + lam["allowed"] * (ow("own_allowed_tr", LEAGUE["allowed"]) - LEAGUE["allowed"])
    pt = max(0.0, LEAGUE["to"] + lam["to"] * (ow("own_to_tr", LEAGUE["to"]) - LEAGUE["to"]))
    ps = max(0.0, LEAGUE["sacks"] + lam["sacks"] * (ow("own_sacks_tr", LEAGUE["sacks"]) - LEAGUE["sacks"]))
    pd = max(0.0, LEAGUE["td"] + lam["td"] * (ow("own_dtd_tr", LEAGUE["td"]) - LEAGUE["td"]))
    return tier_scalar(pa) + 2.0 * pt + ps + 6.0 * pd

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

act = np.array([r["act_allowed"] and 0 for r in rows])  # placeholder
act_pts = np.array([
    tier_scalar(r["act_allowed"]) + 2*r["act_to"] + r["def_sacks"] + 6*r["def_dtd"] if False
    else tier_scalar(r["act_allowed"]) + 2*r["act_to"] + r["def_sacks"] + 6*r["def_tds"]
    for r in rows])
base_pts = np.array([-(r["oi"] if r["oi"] is not None else LEAGUE["allowed"]) for r in rows])

GRID = list(itertools.product(
    [0.0, 0.2, 0.4, 0.6, 0.8],   # lam_allowed
    [0.0, 0.2, 0.5, 1.0],        # lam_to
    [0.0, 0.3, 0.6, 1.0],        # lam_sacks
    [0.0, 0.3, 1.0],             # lam_td
))
print(f"grid: {len(GRID)} configs")

best = None
results = []
for la, lt, ls, ld in GRID:
    lam = {"allowed": la, "to": lt, "sacks": ls, "td": ld}
    proj = np.array([project(r, lam) for r in rows])
    sp = spearman(proj, act_pts)
    sp_base = spearman(base_pts, act_pts)
    results.append((la, lt, ls, ld, sp))

results.sort(key=lambda x: x[4], reverse=True)
print("\nTOP 10 MODELS (pooled 2024+25 Spearman):")
print(f"{'lamA':>5}{'lamTO':>6}{'lamSA':>6}{'lamTD':>6}{'Spearman':>10}")
for la, lt, ls, ld, sp in results[:10]:
    print(f"{la:>5.1f}{lt:>6.1f}{ls:>6.1f}{ld:>6.1f}{sp:>10.3f}")

# baseline
sp_base = spearman(base_pts, act_pts)
print(f"\nBASELINE (opponent Vegas implied total, lowest first): Spearman = {sp_base:.3f}")
print(f"BEST MODEL:                                          Spearman = {results[0][4]:.3f}")
delta = results[0][4] - sp_base
print(f"Δ (model − baseline) = {delta:+.3f}")
print(f"\nDecision: {'SHIP MODEL' if delta > 0.01 else 'USE BASELINE'} (threshold +0.01)")
print(f"  best weights: λ_allowed={results[0][0]} λ_to={results[0][1]} λ_sacks={results[0][2]} λ_td={results[0][3]}")
