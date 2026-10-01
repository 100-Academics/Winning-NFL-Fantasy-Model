"""A/B: ESPN-Standard D/ST projection  vs  Vegas implied-team-total baseline.

The current live board (_topdef.py) is missing sacks, misses D/ST TDs, and
over-projects turnovers (2.1-3.4/game vs ~1.4 forced league-wide). This script:

1. Builds a CLEAN D/ST projection (ESPN Standard scoring:
   points-allowed tier + 2/turnover + 1/sack + 6/D-TD) with turnovers and
   D-TDs shrunk HARD toward the league mean, sacks shrunk moderately.
   Points-allowed is anchored on the opponent's Vegas implied team total
   (the dominant, week-1-available driver) + a small own-defense adjustment.
2. Builds the BASELINE the agent proposed: rank D/ST purely by the opponent's
   Vegas implied team total, lowest first.
3. Scores BOTH against ACTUAL D/ST fantasy points for every team-week in
   2024 and 2025 (the expert OK'd using 2025 for defense).
4. Decides: ship the model only if it beats the baseline on ranking quality;
   otherwise fall back to the baseline.

All "own" signals are strictly in-season trailing (weeks before W); week 1
falls back to the league mean. Opponent implied totals come from the Vegas
total_line/spread_line, available for every week.
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------- ESPN D/ST scoring
def tier(allowed):
    return (pl.when(allowed <= 6).then(10)
             .when(allowed <= 13).then(7)
             .when(allowed <= 17).then(4)
             .when(allowed <= 23).then(1)
             .when(allowed <= 30).then(0)
             .otherwise(-1))

def dst_points(allowed, to, sacks, dst_td):
    return tier(allowed) + 2.0 * to + 1.0 * sacks + 6.0 * dst_td

TO_COLS = ["def_interceptions", "def_fumbles", "def_safeties",
           "def_punt_blocks", "def_pat_blocks", "def_fg_blocks"]

t = pl.read_parquet(REPO / "data/raw/team_weekly_stats.parquet")
sched = pl.read_parquet(REPO / "data/raw/schedules.parquet")

# ---- actual D/ST points per team-week (for 2024 & 2025) ----
t = t.select(["season", "week", "team"] + TO_COLS + ["def_sacks", "def_tds"])
for c in TO_COLS + ["def_sacks", "def_tds"]:
    t = t.with_columns(pl.col(c).fill_null(0.0))
t = t.with_columns((
    pl.col(TO_COLS[0]) + pl.col(TO_COLS[1]) + pl.col(TO_COLS[2]) +
    pl.col(TO_COLS[3]) + pl.col(TO_COLS[4]) + pl.col(TO_COLS[5])
).alias("act_to"))
# actual points allowed = opponent's final score
sc = sched.select(["season", "week", "away_team", "away_score", "home_team", "home_score"])
sc = sc.with_columns(
    pl.col("away_score").fill_null(0.0).cast(pl.Float64),
    pl.col("home_score").fill_null(0.0).cast(pl.Float64))
allowed = pl.concat([
    sc.select(["season", "week", "away_team", "home_score"]).rename({"away_team": "team", "home_score": "act_allowed"}),
    sc.select(["season", "week", "home_team", "away_score"]).rename({"home_team": "team", "away_score": "act_allowed"}),
], how="vertical")
act = t.join(allowed, on=["season", "week", "team"], how="left")
act = act.with_columns(dst_points(pl.col("act_allowed"), pl.col("act_to"),
                                 pl.col("def_sacks"), pl.col("def_tds")).alias("act_dst"))

# ---- league-average shrinkage targets (population params, full history) ----
hist = act.filter(pl.col("season").is_in(range(2016, 2026)))
# restrict to weeks that actually have D stats (exclude any null act_dst)
hist = hist.filter(pl.col("act_dst").is_not_null())
n = len(hist)
LEAGUE = {
    "to": float(hist["act_to"].mean()),
    "sacks": float(hist["def_sacks"].mean()),
    "dst_td": float(hist["def_tds"].mean()),
    "allowed": float(hist["act_allowed"].mean()),
}
print(f"league/game targets (2016-25, n={n}): TO={LEAGUE['to']:.3f} sacks={LEAGUE['sacks']:.3f} "
      f"dstTD={LEAGUE['dst_td']:.4f} allowed={LEAGUE['allowed']:.2f}")

# ---- Vegas implied team totals (opponent's) ----
sched = sched.select(["season", "week", "away_team", "home_team", "spread_line", "total_line"])
sched = sched.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in ["spread_line", "total_line"]])
sched = sched.with_columns(
    ((pl.col("total_line") - pl.col("spread_line")) / 2).alias("away_implied"),
    ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("home_implied"))
opp_imp = pl.concat([
    sched.select(["season", "week", "away_team", "home_implied"]).rename({"away_team": "team", "home_implied": "opp_implied"}),
    sched.select(["season", "week", "home_team", "away_implied"]).rename({"home_team": "team", "away_implied": "opp_implied"}),
], how="vertical")
# coverage check
for yr in [2024, 2025]:
    cov = opp_imp.filter((pl.col("season") == yr) & (pl.col("opp_implied") > 0))
    print(f"  {yr} Vegas coverage: {len(cov)} team-weeks "
          f"({len(cov)/2:.0f} games) "
          f"avg opp implied {cov['opp_implied'].mean():.1f}")

# ---- own in-season trailing (weeks strictly before W) ----
def trail(df, key, col):
    df = df.sort([key, "season", "week"])
    return df.with_columns(pl.col(col).shift(1).over([key, "season"]).rolling_mean(3, min_samples=1)
                           .over([key, "season"]).alias(col + "_tr"))
def allowed_tab():
    a = allowed.clone().rename({"act_allowed": "pts"})
    return trail(a, "team", "pts").with_columns(pl.col("pts_tr").alias("own_allowed_tr")).select(["season", "week", "team", "own_allowed_tr"])
def to_tab():
    d = t.select(["season", "week", "team", "act_to"]).sort(["team", "season", "week"])
    return trail(d, "team", "act_to").with_columns(pl.col("act_to_tr").alias("own_to_tr")).select(["season", "week", "team", "own_to_tr"])
def sacks_tab():
    d = t.select(["season", "week", "team", "def_sacks"]).sort(["team", "season", "week"])
    return trail(d, "team", "def_sacks").with_columns(pl.col("def_sacks_tr").alias("own_sacks_tr")).select(["season", "week", "team", "own_sacks_tr"])
def dtd_tab():
    d = t.select(["season", "week", "team", "def_tds"]).sort(["team", "season", "week"])
    return trail(d, "team", "def_tds").with_columns(pl.col("def_tds_tr").alias("own_dtd_tr")).select(["season", "week", "team", "own_dtd_tr"])

base = (allowed_tab()
        .join(to_tab().select(["season","week","team","own_to_tr"]), on=["season","week","team"], how="left")
        .join(sacks_tab().select(["season","week","team","own_sacks_tr"]), on=["season","week","team"], how="left")
        .join(dtd_tab().select(["season","week","team","own_dtd_tr"]), on=["season","week","team"], how="left"))

panel = base.join(opp_imp, on=["season", "week", "team"], how="left").join(act, on=["season", "week", "team"], how="left")

# ---------------------------------------------------------------- projection
def tier_scalar(a):
    if a <= 6: return 10
    if a <= 13: return 7
    if a <= 17: return 4
    if a <= 23: return 1
    if a <= 30: return 0
    return -1

LAM = {"allowed_adj": 0.30, "to": 0.20, "sacks": 0.30, "td": 0.10}  # heavy shrink on TO/TD
def project(row):
    oi = row["opp_implied"] if row["opp_implied"] is not None else LEAGUE["allowed"]
    ow = lambda col, lg: (row[col] if row[col] is not None else lg)
    proj_allowed = oi + LAM["allowed_adj"] * (ow("own_allowed_tr", LEAGUE["allowed"]) - LEAGUE["allowed"])
    proj_to = max(0.0, LEAGUE["to"] + LAM["to"] * (ow("own_to_tr", LEAGUE["to"]) - LEAGUE["to"]))
    proj_sacks = max(0.0, LEAGUE["sacks"] + LAM["sacks"] * (ow("own_sacks_tr", LEAGUE["sacks"]) - LEAGUE["sacks"]))
    proj_td = max(0.0, LEAGUE["dst_td"] + LAM["td"] * (ow("own_dtd_tr", LEAGUE["dst_td"]) - LEAGUE["dst_td"]))
    return tier_scalar(proj_allowed) + 2.0 * proj_to + 1.0 * proj_sacks + 6.0 * proj_td

for yr in [2024, 2025]:
    sub = panel.filter((pl.col("season") == yr) & pl.col("act_dst").is_not_null()).to_dicts()
    for r in sub:
        r["proj"] = project(r)
        r["base"] = -r["opp_implied"]  # rank: lowest opp implied total first
    df = pl.DataFrame(sub)

    def spearman(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if np.std(a) == 0 or np.std(b) == 0: return float("nan")
        return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
    def topk_overlap(pred_pts, act_pts, k):
        n = len(pred_pts)
        pred_top = set(np.argsort(pred_pts)[::-1][:k])
        act_top = set(np.argsort(act_pts)[::-1][:k])
        return len(pred_top & act_top) / k
    def mean_actual_rank_of_top(pred_pts, act_pts, k):
        order = np.argsort(pred_pts)[::-1][:k]  # projected top-k
        ranks = np.argsort(np.argsort(act_pts))[::-1] + 1  # actual rank (1=best)
        return float(ranks[order].mean())

    ap = df["act_dst"].to_numpy(); pr = df["proj"].to_numpy(); ba = df["base"].to_numpy()
    print(f"\n=== {yr}  (n={len(df)} team-weeks) ===")
    print(f"{'variant':12}{'Spearman':>10}{'top10%':>8}{'top5%':>8}{'meanRank(top5)':>16}")
    for name, pts in [("MODEL", pr), ("BASELINE", ba)]:
        print(f"{name:12}{spearman(pts, ap):>10.3f}{topk_overlap(pts, ap, 10):>8.3f}"
              f"{topk_overlap(pts, ap, 5):>8.3f}{mean_actual_rank_of_top(pts, ap, 5):>16.1f}")

    # quick look: model turnover/sack/allowed projections distribution
    print(f"  MODEL proj: allowed~{np.mean([project(r) for r in sub if r.get('own_to_tr') is not None]) if False else 'n/a'} | "
          f"proj_TO mean={LEAGUE['to']:.2f} proj_SACKS mean={LEAGUE['sacks']:.2f} (shrinkage targets)")
    print(f"  BASELINE opp_implied mean={df['opp_implied'].mean():.1f}  (range {df['opp_implied'].min():.1f}-{df['opp_implied'].max():.1f})")
