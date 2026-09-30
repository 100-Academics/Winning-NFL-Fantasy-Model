"""Q1 A/B: does blending the model's week prediction toward the prior-season
per-game rate (a veteran prior) improve ranking? Tune on the clean 2025 test
holdout. Per position: raw model vs blend at K in {1,2,3,5,8,12}.

Blend: final = w*model + (1-w)*prior, w = games/(games+K).
  - games = player's games played EARLIER in the season (before this week).
  - prior = prior season's per-game rate for that stat (0 if no prior games).
  - only applied for stats the player has a prior for (veterans).
ESPN-standard PPR for the ranking metric (0.04/pyd 4/pTD -2/INT; 0.1/6 no PPR).
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import joblib

payload = joblib.load(REPO/"models/models.joblib")
fc = payload["meta"]["feature_columns"]
models = payload["models"]
POS_TARGETS = payload["meta"]["pos_targets"]

ESPN = {"passing_yards":0.04,"rushing_yards":0.1,"receiving_yards":0.1,
        "passing_tds":4,"rushing_tds":6,"receiving_tds":6,"passing_interceptions":-2}
def pts(d): return sum(ESPN.get(k,0)*v for k,v in d.items())
def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.std(a)==0 or np.std(b)==0: return float("nan")
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def mae(a,b): return float(np.abs(np.asarray(a)-np.asarray(b)).mean())

# prior-season (2024) per-game rates for 2025 players, per target stat
pw = pl.read_parquet(REPO/"data/raw/player_weekly_stats.parquet")
def prior_rates(season):
    d = pw.filter(pl.col("season")==season)
    stats = {s for ts in POS_TARGETS.values() for s in ts}
    stats &= set(d.columns)
    g = d.group_by(["player_id"]).agg([pl.len().alias("_g"),
                                      *[pl.col(s).sum().alias(s) for s in stats]])
    out={}
    for r in g.to_dicts():
        n=r["_g"]; out[r["player_id"]]={s: float(r[s] or 0)/n for s in stats}
    return out
PRIOR = prior_rates(2024)

feat = pl.read_parquet(REPO/"data/processed"/"features.parquet")
f25 = feat.filter(pl.col("season")==2025).sort(["player_id","week"])
# game sequence number (1-based) within each player's 2025 season
f25 = f25.with_columns(pl.col("week").rank().cast(pl.Int64).over("player_id").alias("_seq"))

rows = f25.to_dicts()
Ks = [1,2,3,5,8,12]
# accumulate per position: model_pts, blend_pts[K], actual_pts
acc = {pos: {"model":[], "actual":[], **{f"k{k_}":[] for k_ in Ks}} for pos in POS_TARGETS}
# also an early-season-only accumulator (games_played_s2d <= 4) where shrinkage bites
acc_e = {pos: {"model":[], "actual":[], **{f"k{k_}":[] for k_ in Ks}} for pos in POS_TARGETS}
EARLY_MAX = 4

for r in rows:
    pos=r["position"]; tgt=POS_TARGETS.get(pos,[])
    if not tgt: continue
    rec={k:v for k,v in r.items()}
    X=np.array([0.0 if rec.get(c) is None else float(rec.get(c)) for c in fc],dtype=np.float32).reshape(1,-1)
    model_pred={}; actual={}
    for t in tgt:
        m=models.get(f"{pos}/{t}")
        if m is not None:
            v=float(m.predict(X)[0])
            if t in {"passing_tds","rushing_tds","receiving_tds","receptions","passing_interceptions"} and v<0: v=0.0
            model_pred[t]=v
        a=r.get(t)
        actual[t]=float(a) if a is not None else 0.0
    # games played earlier this season (season-to-date, prior-only feature already in panel)
    gpd = r.get("games_played_s2d")
    games = int(gpd) if gpd is not None else max(1, int(r["_seq"])-1)
    games = max(0, games)
    prior = PRIOR.get(r["player_id"], {})
    model_pts=pts(model_pred); act_pts=pts(actual)
    acc[pos]["model"].append(model_pts); acc[pos]["actual"].append(act_pts)
    if games <= EARLY_MAX:
        acc_e[pos]["model"].append(model_pts); acc_e[pos]["actual"].append(act_pts)
    for k_ in Ks:
        w=games/(games+k_)
        blend={}
        for t in tgt:
            p=prior.get(t)
            blend[t] = (w*model_pred[t] + (1-w)*p) if (t in model_pred and p is not None) else model_pred.get(t,0.0)
        acc[pos][f"k{k_}"].append(pts(blend))
        if games <= EARLY_MAX:
            acc_e[pos][f"k{k_}"].append(pts(blend))

def table(acc, label):
    print(f"\n=== {label} ===")
    print(f"{'POS':4}{'n':>5}{'rho_raw':>9}{'mae_raw':>9} | " + " ".join(f"rho@K{k_}".rjust(9) for k_ in Ks))
    print("-"*100)
    for pos in POS_TARGETS:
        d=acc[pos]; n=len(d["model"])
        if n==0: continue
        print(f"{pos:4}{n:>5}{spearman(d['model'],d['actual']):>+9.3f}{mae(d['model'],d['actual']):>9.2f} | " +
              " ".join(f"{spearman(d[f'k{k_}'],d['actual']):>+9.3f}" for k_ in Ks))
    allm=[]; alla=[]; allk={k_:[] for k_ in Ks}
    for pos in POS_TARGETS:
        d=acc[pos]; allm+=d["model"]; alla+=d["actual"]
        for k_ in Ks: allk[k_]+=d[f"k{k_}"]
    if allm:
        print("-"*100)
        print(f"{'ALL':4}{len(allm):>5}{spearman(allm,alla):>+9.3f}{mae(allm,alla):>9.2f} | " +
              " ".join(f"{spearman(allk[k_],alla):>+9.3f}" for k_ in Ks))

table(acc, "2025 test — ALL weeks")
table(acc_e, "2025 test — EARLY season only (games_played_s2d <= 4)  [where shrinkage bites]")
print("\n(higher rho = better ranking; raw = no prior blend)")
