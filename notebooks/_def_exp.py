"""Defense experiment: do TRUE opponent-defense features (yards/sacks/TDs
allowed, trailing 3) improve the player projection models?

A/B per (position, target) on the SAME split as the saved model:
  train 2016-2023 | val 2024 | test 2025
  baseline = saved model (already has the proxy total_yards_opp features)
  defense  = baseline features + new defense features (allowed yards/sacks/TDs)
Keep only if it improves test MAE and/or Spearman by a meaningful margin.
"""
import sys, itertools
from pathlib import Path
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import joblib

TRAIN = list(range(2016,2024)); VAL=[2024]; TEST=[2025]
KEYS = {"player_id","player_display_name","season","week","team","opponent_team","game_id","position","position_group"}
TARGET_COLS = {"passing_yards","passing_tds","passing_interceptions","passing_air_yards",
    "rushing_yards","rushing_tds","rushing_fumbles","receptions","receiving_yards","receiving_tds","receiving_fumbles",
    "targets","carries","completions","attempts"}

# ---- build true opponent-defense trailing-3 features (strictly prior weeks) ----
def trailing3(df, key, col):
    df = df.sort([key,"season","week"])
    return df.with_columns(
        (pl.col(col).shift(1).over([key,"season"]).rolling_mean(3,min_samples=1).over([key,"season"]).alias(col+"_tr3"))
    )

tw = pl.read_parquet(REPO/"data/raw/team_weekly_stats.parquet")
tw = tw.select(["season","week","team","passing_yards","rushing_yards","receiving_yards",
                "def_sacks","passing_tds","rushing_tds","receiving_tds"])
for c in ["passing_yards","rushing_yards","receiving_yards","def_sacks","passing_tds","rushing_tds","receiving_tds"]:
    tw = tw.with_columns(pl.col(c).fill_null(0.0))
tw = tw.with_columns((pl.col("passing_yards")+pl.col("rushing_yards")+pl.col("receiving_yards")).alias("allowed_total_yds"))
tw = tw.with_columns((pl.col("passing_tds")+pl.col("rushing_tds")+pl.col("receiving_tds")).alias("allowed_tds"))
for col in ["allowed_total_yds","def_sacks","allowed_tds"]:
    tw = trailing3(tw, "team", col)
def_tab = tw.select(["season","week","team",
                     "allowed_total_yds_tr3","def_sacks_tr3","allowed_tds_tr3"])
def_tab = def_tab.rename({"team":"opponent_team"})

# ---- load feature panel (has the baseline features incl proxy opp) ----
feat = pl.read_parquet(REPO/"data/processed/features.parquet")
# join true defense by (season, week, opponent_team)
feat = feat.join(def_tab, on=["season","week","opponent_team"], how="left")
DEF = ["allowed_total_yds_tr3","def_sacks_tr3","allowed_tds_tr3"]
# fill nulls (teams with no prior game) with 0
feat = feat.with_columns([pl.col(c).fill_null(0.0) for c in DEF])

payload = joblib.load(REPO/"models/models.joblib")
base_fc = payload["meta"]["feature_columns"]
models = payload["models"]
POS_TARGETS = payload["meta"]["pos_targets"]
base_fc = [c for c in base_fc if c in feat.columns]
def_fc = base_fc + DEF
print(f"baseline features: {len(base_fc)} | +defense: {len(DEF)} new -> {len(def_fc)}")

def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.std(a)==0 or np.std(b)==0: return float("nan")
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def mae(a,b): return float(np.abs(np.asarray(a)-np.asarray(b)).mean())

GRID = list(itertools.product((150,300,500),(0.03,0.06),(3,5),(1.0,5.0)))
def grid_best(Xtr,ytr,Xva,yva):
    best=None
    for mi,lr,dp,l2 in GRID:
        m=HGBR(loss="squared_error",random_state=0,max_iter=mi,learning_rate=lr,max_depth=dp,l2_regularization=l2,max_leaf_nodes=31,min_samples_leaf=50)
        m.fit(Xtr,ytr); va=float(np.abs(m.predict(Xva)-yva).mean())
        if best is None or va<best[0]: best=(va,m)
    return best[1]

def Xy(sub,cols,target):
    X=sub.select(cols).fill_null(0.0).to_numpy().astype(np.float32)
    y=sub.select(target).to_series().to_numpy().astype(np.float64)
    return X,y

print(f"\n{'pos/tgt':24}{'baseMAE':>8}{'defMAE':>8}{'ΔMAE':>7} | {'baseρ':>7}{'defρ':>7}{'Δρ':>7}")
print("-"*82)
results={}
for pos,targets in POS_TARGETS.items():
    sub=feat.filter(pl.col("position")==pos)
    tr=sub.filter(pl.col("season").is_in(TRAIN)); va=sub.filter(pl.col("season").is_in(VAL)); te=sub.filter(pl.col("season").is_in(TEST))
    for t in targets:
        Xtr,ytr=Xy(tr,base_fc,t); Xva,yva=Xy(va,base_fc,t); Xte,yte=Xy(te,base_fc,t)
        mb=grid_best(Xtr,ytr,Xva,yva).predict(Xte)
        # defense model: refit with defense features
        Xtr_d,ytr_d=Xy(tr,def_fc,t); Xva_d,yva_d=Xy(va,def_fc,t); Xte_d,yte_d=Xy(te,def_fc,t)
        md=grid_best(Xtr_d,ytr_d,Xva_d,yva_d).predict(Xte_d)
        bmae=mae(mb,yte); dmae=mae(md,yte); bsp=spearman(mb,yte); dsp=spearman(md,yte)
        d_mae=dmae-bmae; d_rho=dsp-bsp
        flag = "KEEP" if (d_mae < -0.02 or d_rho > 0.01) else ("drop" if (d_mae > 0.02 or d_rho < -0.01) else "same")
        results[f"{pos}/{t}"]=(bmae,dmae,bsp,dsp)
        print(f"{pos+'/'+t:24}{bmae:>8.2f}{dmae:>8.2f}{d_mae:>+7.2f} | {bsp:>+7.3f}{dsp:>+7.3f}{d_rho:>+7.4f}  {flag}")

# tally
keep=[k for k,(b,d,bp,dp) in results.items() if (d-b)<-0.02 or (dp-bp)>0.01]
drop=[k for k,(b,d,bp,dp) in results.items() if (d-b)>0.02 or (dp-bp)<-0.01]
same=[k for k in results if k not in keep and k not in drop]
print(f"\nTALLY: KEEP {len(keep)} | drop {len(drop)} | same {len(same)}")
print("  keep:", keep)
print("  drop:", drop)
