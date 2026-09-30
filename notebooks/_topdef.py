"""Week 4 top-defense (D/ST) projection, ESPN STANDARD scoring.

For each week-4 team, project (defense as a whole, no specific players):
  * points_allowed  = 0.5*(own trailing pts allowed) + 0.5*(opponent trailing pts scored)
  * turnovers       = projected opponent INTs (from the saved QB passing_interceptions
                      model) + own trailing D turnovers (INT/fumble/safety/blocked kicks)
Then apply ESPN D/ST scoring:
  points allowed: 0-6:+10  7-13:+7  14-17:+4  18-23:+1  24-30:0  31+:-1
  +2 per turnover; -1 if own D/ST unit scores (approximated as 0 here).
Rank all 16 week-4 teams; print top 10.
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import joblib

def pts_allowed_score(p):
    if p <= 6: return 10
    if p <= 13: return 7
    if p <= 17: return 4
    if p <= 23: return 1
    if p <= 30: return 0
    return -1

# team points scored proxy per game (from team stats)
t = pl.read_parquet(REPO/"data/raw/team_weekly_stats.parquet")
t = t.select(["season","week","team","opponent_team",
              "passing_tds","rushing_tds","receiving_tds","fg_made",
              "def_interceptions","def_fumbles","def_safeties",
              "def_punt_blocks","def_pat_blocks","def_fg_blocks"]).with_columns(
    pl.col("passing_tds").fill_null(0.0)+pl.col("rushing_tds").fill_null(0.0)*0,  # ensure numeric
    )
# points scored = 7*passTD + 6*rushTD + 6*recTD + 3*FG
t = t.with_columns(
    (7*pl.col("passing_tds").fill_null(0.0) + 6*pl.col("rushing_tds").fill_null(0.0)
     + 6*pl.col("receiving_tds").fill_null(0.0) + 3*pl.col("fg_made").fill_null(0.0)).alias("pts_scored"))
# own trailing points ALLOWED = opponent's points scored that game -> trailing 3 of (own pts_scored is offense; allowed = opp)
# For "own points allowed" we need opponent's pts in that game. Build allowed = opponent pts.
# trailing3 helper
def trail3(df,key,col):
    df=df.sort([key,"season","week"])
    return df.with_columns(pl.col(col).shift(1).over([key,"season"]).rolling_mean(3,min_samples=1).over([key,"season"]).alias(col+"_tr3"))

# own trailing points allowed: for team X, allowed = opponent's pts_scored in same game
own_allowed = t.select(["season","week","team","opponent_team"]).join(
    t.select(["season","week","team","pts_scored"]).rename({"team":"opponent_team"}),
    on=["season","week","opponent_team"], how="left")
own_allowed = own_allowed.with_columns(pl.col("pts_scored").alias("pts_allowed")).select(["season","week","team","pts_allowed"])
own_allowed = trail3(own_allowed, "team", "pts_allowed")
own_allowed = own_allowed.select(["season","week","team","pts_allowed_tr3"])

# own trailing D turnovers
TO_COLS=["def_interceptions","def_fumbles","def_safeties","def_punt_blocks","def_pat_blocks","def_fg_blocks"]
t2 = t.select(["season","week","team"]+[c for c in TO_COLS]).with_columns([pl.col(c).fill_null(0.0) for c in TO_COLS])
t2 = t2.with_columns((pl.col(TO_COLS[0]) + pl.col(TO_COLS[1]) + pl.col(TO_COLS[2])
                      + pl.col(TO_COLS[3]) + pl.col(TO_COLS[4]) + pl.col(TO_COLS[5])).alias("d_to"))
t2 = trail3(t2,"team","d_to").select(["season","week","team","d_to_tr3"])

# opponent offense trailing pts scored
opp_off = trail3(t.select(["season","week","team","pts_scored"]),"team","pts_scored").select(["season","week","team","pts_scored_tr3"])
opp_off = opp_off.rename({"team":"opponent_team","pts_scored_tr3":"opp_pts_tr3"})

# opponent QB projected INTs (saved model) for week 4
import src.holdout as H
panel4 = H._build_scheduled_panel(2026, 4)
payload=joblib.load(REPO/"models/models.joblib")
fc=payload["meta"]["feature_columns"]; models=payload["models"]
qb_int={}
for r in panel4.filter(pl.col("position")=="QB").to_dicts():
    X=np.array([0.0 if r.get(c) is None else float(r.get(c)) for c in fc],dtype=np.float32).reshape(1,-1)
    m=models.get("QB/passing_interceptions")
    if m is not None:
        v=float(m.predict(X)[0]); v=max(0,v)
        qb_int[r["team"]]=qb_int.get(r["team"],0)+v

sched = pl.read_parquet(REPO/"data/raw/schedules.parquet").filter((pl.col("season")==2026)&(pl.col("week")==4))
games = sched.select(["away_team","home_team"]).to_dicts()
# team -> opponent
team_opp={}
for g in games:
    team_opp[g["away_team"]]=g["home_team"]; team_opp[g["home_team"]]=g["away_team"]

oa2={}
for r in own_allowed.filter((pl.col("season")==2026)&(pl.col("week")<4)).sort("week").to_dicts():
    if r["pts_allowed_tr3"] is not None: oa2[r["team"]]=r["pts_allowed_tr3"]
to2={}
for r in t2.filter((pl.col("season")==2026)&(pl.col("week")<4)).sort("week").to_dicts():
    if r["d_to_tr3"] is not None: to2[r["team"]]=r["d_to_tr3"]
po2={}
for r in opp_off.filter((pl.col("season")==2026)&(pl.col("week")<4)).sort("week").to_dicts():
    if r["opp_pts_tr3"] is not None: po2[r["opponent_team"]]=r["opp_pts_tr3"]

def project(team):
    opp=team_opp[team]
    own_all=oa2.get(team, 17.0)
    opp_off_=po2.get(opp, 20.0)
    pts_allowed=0.5*own_all+0.5*opp_off_
    to=(to2.get(team,0.0)) + (qb_int.get(opp,0.5))
    base=pts_allowed_score(pts_allowed)
    total=base + 2*to
    return dict(team=team, opp=opp, pts_allowed=round(pts_allowed,1),
                turnovers=round(to,2), pts_allowed_pts=base, total=round(total,2))

rows=[project(t) for t in team_opp]
rows.sort(key=lambda r:r["total"],reverse=True)
print(f"{'RANK':5}{'DEF':6}{'vs':5}{'ptsAllwd':>9}{'TO':>6}{'ptsAllwdP':>10}{'D/ST':>7}")
print("-"*48)
for i,r in enumerate(rows,1):
    mark=" <=" if i<=10 else ""
    print(f"{i:5}{r['team']:6}{r['opp']:5}{r['pts_allowed']:>9}{r['turnovers']:>6}{r['pts_allowed_pts']:>10}{r['total']:>7}{mark}")
