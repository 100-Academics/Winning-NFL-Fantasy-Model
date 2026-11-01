"""GOODNESS BATTERY — is the model GOOD, not just above the naive baselines?

The weekly holdout score (src.holdout --score) only measures the model against
two STRONG-ISH naive baselines (last game, trailing-3). This battery adds the
higher bars, on the LIVE 2026 scored weeks (the only true out-of-sample data
for the shipped model, which was trained through 2025):

  1. DECISION VALUE (start/sit): per (week, position), who's TOP-1 pick
     actually scored more — model vs last-game vs tr3. Win rate (tie=0.5)
     and mean point edge. Plus top-3 / top-5 mean actual vs the field mean
     (what your first board is worth).
  2. FLEX VALUE: pool RB+WR+TE (the flex-eligible set), top-10 per method by
     predicted points -> mean actual vs the pooled field mean. This is the
     direct test of why the central estimate was fit as the MEAN (E[points] =
     sum(score*E[stat]) only holds for the mean; a biased model misranks flex).
  3. NOISE FLOOR: MAE vs a zero-signal predictor (each player's position mean
     from 2025 w15-17 actuals — known pre-kickoff, no per-player info) and
     variance explained (1 - Var(model residual) / Var(actual)).
  4. LIVE CALIBRATION: pooled signed bias per position (mean(pred - actual),
     relative to the actual mean) for model / last / tr3.
  5. TOP-OF-BOARD RELIABILITY: per (week, position), the method's projected
     top-5 -> their mean ACTUAL rank that week (lower = better; chance =
     (n+1)/2).

Everything is computed the same way src.holdout.score_week does (latest log
entry per player; baselines via full-season shift(1) / rolling_mean(3,
min_samples=1) over [player_id, season]; ESPN Standard points from
src.scoring). Re-run any week:  uv run python notebooks/_goodness_battery.py

Writes models/goodness_report.json and prints the summary tables.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))
from src.scoring import ESPN_STANDARD as SCORE, SCORING_LABEL
from src.model import POS_TARGETS
from src.holdout import LOG_PATH

SEASON = 2026
POS = ["QB", "RB", "WR", "TE"]
FLEX = ["RB", "WR", "TE"]

# ------------------------------------------------------------------ #
# load: latest log entry per (season, week, player_id)               #
# ------------------------------------------------------------------ #
entries = {}
with LOG_PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e.get("season") != SEASON:
            continue
        entries[(e["season"], e["week"], e["player_id"])] = e

feat = pl.read_parquet(REPO / "data/processed/features.parquet")
stat_names = sorted({t for ts in POS_TARGETS.values() for t in ts})
feat_season = feat.filter(pl.col("season") == SEASON).sort(["player_id", "season", "week"])
for s in stat_names:
    if s in feat_season.columns:
        feat_season = feat_season.with_columns(
            pl.col(s).shift(1).over(["player_id", "season"]).alias(s + "_bg"),
            pl.col(s).shift(1).over(["player_id", "season"])
                .rolling_mean(3, min_samples=1).over(["player_id", "season"])
                .alias(s + "_bg3"))
base_by = {(int(r["week"]), r["player_id"]): r for r in feat_season.to_dicts()}

def pts(stats: dict, suffix: str = "") -> float:
    return sum(SCORE.get(s, 0.0) * float(stats.get(s + suffix) or 0.0) for s in stat_names)

# ------------------------------------------------------------------ #
# zero-signal floor: per-position mean from 2025 w15-17 actuals      #
# ------------------------------------------------------------------ #
f25 = feat.filter((pl.col("season") == 2025) & pl.col("week").is_in([15, 16, 17]))
floor = {}
for pos in POS:
    targets = POS_TARGETS[pos]
    vals = [sum(SCORE.get(s, 0.0) * float(r.get(s) or 0.0) for s in targets)
            for r in f25.filter(pl.col("position") == pos).to_dicts()]
    floor[pos] = float(np.mean(vals)) if vals else 0.0

# ------------------------------------------------------------------ #
# assemble per-week, per-player rows                                 #
# ------------------------------------------------------------------ #
rows = []
for (season, week, pid), e in entries.items():
    pos = e["position"]
    targets = POS_TARGETS.get(pos, [])
    if pos not in POS or not targets:
        continue
    a = feat.filter((pl.col("season") == SEASON) & (pl.col("week") == week)
                   & (pl.col("player_id") == pid))
    if a.height == 0:
        continue  # not scored yet (e.g. w8 before the games)
    ar = a.to_dicts()[0]
    b = base_by.get((week, pid), {})
    model_pts = sum(SCORE.get(s, 0.0) * float(e["preds"].get(s) or 0.0) for s in targets)
    rows.append({
        "week": week, "pos": pos, "player": e["player_display_name"],
        "actual": sum(SCORE.get(s, 0.0) * float(ar.get(s) or 0.0) for s in targets),
        "model": model_pts,
        "last": sum(SCORE.get(s, 0.0) * float(b.get(s + "_bg") or 0.0) for s in targets),
        "tr3": sum(SCORE.get(s, 0.0) * float(b.get(s + "_bg3") or 0.0) for s in targets),
    })
df = pl.DataFrame(rows)
weeks = sorted(df["week"].unique().to_list())
print(f"scored weeks: {weeks} | n rows: {df.height} | floor (2025 w15-17 means): "
      + ", ".join(f"{p}={floor[p]:.2f}" for p in POS))

METHODS = ["model", "last", "tr3"]

def topk_actual(d: pl.DataFrame, col: str, k: int) -> float:
    return float(d.sort(col, descending=True).head(k)["actual"].mean())

def field_mean(d: pl.DataFrame) -> float:
    return float(d["actual"].mean())

# ------------------------------------------------------------------ #
# 1+2. decision value (top-1 / top-3 / top-5) + flex                 #
# ------------------------------------------------------------------ #
decision = {}
for pos in POS:
    per_week, pool = [], []
    for w in weeks:
        d = df.filter((pl.col("week") == w) & (pl.col("pos") == pos))
        if d.height < 10:
            continue
        rec = {"week": w, "n": d.height, "field": round(field_mean(d), 2)}
        for m in METHODS:
            d2 = d.with_columns(pl.col(m).alias("m")).sort("m", descending=True)
            rec[m] = {"top1": round(float(d2.head(1)["actual"].mean()), 2),
                      "top3": round(topk_actual(d2, "m", 3), 2),
                      "top5": round(topk_actual(d2, "m", 5), 2)}
            pool.append({"m": m, "top1": d2.head(1)["actual"].mean(),
                         "top3": d2.head(3)["actual"].mean(),
                         "top5": d2.head(5)["actual"].mean(),
                         "field": field_mean(d)})
        decision.setdefault(pos, {"weeks": [], "pool": []})
        decision[pos]["weeks"].append(rec)
        decision[pos]["pool"].extend(pool)
    # pooled win rates computed below (head-to-head section)
flex = {}
for w in weeks:
    d = df.filter((pl.col("week") == w) & pl.col("pos").is_in(FLEX))
    if d.height < 30:
        continue
    rec = {"week": w, "n": d.height, "field": round(field_mean(d), 2)}
    for m in METHODS:
        d2 = d.sort(m, descending=True)
        rec[m] = {"top10": round(float(d2.head(10)["actual"].mean()), 2),
                  "top20": round(float(d2.head(20)["actual"].mean()), 2)}
    flex[w] = rec

# ------------------------------------------------------------------ #
# 3. noise floor + 4. calibration + 5. top-of-board                  #
# ------------------------------------------------------------------ #
floor_rep, calib, topboard = {}, {}, {}
for pos in POS:
    d = df.filter(pl.col("pos") == pos)
    act = d["actual"].to_numpy()
    model = d["model"].to_numpy()
    mae_model = float(np.abs(model - act).mean())
    mae_floor = float(np.abs(np.full_like(act, floor[pos]) - act).mean())
    var_expl = 1.0 - float(np.var(act - model)) / float(np.var(act)) if np.var(act) > 0 else float("nan")
    floor_rep[pos] = {"n": d.height, "mae_model": round(mae_model, 3),
                      "mae_floor": round(mae_floor, 3),
                      "lift": round(mae_floor / mae_model, 2),
                      "var_explained": round(var_expl, 3)}
    rel = {m: round(float((d[m].to_numpy() - act).mean()) / float(act.mean()) * 100, 1)
           for m in METHODS}
    absb = {m: round(float((d[m].to_numpy() - act).mean()), 2) for m in METHODS}
    calib[pos] = {"n": d.height, "act_mean": round(float(act.mean()), 2),
                  "rel_bias_pct": rel, "abs_bias": absb}
    tb = {m: [] for m in METHODS}
    for w in weeks:
        dw = d.filter(pl.col("week") == w)
        if dw.height < 10:
            continue
        actual_rank = {r["player"]: i + 1 for i, r in
                       enumerate(dw.sort("actual", descending=True).to_dicts())}
        for m in METHODS:
            top5 = dw.sort(m, descending=True).head(5)["player"].to_list()
            tb[m].append(np.mean([actual_rank[p] for p in top5]))
    topboard[pos] = {m: round(float(np.mean(v)), 1) for m, v in tb.items() if v}
    topboard[pos]["n_weeks"] = len(tb["model"])
    topboard[pos]["chance"] = round((np.median([dw.height for dw in
                                                [d.filter(pl.col('week')==w) for w in weeks] if dw.height>=10]) + 1) / 2, 1)

report = {"season": SEASON, "scoring": SCORING_LABEL,
          "note": ("goodness battery on live scored weeks (only true OOS for the "
                   "shipped model, trained through 2025). topk = mean ACTUAL pts of "
                   "the method's top-k by predicted pts; field = all scored players "
                   "that position/week. topboard = mean ACTUAL rank of the method's "
                   "projected top-5 (chance = median (n+1)/2). flex pools RB/WR/TE."),
          "weeks": weeks, "decision": decision, "flex": flex,
          "floor": floor_rep, "calibration": calib, "topboard": topboard}
(REPO / "models/goodness_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

# ------------------------------------------------------------------ #
# print                                                               #
# ------------------------------------------------------------------ #
print("\n=== 1. DECISION VALUE — top-1 / top-3 / top-5 mean ACTUAL (per position) ===")
for pos in POS:
    p = [x for x in decision[pos]["pool"] if "top1" in x]
    agg = {m: {k: round(float(np.mean([x[k] for x in p if x["m"] == m])), 2)
               for k in ("top1", "top3", "top5")} for m in METHODS}
    fld = round(float(np.mean([x["field"] for x in p])), 2)
    print(f"  {pos:2s} field={fld:5.2f} | " +
          "  ".join(f"{m}: t1={agg[m]['top1']:.2f} t3={agg[m]['top3']:.2f} t5={agg[m]['top5']:.2f}"
                    for m in METHODS))
print("\n  (top-1 head-to-head win rates, pooled over weeks, tie=0.5:)")
for pos in POS:
    line = []
    for m in ("last", "tr3"):
        wins = ties = n = 0.0
        for w in weeks:
            d = df.filter((pl.col("week") == w) & (pl.col("pos") == pos))
            if d.height < 10:
                continue
            dm = d.sort("model", descending=True).head(1)["actual"].to_list()[0]
            do = d.sort(m, descending=True).head(1)["actual"].to_list()[0]
            n += 1
            if dm > do: wins += 1
            elif dm == do: wins += 0.5; ties += 1
        line.append(f"vs {m}: {wins:.1f}/{int(n)}" + (f" ({ties} ties)" if ties else ""))
    print(f"  {pos:2s} " + "  ".join(line))

print("\n=== 2. FLEX VALUE — pooled RB/WR/TE, top-10 / top-20 mean ACTUAL ===")
for w in sorted(flex):
    r = flex[w]
    print(f"  w{w}: n={r['n']} field={r['field']:.2f} | " +
          "  ".join(f"{m}: t10={r[m]['top10']:.2f} t20={r[m]['top20']:.2f}" for m in METHODS))

print("\n=== 3. NOISE FLOOR (MAE vs zero-signal position mean; lift = floor/model) ===")
for pos in POS:
    f = floor_rep[pos]
    print(f"  {pos:2s} n={f['n']} MAE model={f['mae_model']:.2f} floor={f['mae_floor']:.2f} "
          f"lift={f['lift']:.2f}x  var_explained={f['var_explained']:.3f}")

print("\n=== 4. LIVE CALIBRATION (signed bias, pooled over scored weeks) ===")
print(f"  {'pos':3s} {'act_mean':>8s} | " + "  ".join(f"{m+' %':>14s}" for m in METHODS) +
      "  | " + "  ".join(f"{m+' abs':>14s}" for m in METHODS))
for pos in POS:
    c = calib[pos]
    print(f"  {pos:3s} {c['act_mean']:8.2f} | " +
          "  ".join(f"{c['rel_bias_pct'][m]:>14.1f}" for m in METHODS) +
          "  | " + "  ".join(f"{c['abs_bias'][m]:>14.2f}" for m in METHODS))

print("\n=== 5. TOP-OF-BOARD (mean ACTUAL rank of projected top-5; lower=better) ===")
for pos in POS:
    t = topboard[pos]
    print(f"  {pos:2s} chance={t['chance']} | " +
          "  ".join(f"{m}={t[m]}" for m in METHODS))
print("\nwrote models/goodness_report.json")
