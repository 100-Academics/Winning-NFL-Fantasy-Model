"""Diagnose what the model likes/dislikes about Joe Burrow (2026-11-17).

Parts:
  1. Live 2026 track record: logged prediction (ESPN Std pts) vs actual per week.
  2. W11 feature row vs the QB cohort on the board (what the model saw).
  3. Feature-knockout decomposition of his QB/passing_yards and passing_tds
     predictions vs Prescott (board QB1) — which features push his number
     up/down relative to the all-zeros baseline.
"""
import json
import polars as pl

import src.holdout as H
import src.scoring as S
from src.model import POS_TARGETS

SEASON, WEEK = 2026, 11

# ---------------------------------------------------------------- 1. record
feat = pl.read_parquet("data/processed/features.parquet")
actuals = feat.filter(pl.col("season") == SEASON).sort(["week"])
logged = {}
for line in open("models/holdout_log.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if e["season"] == SEASON:
        logged[(e["week"], e["player_id"])] = e

# Burrow's player_id from a logged entry
burrow = next(e for (wk, pid), e in logged.items()
              if "Burrow" in e["player_display_name"])
BID = burrow["player_id"]
print("Burrow player_id:", BID)

def pts(preds):
    return round(S.fantasy_points(preds), 2)

print("\n=== 1. Live 2026: logged pred (ESPN Std pts) vs actual ===")
print(f"{'wk':>3} {'pred':>7} {'act':>7} {'bias':>7}  detail")
for wk in sorted({w for (w, p) in logged if p == BID}):
    e = logged[(wk, BID)]
    pred_pts = pts(e["preds"])
    a = actuals.filter((pl.col("week") == wk) & (pl.col("player_id") == BID))
    if a.height:
        ar = a.row(0, named=True)
        act = (0.04 * ar.get("passing_yards", 0) + 4 * ar.get("passing_tds", 0)
               - 2 * ar.get("passing_interceptions", 0)
               + 0.1 * ar.get("rushing_yards", 0) + 6 * ar.get("rushing_tds", 0))
        bias = round(pred_pts - act, 2)
        detail = (f"py {ar.get('passing_yards')}/{round(e['preds'].get('passing_yards',0))} "
                  f"td {ar.get('passing_tds')}/{round(e['preds'].get('passing_tds',2),1)} "
                  f"int {ar.get('passing_interceptions')}/{round(e['preds'].get('passing_interceptions',1),1)}")
    else:
        act, bias, detail = float("nan"), float("nan"), "(W11 unplayed)"
    print(f"{wk:>3} {pred_pts:>7} {act if act==act else float('nan'):>7} {bias if bias==bias else 0:>7}  {detail}")

# ---------------------------------------------------------------- 2. W11 row
panel = H._build_scheduled_panel(SEASON, WEEK)
payload = H._load_payload()
feat_cols = payload["meta"]["feature_columns"]
models = payload["models"]
qbs = panel.filter(pl.col("position") == "QB")
names = ["Joe Burrow", "Dak Prescott", "Josh Allen", "Caleb Williams", "C.J. Stroud"]
sel = qbs.filter(pl.col("player_display_name").is_in(names))
print(f"\n=== 2. W11 scheduled panel — QB cohort (n={qbs.height}) ===")
# cohort percentiles for each model feature (QB rows only)
qdf = qbs.select(feat_cols).fill_null(0.0)
key = [c for c in feat_cols if any(k in c for k in
    ["pass_yd", "passing_yard", "pass_att", "attempt", "implied", "total",
     "opp_", "bridge", "share", "ngs_", "wopr", "target", "time_to_throw",
     "aggress", "sack", "injury"])]
print(f"key features ({len(key)}):")
rows = []
for name in names:
    x = qbs.filter(pl.col("player_display_name") == name).select(feat_cols).fill_null(0.0)
    if x.height == 0:
        continue
    r = x.row(0)
    row = {"player": name}
    for c in key:
        v = float(r[qdf.columns.index(c)])
        col = qdf[c].to_numpy()
        pct = float((col < v).mean()) * 100
        row[c] = (round(v, 2), f"{pct:.0f}%")
    rows.append(row)
for c in key:
    print(f"\n-- {c}")
    for r in rows:
        v, pct = r[c]
        print(f"   {r['player']:15} {v:>10}  (cohort {pct})")

# ---------------------------------------------------------------- 3. knockouts
import numpy as np

def row_of(name):
    r = sel.filter(pl.col("player_display_name") == name).row(0, named=True)
    return np.array([float(r[c]) if r[c] is not None else 0.0 for c in feat_cols],
                    dtype=np.float32)

def knockout_contribs(model, x, top=15):
    base0 = float(model.predict(np.zeros_like(x)[None, :])[0])
    full = float(model.predict(x[None, :])[0])
    contribs = []
    for i, c in enumerate(feat_cols):
        if x[i] == 0.0:
            continue
        x0 = x.copy(); x0[i] = 0.0
        d = full - float(model.predict(x0[None, :])[0])
        contribs.append((c, round(float(x[i]), 3), round(d, 3)))
    contribs.sort(key=lambda t: abs(t[2]), reverse=True)
    return base0, full, contribs

for stat in ["passing_yards", "passing_tds"]:
    m = models[f"QB/{stat}"]
    print(f"\n=== 3. QB/{stat}: baseline(zeros) / full pred / top knockouts ===")
    for name in ["Joe Burrow", "Dak Prescott"]:
        x = row_of(name)
        base0, full, contribs = knockout_contribs(m, x, top=12)
        print(f"\n{name}: zeros={base0:.1f}  full={full:.1f}  (lift={full-base0:+.1f})")
        print(f"  {'feature':42} {'value':>10} {'knockout Δ':>10}")
        for c, v, d in contribs[:12]:
            print(f"  {c:42} {v:>10} {d:>+10.1f}")
