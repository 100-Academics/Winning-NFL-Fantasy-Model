"""Experiment: which loss function per (position, target) gives the right
CENTRAL ESTIMATE (low signed bias) without hurting MAE / Spearman.

Current model.py fits quantile=0.5 (the conditional MEDIAN). For zero-inflated,
right-skewed targets the median sits below the mean, so the central estimate
systematically undershoots — that is the calibration bias we found. The fix that
is "right" and stable is to fit the mean (squared_error) for the central model
and keep the p10/p90 quantile models for the bands.

This experiment confirms that, per stat, on the exact Phase-4 split:
  * signed bias (test) under each loss
  * MAE (test) under each loss  (does switching to mean hurt accuracy?)
  * Spearman (test)  (does it hurt ranking?)

We also check the drift hypothesis: is the test-season mean of each stat higher
than the train-season mean? (a league-level inflation that no per-row loss can
fix alone.)

Run:  uv run python notebooks/exp_loss.py
"""
from __future__ import annotations
import sys
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.model import (load_features, feature_columns, _split, TRAIN_SEASONS,
                       VAL_SEASONS, TEST_SEASONS, POS_TARGETS)

feat = load_features()
feat_cols = feature_columns(feat)
print(f"features {feat.height:,} rows x {len(feat_cols)} cols\n")

# fixed hyper-params (same family as model.py best_params, conservative)
BASE = dict(max_iter=300, learning_rate=0.06, max_depth=4,
            l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=50)

def mae(p, a): return float(np.abs(p-a).mean())
def bias(p, a):
    am = float(a.mean())
    return float(p.mean()-am), (float(p.mean()-am)/am if am else float("nan"))
def spear(p, a):
    from scipy.stats import rankdata
    if np.std(p)==0 or np.std(a)==0: return float("nan")
    return float(np.corrcoef(rankdata(p), rankdata(a))[0,1])

rows = []
for pos, targets in POS_TARGETS.items():
    sub = feat.filter(pl.col("position")==pos)
    tr = sub.filter(pl.col("season").is_in(TRAIN_SEASONS))
    te = sub.filter(pl.col("season").is_in(TEST_SEASONS))
    for t in targets:
        Xtr = tr.select(feat_cols).fill_null(0.0).to_numpy().astype("float32")
        ytr = tr.select(t).to_series().to_numpy().astype("float64")
        Xte = te.select(feat_cols).fill_null(0.0).to_numpy().astype("float32")
        yte = te.select(t).to_series().to_numpy().astype("float64")
        # drift check
        tr_mean = float(ytr.mean()); te_mean = float(yte.mean())
        frac0_tr = float((ytr==0).mean()); frac0_te = float((yte==0).mean())
        for loss, q in (("quantile",0.5), ("squared_error",None), ("absolute_error",None)):
            if q is not None:
                m = HGBR(loss=loss, quantile=q, random_state=0, **BASE).fit(Xtr,ytr)
            else:
                m = HGBR(loss=loss, random_state=0, **BASE).fit(Xtr,ytr)
            p = m.predict(Xte)
            b, rel = bias(p, yte)
            rows.append((pos,t,loss,tr_mean,te_mean,frac0_tr,frac0_te,
                         mae(p,yte), b, rel, spear(p,yte)))

# print table, grouped by stat
from collections import defaultdict
by = defaultdict(list)
for r in rows: by[(r[0],r[1])].append(r)
print(f"{'stat':<22}{'trμ':>7}{'teμ':>7}{'drift':>8}{'%0tr':>6}{'%0te':>6}   "
      f"{'LOSS':<15}{'MAE':>8}{'bias':>8}{'rel%':>8}{'spear':>7}")
print("-"*108)
for (pos,t), rs in by.items():
    r0 = rs[0]
    drift = (r0[4]-r0[3])/r0[3]*100 if r0[3] else 0
    for r in sorted(rs, key=lambda x:x[2]):
        print(f"{pos+'/'+t:<22}{r0[3]:>7.2f}{r0[4]:>7.2f}{drift:>+7.0f}%"
              f"{r0[5]*100:>5.0f}%{r0[6]*100:>5.0f}%   "
              f"{r[2]:<15}{r[7]:>8.2f}{r[8]:>+8.2f}{r[9]*100:>+7.0f}%"
              f"{r[10]:>7.3f}")
    print()
