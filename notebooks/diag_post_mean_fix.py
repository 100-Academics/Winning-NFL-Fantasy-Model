"""Post-mean-fix diagnostics (2024-25 test, same split as src.calibrate).

Checks requested 2026-09-30, on the CURRENT models (mean-fit central
estimate, quantile p10/p90 bands):
  1. Signed bias broken out BY SEASON (2024, 2025) and pooled — per position
     (PPR) and per stat — a pooled +3.3% RB bias could be +8% in one season
     and -2% in the other.
  2. Decile calibration BY POSITION, all 10 deciles — especially the middle
     and bottom deciles (mean-objective models on zero-inflated stats tend to
     over-project low-usage players), per season + pooled.
  3. p10/p90 BAND COVERAGE per position (headline stats): empirical coverage
     vs the nominal 80%, and where the mean (central estimate) sits inside
     the band — required if the bands are used for risk-based start/sit.
  4. Negative predictions in COUNT stats (squared-error can go below 0):
     count, min, and the PPR mass that would change if clipped at 0.

Read-only: loads models.joblib + features.parquet, no retraining.
Run:  uv run python notebooks/diag_post_mean_fix.py
"""
from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.clean import PROCESSED_DIR
from src.model import POS_TARGETS, HEADLINE, MODELS_DIR

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SCORE = {"passing_yards": 0.1, "rushing_yards": 0.1, "receiving_yards": 0.1,
         "passing_tds": 6.0, "rushing_tds": 6.0, "receiving_tds": 6.0,
         "receptions": 1.0, "passing_interceptions": -2.0}
COUNT_STATS = {"receptions", "passing_tds", "rushing_tds", "receiving_tds",
               "passing_interceptions"}
TEST_SEASONS = [2024, 2025]
N_DEC = 10


def spearman(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def bias_stats(pred: np.ndarray, actual: np.ndarray) -> dict:
    n = len(actual)
    if n == 0:
        return {"n": 0}
    am, pm = float(actual.mean()), float(pred.mean())
    b = pm - am
    return {"n": n, "actual_mean": round(am, 3), "pred_mean": round(pm, 3),
            "bias": round(b, 3), "rel_bias": round(b / am, 4) if am else None,
            "mae": round(float(np.abs(pred - actual).mean()), 3),
            "rmse": round(float(np.sqrt(((pred - actual) ** 2).mean())), 3),
            "spearman": round(spearman(pred, actual), 4)}


def deciles(pred: np.ndarray, actual: np.ndarray) -> list[dict]:
    n = len(pred)
    if n == 0:
        return []
    order = np.argsort(pred, kind="stable")
    d = np.clip(np.floor(np.arange(n) / (n / N_DEC)), 0, N_DEC - 1).astype(int)
    assign = np.empty(n, dtype=int)
    assign[order] = d
    out = []
    for i in range(N_DEC):
        m = assign == i
        p, a = pred[m], actual[m]
        mp, ma = float(p.mean()), float(a.mean())
        out.append({"d": i + 1, "n": int(m.sum()), "mp": round(mp, 3),
                    "ma": round(ma, 3),
                    "ratio": round(ma / mp, 4) if mp else None,
                    "resid": round(ma - mp, 3)})
    return out


def main() -> None:
    payload = joblib.load(MODELS_DIR / "models.joblib")
    feat_cols = payload["meta"]["feature_columns"]
    models = payload["models"]
    bands = payload.get("bands", {})

    feat = (pl.read_parquet(PROCESSED_DIR / "features.parquet")
            .sort(["player_id", "season", "week"])
            .filter(pl.col("season").is_in(TEST_SEASONS)))
    print(f"test rows: {feat.height:,}  (2024: {feat.filter(pl.col('season')==2024).height:,}, "
          f"2025: {feat.filter(pl.col('season')==2025).height:,})\n")

    # Precompute per (pos, target) predictions on the test frame — per season.
    preds: dict[str, dict[int, dict[str, np.ndarray]]] = {}   # pos -> season -> target -> pred
    actuals: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for pos, targets in POS_TARGETS.items():
        for season in TEST_SEASONS:
            sub = feat.filter((pl.col("position") == pos) & (pl.col("season") == season))
            if sub.height == 0:
                continue
            X = sub.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
            preds.setdefault(pos, {})[season] = {}
            actuals.setdefault(pos, {})[season] = {}
            for t in targets:
                m = models.get(f"{pos}/{t}")
                if m is None:
                    continue
                preds[pos][season][t] = m.predict(X)
                actuals[pos][season][t] = np.nan_to_num(sub[t].to_numpy(), nan=0.0)

    def ppr(preds_t: dict[str, np.ndarray], actuals_t: dict[str, np.ndarray]):
        p = np.zeros(len(next(iter(preds_t.values()))))
        a = np.zeros_like(p)
        for t, w in SCORE.items():
            if t in preds_t and w:
                p += w * preds_t[t]
                a += w * actuals_t[t]
        return p, a

    report: dict = {}

    def assemble(pos: str, seasons: list[int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Per-target arrays for the given seasons, concatenated in season
        order (predictions and actuals aligned row-for-row)."""
        pt, at = {}, {}
        for s in seasons:
            if s not in preds.get(pos, {}):
                continue
            for t in preds[pos][s]:
                pt[t] = np.concatenate([pt[t], preds[pos][s][t]]) if t in pt else preds[pos][s][t].copy()
                at[t] = np.concatenate([at[t], actuals[pos][s][t]]) if t in at else actuals[pos][s][t].copy()
        return pt, at

    # ------------------------------------------------------------------ #
    # 1. BIAS BY SEASON (PPR, per position) + per-stat
    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("1) SIGNED BIAS BY POSITION (PPR) — BY SEASON  (+ = over-projecting)")
    print("=" * 78)
    hdr = (f"{'POS':<4}{'SCOPE':<8}{'n':>6}{'actμ':>8}{'predμ':>8}{'bias':>7}{'rel%':>8}"
           f"{'MAE':>7}{'RMSE':>7}{'rho':>8}")
    print(hdr)
    print("-" * 78)
    bias_rows = []
    for pos in POS_TARGETS:
        for scope, seasons in [("2024", [2024]), ("2025", [2025]), ("pool", TEST_SEASONS)]:
            pt, at = assemble(pos, seasons)
            if not pt:
                continue
            p, a = ppr(pt, at)
            st = bias_stats(p, a)
            bias_rows.append({"pos": pos, "scope": scope, **st})
            print(f"{pos:<4}{scope:<8}{st['n']:>6}{st['actual_mean']:>8.2f}"
                  f"{st['pred_mean']:>8.2f}{st['bias']:>+7.2f}"
                  f"{st['rel_bias']*100:>+8.1f}{st['mae']:>7.2f}{st['rmse']:>7.2f}"
                  f"{st['spearman']:>8.3f}")
    report["bias_by_position"] = bias_rows

    print("\n" + "-" * 78)
    print("PER-STAT SIGNED BIAS (model, pooled 2024-25)  (+ = over, - = under):")
    stat_rows = []
    for pos, targets in POS_TARGETS.items():
        pt, at = assemble(pos, TEST_SEASONS)
        for t in targets:
            if t not in pt:
                continue
            st = bias_stats(pt[t], at[t])
            neg = float((pt[t] < 0).sum())
            stat_rows.append({"pos": pos, "stat": t, **st,
                              "n_negative_pred": int(neg)})
            rel = f"{st['rel_bias']*100:+6.1f}%" if st["rel_bias"] is not None else "   n/a"
            print(f"  {pos:<3} {t:<24} act {st['actual_mean']:>6.2f}  pred {st['pred_mean']:>6.2f}  "
                  f"bias {st['bias']:>+6.2f} ({rel})  rho {st['spearman']:+.3f}  "
                  f"neg_pred {int(neg)}/{st['n']}")
    report["bias_by_stat"] = stat_rows

    # ------------------------------------------------------------------ #
    # 2. DECILE CALIBRATION BY POSITION (pooled + per season)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("2) DECILE CALIBRATION BY POSITION (PPR; residual = actual - pred)")
    print("=" * 78)
    dec_rows = []
    for pos in POS_TARGETS:
        for scope, seasons in [("pool", TEST_SEASONS), ("2024", [2024]), ("2025", [2025])]:
            pt, at = assemble(pos, seasons)
            if not pt:
                continue
            p, a = ppr(pt, at)
            dl = deciles(p, a)
            dec_rows.append({"pos": pos, "scope": scope, "deciles": dl})
            if scope == "pool":
                line = []
                for d in dl:
                    line.append(f"D{d['d']}:{d['resid']:+.1f}")
                top, mid1, mid2, bot = dl[9], dl[4], dl[3], dl[0]
                print(f"  {pos:<3} pooled n={len(p)}  " + "  ".join(line))
                print(f"      top D10: pred {top['mp']:.1f} vs act {top['ma']:.1f} (ratio {top['ratio']}) | "
                      f"bottom D1: pred {bot['mp']:.1f} vs act {bot['ma']:.1f} (ratio {bot['ratio']}) | "
                      f"D3-D5 resid {mid2['resid']:+.1f}/{mid1['resid']:+.1f}")
    report["deciles"] = dec_rows

    # ------------------------------------------------------------------ #
    # 3. BAND COVERAGE (p10/p90) per position, headline stats
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("3) p10/p90 BAND COVERAGE (nominal 80%) per position — headline stats")
    print("=" * 78)
    band_rows = []
    for pos, targets in POS_TARGETS.items():
        # Self-consistent frame for this position: one X, and the actuals +
        # central prediction all pulled from THAT SAME row order, so
        # central / band / actual arrays are row-aligned by construction.
        sub = feat.filter(pl.col("position") == pos)
        X = sub.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
        for t in targets:
            if t not in HEADLINE.get(pos, set()):
                continue
            key = f"{pos}/{t}"
            if key not in bands or "p10" not in bands[key]:
                print(f"  {pos:<3} {t:<18} (no band model saved — skipping)")
                continue
            lo_m, hi_m = bands[key]["p10"], bands[key]["p90"]
            lo, hi = lo_m.predict(X), hi_m.predict(X)
            a = np.nan_to_num(sub[t].to_numpy(), nan=0.0)
            m = models[f"{pos}/{t}"].predict(X)
            cover = float(((a >= lo) & (a <= hi)).mean())
            # Where the central (mean) estimate sits inside the band — only
            # over bands with positive width (degenerate hi==lo rows excluded;
            # reported as a fraction so sparse stats don't dominate the mean).
            width = hi - lo
            valid = width > 1e-6
            pos_in = (m[valid] - lo[valid]) / width[valid] if valid.any() else np.array([0.5])
            pos_frac_med = float(np.median(pos_in))
            degenerate = float((~valid).mean())
            # band width and where the actual mean sits relative to the band
            mean_bw = float(np.mean(width))
            mean_act = float(a.mean())
            # inverted bands (p10 above the central mean / p90 below it) would
            # break risk-based start/sit — count rows where that happens.
            inv_lo = float((lo > m).mean())
            inv_hi = float((hi < m).mean())
            band_rows.append({"pos": pos, "stat": t, "n": int(len(a)),
                              "coverage": round(cover, 4),
                              "pos_in_band_median": round(pos_frac_med, 4),
                              "degenerate_band_frac": round(degenerate, 4),
                              "mean_band_width": round(mean_bw, 3),
                              "p10_above_mean_frac": round(inv_lo, 4),
                              "p90_below_mean_frac": round(inv_hi, 4)})
            print(f"  {pos:<3} {t:<18} n={len(a):>5}  coverage {cover*100:5.1f}% (want ~80)  "
                  f"mean-in-band {pos_frac_med*100:3.0f}% (want 40-70)  "
                  f"degen {degenerate*100:4.1f}%  "
                  f"band width {mean_bw:.2f} (act μ {mean_act:.2f})  "
                  f"p10>mean {inv_lo*100:.1f}%  p90<mean {inv_hi*100:.1f}%")
    report["bands"] = band_rows

    # ------------------------------------------------------------------ #
    # 4. NEGATIVE PREDICTIONS in count stats + PPR mass if clipped
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("4) NEGATIVE PREDICTIONS (count stats) — clipping candidates")
    print("=" * 78)
    neg_rows = []
    for row in stat_rows:
        if row["stat"] not in COUNT_STATS:
            continue
        pos, t = row["pos"], row["stat"]
        pt, at = assemble(pos, TEST_SEASONS)
        p, a = pt[t], at[t]
        neg_mask = p < 0
        w = SCORE[t]
        # MAE of this stat's PPR contribution before/after clipping at zero.
        err_before = float(np.abs(w * p - w * a).mean())
        err_after = float(np.abs(w * np.clip(p, 0, None) - w * a).mean())
        min_p = float(p.min())
        neg_rows.append({"pos": pos, "stat": t, "n": int(len(p)),
                         "n_negative": int(neg_mask.sum()),
                         "min_pred": round(min_p, 3),
                         "mean_neg": round(float(p[neg_mask].mean()), 3) if neg_mask.any() else None,
                         "mae_before_clip": round(err_before, 4),
                         "mae_after_clip": round(err_after, 4)})
        print(f"  {pos:<3} {t:<24} neg {int(neg_mask.sum()):>4}/{len(p):>5} "
              f"({neg_mask.mean()*100:5.1f}%)  min {min_p:+6.2f}  "
              f"mean-of-neg {float(p[neg_mask].mean()) if neg_mask.any() else 0:+.3f}  "
              f"stat-MAE {err_before:.3f} -> clipped {err_after:.3f}")
    report["negatives"] = neg_rows

    out = REPO_ROOT / "models" / "diag_post_mean_fix.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
