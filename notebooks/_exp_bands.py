"""Band recalibration test (third-party feedback item: conformal / Poisson bands).

The shipped p10/p90 quantile bands were measured under-covering on yards
(QB pass_yds ~74% vs 80% nominal) and degenerate on sparse TD stats (see
diag_post_mean_fix). This script tests two fixes WITHOUT retraining the point
model (band-level only, same as the output-surface clipping):

  A. SPLIT CONFORMAL: add the validation-set (2024) 10th/90th percentile of
     the SIGNED residual (actual - p10) / (actual - p90) to the band edges.
  B. POISSON (count stats only): derive p10/p90 from Poisson(lambda) with
     lambda = max(point-pred, 0.02).

Measured on the CLEAN 2025 test: coverage (share of actuals inside the band)
and mean band width. Calibrating on 2024 (the model's val season) is a known
compromise — the HPs were chosen on 2024, but the residual distribution of
the fitted model is still the right empirical object to calibrate.

Run:  uv run python notebooks/_exp_bands.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import polars as pl

from src.model import MODELS_DIR, PROCESSED_DIR, POS_TARGETS, HEADLINE, TEST_SEASONS, VAL_SEASONS


def main():
    payload = joblib.load(MODELS_DIR / "models.joblib")
    models = payload["models"]
    bands = payload.get("bands", {})
    feat_cols = payload["meta"]["feature_columns"]

    feat = pl.read_parquet(PROCESSED_DIR / "features.parquet")
    val = feat.filter(pl.col("season").is_in(VAL_SEASONS))
    test = feat.filter(pl.col("season").is_in(TEST_SEASONS))

    out = {}
    rows = []
    for pos in ["QB", "RB", "WR", "TE"]:
        for target in HEADLINE.get(pos, ()):
            key = f"{pos}/{target}"
            m = models.get(key)
            b = bands.get(key)
            if m is None or b is None:
                continue
            is_count = target.endswith(("_tds", "_interceptions")) or target == "receptions"

            def predict(df):
                X = df.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
                y = df.select(target).to_series().to_numpy()
                return m.predict(X), b["p10"].predict(X), b["p90"].predict(X), y

            p_val, lo_val, hi_val, y_val = predict(val.filter(pl.col("position") == pos))
            p_te, lo_te, hi_te, y_te = predict(test.filter(pl.col("position") == pos))
            p_te = np.clip(p_te, 0.0, None)

            cov_base = float(((y_te >= lo_te) & (y_te <= hi_te)).mean())
            width_base = float((hi_te - lo_te).mean())

            # A) split conformal shifts from 2024
            shift_lo = float(np.percentile(y_val - lo_val, 10))
            shift_hi = float(np.percentile(y_val - hi_val, 90))
            lo_c, hi_c = lo_te + shift_lo, hi_te + shift_hi
            cov_conf = float(((y_te >= lo_c) & (y_te <= hi_c)).mean())
            width_conf = float((hi_c - lo_c).mean())

            # B) Poisson bands (count stats)
            if is_count:
                from scipy.stats import poisson
                lam = np.maximum(p_te, 0.02)
                lo_p = poisson.ppf(0.10, lam)
                hi_p = poisson.ppf(0.90, lam)
                cov_pois = float(((y_te >= lo_p) & (y_te <= hi_p)).mean())
                width_pois = float((hi_p - lo_p).mean())
            else:
                cov_pois, width_pois = None, None

            rec = {"n_val": int(y_val.size), "n_test": int(y_te.size),
                   "base": {"coverage": round(cov_base, 3), "width": round(width_base, 2)},
                   "conformal": {"shift_lo": round(shift_lo, 3), "shift_hi": round(shift_hi, 3),
                                 "coverage": round(cov_conf, 3), "width": round(width_conf, 2)}}
            if is_count:
                rec["poisson"] = {"coverage": round(cov_pois, 3), "width": round(width_pois, 2)}
            out[key] = rec
            rows.append((key, rec))

    print(f"{'stat':<22}{'n':>6}  {'base cov':>9}{'base w':>8}   {'conf cov':>9}{'conf w':>8}   {'pois cov':>9}{'pois w':>8}")
    for key, rec in rows:
        po = rec.get("poisson")
        po_cov = f"{po['coverage']:.3f}" if po else "    n/a"
        po_w = f"{po['width']:.1f}" if po else "    n/a"
        print(f"{key:<22}{rec['n_test']:>6}  "
              f"{rec['base']['coverage']:>9.3f}{rec['base']['width']:>8.1f}   "
              f"{rec['conformal']['coverage']:>9.3f}{rec['conformal']['width']:>8.1f}   "
              f"{po_cov:>9}{po_w:>8}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = MODELS_DIR / "band_calib_report.json"
    p.write_text(json.dumps({"generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                             "calib_seasons": VAL_SEASONS, "test_seasons": TEST_SEASONS,
                             "results": out}, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
