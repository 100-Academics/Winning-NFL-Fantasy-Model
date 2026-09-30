"""Calibration & signed-bias diagnostics for the projection model.

Why this exists
---------------
The Phase 4/6 scorecard (``src/bench.py``) reports MAE / RMSE / Spearman.
All three are **unsigned** — they hide whether the model systematically
OVER- or UNDER-projects a position. That signed bias is precisely what breaks
*flex* decisions: if the model over-projects RBs by ~12% and under-projects
WRs by ~6% (with equal within-position MAE), the cross-position ranking used
to fill flex is wrong, even though every per-position MAE looks fine.

This module adds the two missing views, on the **same 2024-2025 test split**
the bench uses, so the numbers line up with ``bench_report.json``:

  1. **Signed mean error per position (PPR)** — ``mean(pred - actual)`` plus
     the *relative* bias (bias / actual mean) so positions are comparable.
     Also computed for the two naive baselines (``last_game``, ``trailing3``)
     so we can see whether the model is *better* or *worse* at calibration
     than "last week" / "3-week average".
  2. **Projected-vs-actual BY DECILE** — rank players by predicted PPR, split
     into 10 deciles, report mean predicted vs mean actual per decile (the
     calibration curve) + the actual/predicted ratio. A slope < 1 means the
     top decile is over-projected — the flex-killing case.
  3. **Per-position ``actual ~ predicted`` linear fit** (slope / intercept) as
     a one-number bias summary: slope < 1 = over-projecting, > 1 =
     under-projecting.

Scoring is standard 1-PPR and is imported from ``src.bench`` (single source of
truth, identical to the bench). This is a *diagnostic only* — it reads the
saved models, does not retrain, and does not change predictions.

Output
------
  * ``models/calibration_report.json``
  * ``notebooks/charts/calibration_curve.png``  (decile: global + per-position)
  * ``notebooks/charts/position_bias.png``       (signed bias per position)
  * a stdout summary ranked by |relative bias|

Run:  uv run python -m src.calibrate
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import sys

import joblib
import numpy as np
import polars as pl

# Standard 1-PPR scoring — identical to src.bench (self-contained so this
# diagnostic doesn't depend on the bench module being present).
SCORE = {
    "passing_yards": 0.1, "rushing_yards": 0.1, "receiving_yards": 0.1,
    "passing_tds": 6.0, "rushing_tds": 6.0, "receiving_tds": 6.0,
    "receptions": 1.0,
    "passing_interceptions": -2.0,
}
from src.clean import PROCESSED_DIR
from src.model import POS_TARGETS, MODELS_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CHARTS_DIR = REPO_ROOT / "notebooks" / "charts"
TEST_SEASONS = [2024, 2025]
N_DECILES = 10

# Positions that are normally flex-eligible — the cross-position bias among
# these is the one that distorts the flex choice.
FLEX_POS = ["RB", "WR", "TE"]


# --------------------------------------------------------------------------- #
# Prediction assembly (vectorized per position; comparable to bench)
# --------------------------------------------------------------------------- #
def _load_payload() -> dict:
    return joblib.load(MODELS_DIR / "models.joblib")


def _test_frame() -> pl.DataFrame:
    """Test-split frame with the bench's baseline columns (last game / tr3).

    Identical baseline construction to ``src.bench``: sort by
    (player, season, week), then ``shift(1).over([player, season])`` so the
    naive forecasts never cross a season boundary.
    """
    feat = (pl.read_parquet(PROCESSED_DIR / "features.parquet")
            .sort(["player_id", "season", "week"])
            .filter(pl.col("season").is_in(TEST_SEASONS)))
    stat_names = sorted({t for ts in POS_TARGETS.values() for t in ts})
    for s in stat_names:
        if s in feat.columns:
            feat = feat.with_columns(
                pl.col(s).shift(1).over(["player_id", "season"]).alias(s + "_bg"),
                (pl.col(s).shift(1).over(["player_id", "season"])
                 .rolling_mean(3, min_samples=1)
                 .over(["player_id", "season"]).alias(s + "_bg3")),
            )
    return feat


def _position_predictions(feat: pl.DataFrame, payload: dict) -> dict[str, dict[str, np.ndarray]]:
    """Per position: arrays of model / last_game / trailing3 / actual PPR.

    Model PPR is the sum of the per-target HGBR predictions weighted by the
    1-PPR score — exactly what ``src.bench`` and the predict CLI score.
    """
    feat_cols = payload["meta"]["feature_columns"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for pos, targets in POS_TARGETS.items():
        sub = feat.filter(pl.col("position") == pos)
        if sub.height == 0:
            continue
        X = sub.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)

        model_ppr = np.zeros(X.shape[0])
        for t in targets:
            m = payload["models"].get(f"{pos}/{t}")
            if m is None or t not in SCORE:
                continue
            model_ppr = model_ppr + SCORE[t] * m.predict(X)

        actual_ppr = np.zeros(X.shape[0])
        last_ppr = np.zeros(X.shape[0])
        tr3_ppr = np.zeros(X.shape[0])
        for t in targets:
            w = SCORE.get(t, 0.0)
            if w == 0.0:
                continue
            actual_ppr = actual_ppr + w * np.nan_to_num(sub[t].to_numpy(), nan=0.0)
            last_ppr = last_ppr + w * np.nan_to_num(sub[t + "_bg"].to_numpy(), nan=0.0)
            tr3_ppr = tr3_ppr + w * np.nan_to_num(sub[t + "_bg3"].to_numpy(), nan=0.0)

        out[pos] = {"model": model_ppr, "actual": actual_ppr,
                    "last_game": last_ppr, "trailing3": tr3_ppr}
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _signed_bias(pred: np.ndarray, actual: np.ndarray) -> dict:
    n = len(actual)
    if n == 0:
        return {"n": 0, "actual_mean": None, "pred_mean": None,
                "signed_bias": None, "rel_bias": None}
    am = float(actual.mean())
    pm = float(pred.mean())
    bias = pm - am                       # + => over-projecting
    rel = (bias / am) if am else float("nan")
    return {"n": n,
            "actual_mean": round(am, 3),
            "pred_mean": round(pm, 3),
            "signed_bias": round(bias, 3),
            "rel_bias": round(rel, 4)}


def _linear_fit(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Fit actual = intercept + slope * pred (OLS, closed form).

    slope < 1 => the model over-projects (predicted spread wider than reality);
    slope > 1 => under-projects. Intercept > 0 => a flat positive offset.
    """
    if len(pred) < 2 or np.std(pred) == 0:
        return {"slope": None, "intercept": None}
    x = pred - pred.mean()
    slope = float(np.sum(x * (actual - actual.mean())) / np.sum(x * x))
    intercept = float(actual.mean() - slope * pred.mean())
    return {"slope": round(slope, 4), "intercept": round(intercept, 3)}


def _deciles(pred: np.ndarray, actual: np.ndarray) -> list[dict]:
    """Split into N_DECILES equal-sized groups by predicted value (ascending).

    Returns one row per decile (index 0 = lowest predicted .. 9 = highest).
    """
    n = len(pred)
    if n == 0:
        return []
    order = np.argsort(pred, kind="stable")
    decile = np.clip(np.floor(np.arange(n) / (n / N_DECILES)), 0, N_DECILES - 1).astype(int)
    assign = np.empty(n, dtype=int)
    assign[order] = decile

    out = []
    for d in range(N_DECILES):
        mask = assign == d
        p, a = pred[mask], actual[mask]
        mp, ma = float(p.mean()), float(a.mean())
        out.append({
            "decile": d + 1,
            "n": int(mask.sum()),
            "mean_pred": round(mp, 3),
            "mean_actual": round(ma, 3),
            "ratio_actual_over_pred": round(ma / mp, 4) if mp else None,
            "residual_actual_minus_pred": round(ma - mp, 3),
        })
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report() -> dict:
    payload = _load_payload()
    feat = _test_frame()
    per_pos = _position_predictions(feat, payload)

    report: dict = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "scoring": "standard 1-PPR (imported from src.bench)",
        "test_seasons": TEST_SEASONS,
        "n_deciles": N_DECILES,
        "positions": {},
        "flex_distortion": None,
        "aggregate": {},
    }

    # Per-position: signed bias + fit + decile calibration (model vs baselines).
    for pos in POS_TARGETS:
        if pos not in per_pos:
            continue
        d = per_pos[pos]
        report["positions"][pos] = {
            "n": int(d["model"].shape[0]),
            "model": _signed_bias(d["model"], d["actual"]),
            "last_game": _signed_bias(d["last_game"], d["actual"]),
            "trailing3": _signed_bias(d["trailing3"], d["actual"]),
            "fit_model": _linear_fit(d["model"], d["actual"]),
            "fit_last_game": _linear_fit(d["last_game"], d["actual"]),
            "fit_trailing3": _linear_fit(d["trailing3"], d["actual"]),
            "decile_model": _deciles(d["model"], d["actual"]),
            "decile_last_game": _deciles(d["last_game"], d["actual"]),
            "decile_trailing3": _deciles(d["trailing3"], d["actual"]),
        }

    # Per-position, per-STAT signed bias (model) — pinpoints WHICH components
    # drive the position-level under/over-projection (the actionable fix target).
    feat_cols = payload["meta"]["feature_columns"]
    for pos, targets in POS_TARGETS.items():
        if pos not in report["positions"]:
            continue
        sub = feat.filter(pl.col("position") == pos)
        if sub.height == 0:
            continue
        X = sub.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
        stats = {}
        for t in targets:
            m = payload["models"].get(f"{pos}/{t}")
            if m is None or t not in sub.columns:
                continue
            pred = m.predict(X)
            actual = np.nan_to_num(sub[t].to_numpy(), nan=0.0)
            am = float(actual.mean())
            pm = float(pred.mean())
            stats[t] = {
                "actual_mean": round(am, 3), "pred_mean": round(pm, 3),
                "signed_bias": round(pm - am, 3),
                "rel_bias": round((pm - am) / am, 4) if am else None,
            }
        report["positions"][pos]["per_stat_model"] = stats

    # Flex distortion: spread of relative bias among flex-eligible positions.
    # Direction-agnostic: report the two extremes and the spread; the sign of
    # the values tells whether the model over- or under-projects each one.
    flex_rel = {pos: report["positions"][pos]["model"]["rel_bias"]
                for pos in FLEX_POS if pos in report["positions"]}
    if len(flex_rel) >= 2:
        vals = list(flex_rel.values())
        report["flex_distortion"] = {
            "positions": {k: v for k, v in flex_rel.items()},
            "spread": round(max(vals) - min(vals), 4),
            "extreme_high": [max(flex_rel, key=flex_rel.get), max(vals)],
            "extreme_low": [min(flex_rel, key=flex_rel.get), min(vals)],
            "all_underprojected": all(v < 0 for v in vals),
            "all_overprojected": all(v > 0 for v in vals),
        }

    # Aggregate (all positions pooled) — the global calibration curve.
    all_model = np.concatenate([per_pos[p]["model"] for p in per_pos])
    all_actual = np.concatenate([per_pos[p]["actual"] for p in per_pos])
    all_last = np.concatenate([per_pos[p]["last_game"] for p in per_pos])
    all_tr3 = np.concatenate([per_pos[p]["trailing3"] for p in per_pos])
    report["aggregate"] = {
        "n": int(all_actual.shape[0]),
        "model": _signed_bias(all_model, all_actual),
        "last_game": _signed_bias(all_last, all_actual),
        "trailing3": _signed_bias(all_tr3, all_actual),
        "fit_model": _linear_fit(all_model, all_actual),
        "decile_model": _deciles(all_model, all_actual),
        "decile_last_game": _deciles(all_last, all_actual),
        "decile_trailing3": _deciles(all_tr3, all_actual),
    }
    return report


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _charts(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    pos_order = [p for p in POS_TARGETS if p in report["positions"]]
    agg = report["aggregate"]

    # ---- Calibration curve (decile): global + per-position residual ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    dm = agg["decile_model"]
    dl = agg["decile_last_game"]
    d3 = agg["decile_trailing3"]
    decs = [d["decile"] for d in dm]
    axL.plot(decs, [d["mean_pred"] for d in dm], "o-", color="#1f77b4", label="model: predicted")
    axL.plot(decs, [d["mean_actual"] for d in dm], "s-", color="#d62728", label="model: actual")
    axL.plot(decs, [d["mean_pred"] for d in dl], "o--", color="#1f77b4", alpha=0.4, label="last_game: predicted")
    axL.plot(decs, [d["mean_actual"] for d in dl], "s--", color="#d62728", alpha=0.4, label="last_game: actual")
    # ideal diagonal through the origin
    lim = max(max(d["mean_pred"] for d in dm), max(d["mean_actual"] for d in dm)) * 1.05
    axL.plot([0, lim], [0, lim], "k:", lw=1, alpha=0.5, label="ideal (1:1)")
    axL.set_xlabel("decile (1 = lowest predicted, 10 = highest predicted)")
    axL.set_ylabel("mean PPR points")
    axL.set_title(f"Global calibration curve, test {TEST_SEASONS}\n"
                  f"(actual vs predicted, by predicted decile)")
    axL.legend(fontsize=7, ncol=2)
    axL.grid(alpha=0.25)

    # per-position: residual (actual - predicted) by decile, model
    for pos in pos_order:
        dd = report["positions"][pos]["decile_model"]
        axR.plot([d["decile"] for d in dd], [d["residual_actual_minus_pred"] for d in dd],
                 "o-", label=pos, lw=1.5)
    axR.axhline(0, color="k", lw=0.8, alpha=0.6)
    axR.set_xlabel("decile (1 = lowest predicted, 10 = highest predicted)")
    axR.set_ylabel("residual = actual − predicted (PPR)")
    axR.set_title("Per-position calibration residual by decile (model)\n"
                  "(>0 = model under-projects that tier; <0 = over-projects)")
    axR.legend(fontsize=8)
    axR.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "calibration_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Signed bias per position ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(pos_order))
    w = 0.26
    model_bias = [report["positions"][p]["model"]["signed_bias"] for p in pos_order]
    last_bias = [report["positions"][p]["last_game"]["signed_bias"] for p in pos_order]
    tr3_bias = [report["positions"][p]["trailing3"]["signed_bias"] for p in pos_order]

    axA.bar(x - w, model_bias, w, label="model",
            color=[("#2ca02c" if v < 0 else "#d62728") for v in model_bias])
    axA.bar(x, last_bias, w, label="last game", color="#ff7f0e", alpha=0.6)
    axA.bar(x + w, tr3_bias, w, label="trailing 3-avg", color="#9467bd", alpha=0.6)
    axA.axhline(0, color="k", lw=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels(pos_order)
    axA.set_ylabel("signed bias = mean(pred − actual), PPR")
    axA.set_title("Signed bias per position (green = under, red = over)\n"
                  "Model vs naive baselines, test 2024-2025")
    axA.legend(fontsize=8)
    axA.grid(axis="y", alpha=0.25)

    rel = [report["positions"][p]["model"]["rel_bias"] * 100 for p in pos_order]
    colors = []
    for p, v in zip(pos_order, rel):
        if p not in FLEX_POS:
            colors.append("#7f7f7f")
        else:
            colors.append("#2ca02c" if v < 0 else "#d62728")
    axB.bar(x, rel, color=colors)
    axB.axhline(0, color="k", lw=0.8)
    for xi, v in zip(x, rel):
        axB.text(xi, v + (1.2 if v >= 0 else -2.2), f"{v:+.1f}%", ha="center", fontsize=8)
    axB.set_xticks(x)
    axB.set_xticklabels([f"{p}{'*' if p in FLEX_POS else ''}" for p in pos_order])
    axB.set_ylabel("relative bias = bias / actual mean (%)")
    axB.set_title("Relative bias per position (model)\n"
                  "(* = flex-eligible RB/WR/TE; the spread among these distorts flex)")
    axB.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "position_bias.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Stdout summary
# --------------------------------------------------------------------------- #
def _print_summary(report: dict) -> None:
    print("=" * 74)
    print(f"SIGNED BIAS BY POSITION  (PPR; + = over-projecting, test {TEST_SEASONS})")
    print("=" * 74)
    hdr = f"{'POS':<4}{'n':>6}{'actμ':>9}{'predμ':>9}{'bias':>8}{'rel%':>8}  " \
          f"{'lg_bias':>8}{'tr3_bias':>9}  {'fit_slope':>10}"
    print(hdr)
    print("-" * 74)
    for pos in [p for p in POS_TARGETS if p in report["positions"]]:
        d = report["positions"][pos]
        m, lg, t3 = d["model"], d["last_game"], d["trailing3"]
        slope = d["fit_model"].get("slope")
        slope_s = f"{slope:.3f}" if slope is not None else "  n/a"
        print(f"{pos:<4}{m['n']:>6}{m['actual_mean']:>9.2f}{m['pred_mean']:>9.2f}"
              f"{m['signed_bias']:>+8.2f}{m['rel_bias']*100:>+8.1f}"
              f"{lg['signed_bias']:>+8.2f}{t3['signed_bias']:>+9.2f}"
              f"  {slope_s:>9}")

    # Per-stat signed bias (model) — which components are driving the bias.
    print("\n" + "-" * 74)
    print("PER-STAT SIGNED BIAS (model; + = over, - = under):")
    for pos in [p for p in POS_TARGETS if p in report["positions"]]:
        stats = report["positions"][pos].get("per_stat_model", {})
        if not stats:
            continue
        parts = []
        for t, s in stats.items():
            rel = f"{s['rel_bias']*100:+.0f}%" if s["rel_bias"] is not None else "  n/a"
            parts.append(f"{t} {s['signed_bias']:+.2f} ({rel})")
        print(f"  {pos:<3} " + " | ".join(parts))

    if report.get("flex_distortion"):
        f = report["flex_distortion"]
        hi, hi_v = f["extreme_high"]
        lo, lo_v = f["extreme_low"]
        if f.get("all_underprojected"):
            tone = "ALL flex positions are UNDER-projected; the one least under-projected " \
                   f"is {hi} ({hi_v*100:+.1f}%), the most is {lo} ({lo_v*100:+.1f}%)"
        elif f.get("all_overprojected"):
            tone = "ALL flex positions are OVER-projected; the one least over-projected " \
                   f"is {hi} ({hi_v*100:+.1f}%), the most is {lo} ({lo_v*100:+.1f}%)"
        else:
            tone = f"{hi} {hi_v*100:+.1f}% vs {lo} {lo_v*100:+.1f}%"
        print("\n" + "-" * 74)
        print(f"FLEX DISTORTION ({'/'.join(FLEX_POS)}): {tone}")
        print(f"  -> cross-position bias spread = {f['spread']*100:.1f} pts of relative bias;")
        print("     a flex pick between these positions is distorted by that spread.")

    a = report["aggregate"]
    am, fit = a["model"], a["fit_model"]
    print("\n" + "-" * 74)
    print(f"AGGREGATE (all positions, n={a['n']}): "
          f"model signed bias {am['signed_bias']:+.2f} "
          f"({am['rel_bias']*100:+.1f}%), fit slope {fit.get('slope')} "
          f"(<1 = over, >1 = under)")
    print(f"  baselines: last_game {a['last_game']['signed_bias']:+.2f} "
          f"({a['last_game']['rel_bias']*100:+.1f}%), "
          f"trailing3 {a['trailing3']['signed_bias']:+.2f} "
          f"({a['trailing3']['rel_bias']*100:+.1f}%)")

    # Decile calibration tail (the flex-killing tiers).
    dm = a["decile_model"]
    if dm:
        top = dm[-1]
        bot = dm[0]
        print("\n" + "-" * 74)
        print("DECILE CALIBRATION (model, aggregate):")
        print(f"  top decile (highest predicted):  pred {top['mean_pred']:.1f} "
              f"vs actual {top['mean_actual']:.1f}  ->  "
              f"{top['residual_actual_minus_pred']:+.1f} "
              f"(ratio actual/pred {top['ratio_actual_over_pred']})")
        print(f"  bottom decile (lowest predicted): pred {bot['mean_pred']:.1f} "
              f"vs actual {bot['mean_actual']:.1f}  ->  "
              f"{bot['residual_actual_minus_pred']:+.1f} "
              f"(ratio actual/pred {bot['ratio_actual_over_pred']})")
    print("=" * 74)


def main() -> int:
    report = build_report()
    (MODELS_DIR).mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "calibration_report.json"
    out.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    try:
        _charts(report)
        print(f"chart -> {CHARTS_DIR / 'calibration_curve.png'}")
        print(f"chart -> {CHARTS_DIR / 'position_bias.png'}")
    except Exception as e:  # charts are non-fatal for the JSON report
        print(f"[charts skipped: {e!r}]", file=sys.stderr)
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
