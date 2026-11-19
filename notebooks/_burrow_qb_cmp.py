"""W11 QB component predictions (all 5 board QBs) + cohort ceiling."""
import polars as pl
import numpy as np
import src.holdout as H
import src.scoring as S

panel = H._build_scheduled_panel(2026, 11)
payload = H._load_payload()
feat_cols = payload["meta"]["feature_columns"]
models = payload["models"]
qbs = panel.filter(pl.col("position") == "QB")

names = ["Dak Prescott", "Josh Allen", "Caleb Williams", "C.J. Stroud", "Joe Burrow"]
sel = (qbs.filter(pl.col("player_display_name").is_in(names))
       .select(feat_cols).fill_null(0.0))
X = sel.to_numpy().astype("float32")
targets = ["passing_yards", "passing_tds", "passing_interceptions",
           "rushing_yards", "rushing_tds"]
preds = {t: models[f"QB/{t}"].predict(X) for t in targets}
print(f"{'QB':15} {'py':>6} {'td':>5} {'int':>5} {'rshy':>6} {'rstd':>5} {'pts':>7}")
for i, n in enumerate(names):
    p = {t: float(preds[t][i]) for t in targets}
    print(f"{n:15} {p['passing_yards']:>6.1f} {p['passing_tds']:>5.2f} "
          f"{p['passing_interceptions']:>5.2f} {p['rushing_yards']:>6.1f} "
          f"{p['rushing_tds']:>5.2f} {S.fantasy_points(p):>7.2f}")

Xall = qbs.select(feat_cols).fill_null(0.0).to_numpy().astype("float32")
py = models["QB/passing_yards"].predict(Xall)
td = models["QB/passing_tds"].predict(Xall)
print(f"\ncohort n={len(py)}: py max={py.max():.1f} p95={np.percentile(py,95):.1f} "
      f"mean={py.mean():.1f}")
print(f"cohort n={len(td)}: td max={td.max():.2f} p95={np.percentile(td,95):.2f} "
      f"mean={td.mean():.2f}")

feat = pl.read_parquet('data/processed/features.parquet').filter(
    (pl.col('season') == 2026) & (pl.col('week') <= 10)
    & (pl.col('player_id') == '00-0036442'))
print(f"\nBurrow 2026 actuals (w1-10, n={feat.height}): "
      f"avg py={feat['passing_yards'].mean():.0f} "
      f"td={feat['passing_tds'].mean():.2f} "
      f"int={feat['passing_interceptions'].mean():.2f}")
