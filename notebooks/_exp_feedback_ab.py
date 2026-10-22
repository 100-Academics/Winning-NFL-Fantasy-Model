"""A/B test of the 2026-10-22 third-party feedback proposals (repro).

Every arm uses the SAME split as src.model (train 2016-23, val 2024, test 2025),
the SAME 48-config grid, and is scored on the 2025 test holdout (+ 2024
in-sample-for-diagnosis). Decision rule (NOTES.md): a change ships ONLY if it
beats the baseline (V0) on the 2025 holdout.

Arms
----
V0 baseline  : current features.parquet, squared_error loss, MAE grid selection
               (retrained here so every arm is trained on the identical
               features.parquet + grid).
V1 bugfix    : the two "feature bugs" from the feedback:
                (a) NGS features joined on player_gsis_id == player_id
                    (was player_display_name),
                (b) team yardage = passing + rushing ONLY (in
                    team_weekly_stats, passing_yards == receiving_yards
                    exactly, so the old sum double-counted passing), and the
                    opponent feature = the opponent's trailing DEFENSE yards
                    ALLOWED (the old one was the opponent's own trailing
                    offensive yards, mislabeled "allowed").
V2 usage     : V0 features + trailing-3 / s2d of target_share,
               air_yards_share, wopr, racr (already in player_weekly_stats).
V3 priors    : V0 features + prior-season per-game usage (targets/carries/
               attempts/receptions/receiving_yards/passing_yards per game +
               games played) — the "veteran prior-season bridge" as INPUT
               features (distinct from the rejected post-hoc blend, _q1_ab.py).
V4 rmse-sel  : V0 features, grid selected on val RMSE (was MAE).
V5 poisson   : V0 features, poisson loss on the 5 count stats
               (receptions, passing/rushing/receiving tds, passing int),
               squared_error on yards.

Usage:
    uv run python notebooks/_exp_feedback_ab.py [--arms V0 V1 ...]
Output:
    models/ab_feedback_report.json  (+ printed tables)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import pathlib
import time

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

from src.clean import RAW_DIR, PROCESSED_DIR, load_team_stats
from src.model import (MODELS_DIR, POS_TARGETS, TRAIN_SEASONS, VAL_SEASONS,
                       TEST_SEASONS, _split, _baselines, _grid, _season_weights,
                       load_features, feature_columns)

OUT = MODELS_DIR / "ab_feedback_report.json"
RELEVANT = ["QB", "RB", "WR", "TE"]
COUNTS = {"receptions", "passing_tds", "rushing_tds", "receiving_tds",
          "passing_interceptions"}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _mae(p, a): return float(np.abs(p - a).mean())
def _rmse(p, a): return float(np.sqrt(((p - a) ** 2).mean()))
def _rho(p, a):
    if np.std(p) == 0 or np.std(a) == 0: return float("nan")
    return float(np.corrcoef(rankdata(p), rankdata(a))[0, 1])


def _score(pred, actual, base):
    base_mae = min(base["last_game"]["mae"], base["trailing3"]["mae"])
    return {"mae": round(_mae(pred, actual), 3),
            "rmse": round(_rmse(pred, actual), 3),
            "spearman": round(_rho(pred, actual), 4),
            "bias": round(float(pred.mean() - actual.mean()), 3),
            "base_mae": round(base_mae, 3),
            "beats_base": bool(_mae(pred, actual) < base_mae)}


# --------------------------------------------------------------------------- #
# Feature variants (all return a frame compatible with feature_columns)
# --------------------------------------------------------------------------- #
def _trailing(key_cols, col, window=3):
    return (pl.col(col).shift(1).over(key_cols + ["season"])
            .rolling_mean(window, min_samples=1).over(key_cols + ["season"]))


def _s2d(key_cols, col):
    prior = pl.col(col).shift(1).over(key_cols + ["season"])
    return (prior.rolling_mean(20, min_samples=1).over(key_cols + ["season"])
            .fill_null(0.0))


def feat_v1(feat: pl.DataFrame) -> pl.DataFrame:
    """(a) NGS by player_id; (b) corrected team/opp yardage features."""
    # ---- (a) NGS re-joined on gsis id ------------------------------------
    ngs_map = {
        "passing": (RAW_DIR / "nextgen_passing.parquet",
                    ["completion_percentage", "avg_time_to_throw",
                     "aggressiveness", "avg_completed_air_yards"]),
        "receiving": (RAW_DIR / "nextgen_receiving.parquet",
                      ["avg_cushion", "avg_separation",
                       "percent_share_of_intended_air_yards", "catch_percentage"]),
        "rushing": (RAW_DIR / "nextgen_rushing.parquet",
                    ["efficiency", "rush_yards_over_expected_per_att",
                     "avg_time_to_los"]),
    }
    out = feat
    for tag, (path, cols) in ngs_map.items():
        g = pl.read_parquet(path).filter(pl.col("season_type") == "REG") \
            .select(["season", "week", "player_gsis_id",
                     *[c for c in cols if c in pl.read_parquet(path, n_rows=0).columns]])
        keep = [c for c in cols if c in g.columns]
        if not g.height or not keep:
            continue
        g = g.sort(["player_gsis_id", "season", "week"])
        for c in keep:
            g = g.with_columns(_trailing(["player_gsis_id"], c, 3).alias(f"ngs_{c}_tr3"))
        g = (g.select(["season", "week",
                       pl.col("player_gsis_id").alias("player_id")]
                      + [f"ngs_{c}_tr3" for c in keep])
             .unique(subset=["season", "week", "player_id"]))
        # replace (or add) the ngs_* columns
        drop = [c for c in out.columns if c.startswith("ngs_") and
                any(c == f"ngs_{k}_tr3" for k in keep)]
        out = out.drop(drop).join(g, on=["season", "week", "player_id"], how="left")

    # ---- (b) team / opponent yardage --------------------------------------
    team = load_team_stats(RAW_DIR)
    t = team.select(["season", "week", "team", "opponent_team",
                     "passing_yards", "rushing_yards"]).filter(
        pl.col("team").is_not_null() & pl.col("opponent_team").is_not_null())
    t = t.with_columns((pl.col("passing_yards").fill_null(0.0)
                        + pl.col("rushing_yards").fill_null(0.0)).alias("off_yards"))
    # yards ALLOWED by a team = its OPPONENT's offensive yards in the same game
    off = t.select(["season", "week", "team", "off_yards"]).unique(
        subset=["season", "week", "team"])
    allowed = (t.select(["season", "week", "team", "opponent_team"])
               .join(off, left_on=["season", "week", "opponent_team"],
                     right_on=["season", "week", "team"], how="left")
               .select(["season", "week", "team",
                        pl.col("off_yards").fill_null(0.0).alias("allowed_yards")])
               .unique(subset=["season", "week", "team"]))

    def _trail(tab, col):
        x = tab.sort(["team", "season", "week"])
        x = x.with_columns([_trailing(["team"], col, w).alias(f"{col}_tr{w}")
                            for w in (3, 5)])
        x = x.with_columns(_s2d(["team"], col).alias(f"{col}_s2d"))
        return x.select(["season", "week", "team", f"{col}_tr3", f"{col}_tr5",
                         f"{col}_s2d"])

    own = _trail(off, "off_yards")
    opp = _trail(allowed, "allowed_yards")

    rename = {"off_yards_tr3": "total_yards_tr3", "off_yards_tr5": "total_yards_tr5",
              "off_yards_s2d": "total_yards_s2d"}
    own = own.rename(rename)
    opp = opp.select(["season", "week",
                      pl.col("team").alias("opponent_team"),
                      pl.col("allowed_yards_tr3").alias("total_yards_tr3_opp"),
                      pl.col("allowed_yards_tr5").alias("total_yards_tr5_opp"),
                      pl.col("allowed_yards_s2d").alias("total_yards_s2d_opp")])
    old_own = [c for c in out.columns if c in ("total_yards_tr3", "total_yards_tr5",
                                               "total_yards_s2d")]
    old_opp = [c for c in out.columns if c in ("total_yards_tr3_opp",
                                               "total_yards_tr5_opp", "total_yards_s2d_opp")]
    out = out.drop(old_own + old_opp)
    out = out.join(own, on=["season", "week", "team"], how="left")
    out = out.join(opp, on=["season", "week", "opponent_team"], how="left")
    return out


def feat_v2(feat: pl.DataFrame) -> pl.DataFrame:
    """V0 + trailing usage/efficiency shares from player_weekly_stats."""
    pw = pl.read_parquet(RAW_DIR / "player_weekly_stats.parquet") \
        .filter(pl.col("season_type") == "REG") \
        .select(["season", "week", "player_id", "target_share",
                 "air_yards_share", "wopr", "racr"])
    pw = pw.sort(["player_id", "season", "week"]).unique(
        subset=["season", "week", "player_id"])
    for c in ("target_share", "air_yards_share", "wopr", "racr"):
        pw = pw.with_columns([_trailing(["player_id"], c, 3).alias(f"{c}_tr3"),
                              _s2d(["player_id"], c).alias(f"{c}_s2d")])
    pw = pw.select(["season", "week", "player_id", "target_share_tr3",
                    "target_share_s2d", "air_yards_share_tr3",
                    "air_yards_share_s2d", "wopr_tr3", "wopr_s2d",
                    "racr_tr3", "racr_s2d"])
    return feat.join(pw, on=["season", "week", "player_id"], how="left")


def feat_v3(feat: pl.DataFrame) -> pl.DataFrame:
    """V0 + prior-season per-game usage (input-feature veteran bridge)."""
    panel = pl.read_parquet(PROCESSED_DIR / "panel.parquet")
    stats = ["targets", "carries", "attempts", "receptions", "receiving_yards",
             "passing_yards"]
    p = panel.select(["player_id", "season", "week", *stats])
    per_season = (p.group_by(["player_id", "season"])
                  .agg([pl.col(s).mean().alias(f"{s}_pg") for s in stats]
                       + [pl.len().alias("prev_season_games")]))
    # A player's stats in play-season S are their PRIOR-season features for
    # season S+1, so tag each play-season group with (S + 1) to join onto the
    # following season's rows (leak-free: strictly the previous season).
    per_season = (per_season
        .select(["player_id",
                 (pl.col("season") + 1).cast(pl.Int32).alias("season"),
                 *[f"{s}_pg" for s in stats],
                 "prev_season_games"])
        .rename({f"{s}_pg": f"prev_season_{s}_pg" for s in stats}))
    return feat.join(per_season, on=["player_id", "season"], how="left")


# --------------------------------------------------------------------------- #
# Train one arm
# --------------------------------------------------------------------------- #
def run_arm(name: str, feat: pl.DataFrame, feat_cols: list[str],
            select_by: str = "mae", poisson_counts: bool = False) -> dict:
    t0 = time.time()
    results = {}
    for pos in RELEVANT:
        for target in POS_TARGETS[pos]:
            sp = _split(feat, pos)
            Xtr = sp["train"].select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
            ytr = sp["train"].select(target).to_series().to_numpy()
            Xva = sp["val"].select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
            yva = sp["val"].select(target).to_series().to_numpy()
            Xte = sp["test"].select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
            yte = sp["test"].select(target).to_series().to_numpy()
            base = _baselines(sp["test"], target)
            base_m = {k: {"mae": _mae(np.nan_to_num(v, nan=0.0), yte)}
                      for k, v in base.items()}

            loss = "poisson" if (poisson_counts and target in COUNTS) else "squared_error"
            best = None
            for params in _grid(False):
                m = HGBR(loss=loss, random_state=0, **params)
                m.fit(Xtr, ytr)
                va = m.predict(Xva)
                score = _mae(va, yva) if select_by == "mae" else _rmse(va, yva)
                if best is None or score < best[0]:
                    best = (score, params, m)
            pred = best[2].predict(Xte)
            sc = _score(pred, yte, base_m)
            sc["val"] = round(float(best[0]), 3)
            sc["best_params"] = best[1]
            sc["loss"] = loss
            results[f"{pos}/{target}"] = sc
            # early-season slice (the cold-start population)
            w = sp["test"].sort(["player_id", "season", "week"]).with_columns(
                pl.col("week").is_not_null().shift(1).over(["player_id", "season"])
                .rolling_sum(20, min_samples=1).over(["player_id", "season"])
                .cast(pl.Float64).fill_null(0.0).alias("_gp"))
            gs = w.filter(pl.col("_gp") <= 4)
            if gs.height:
                Xte2 = gs.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
                yte2 = gs.select(target).to_series().to_numpy()
                p2 = best[2].predict(Xte2)
                sc["early_spearman"] = round(_rho(p2, yte2), 4)
                sc["early_mae"] = round(_mae(p2, yte2), 3)
                sc["n_early"] = int(gs.height)
    # pooled per-position + overall
    pooled = {}
    for pos in RELEVANT:
        for t in POS_TARGETS[pos]:
            pooled.setdefault(pos, []).append(results[f"{pos}/{t}"])
    def _agg(items):
        return {"n_targets": len(items),
                "mean_mae": round(float(np.mean([i["mae"] for i in items])), 3),
                "mean_spearman": round(float(np.mean([i["spearman"] for i in items
                                                      if not np.isnan(i["spearman"])])), 4),
                "n_beat_base": sum(i["beats_base"] for i in items),
                "early_spearman": round(float(np.mean([i["early_spearman"] for i in items
                                                       if "early_spearman" in i and
                                                       not np.isnan(i["early_spearman"])])), 4)
                        if any("early_spearman" in i for i in items) else None}
    out = {"arm": name, "wall_s": round(time.time() - t0, 1),
           "per_target": results,
           "by_position": {p: _agg(items) for p, items in pooled.items()},
           "pooled": _agg([results[k] for k in results])}
    print(f"  [{name}] done in {out['wall_s']:.0f}s | "
          f"pooled MAE {out['pooled']['mean_mae']} rho {out['pooled']['mean_spearman']} "
          f"beat-base {out['pooled']['n_beat_base']}/{out['pooled']['n_targets']} "
          f"early-rho {out['pooled']['early_spearman']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*",
                    default=["V0", "V1", "V2", "V3", "V4", "V5"])
    args = ap.parse_args()

    print("loading features...")
    feat0 = load_features()
    feat_cols0 = feature_columns(feat0)
    print(f"  {feat0.height} rows x {len(feat_cols0)} baseline features")

    arms = {
        "V0": (feat0, feat_cols0, "mae", False),
    }
    want = set(args.arms)
    if "V1" in want:
        f = feat_v1(feat0); arms["V1"] = (f, feature_columns(f), "mae", False)
    if "V2" in want:
        f = feat_v2(feat0); arms["V2"] = (f, feature_columns(f), "mae", False)
    if "V3" in want:
        f = feat_v3(feat0); arms["V3"] = (f, feature_columns(f), "mae", False)
    if "V4" in want:
        arms["V4"] = (feat0, feat_cols0, "rmse", False)
    if "V5" in want:
        arms["V5"] = (feat0, feat_cols0, "mae", True)

    report = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
              "split": {"train": TRAIN_SEASONS, "val": VAL_SEASONS, "test": TEST_SEASONS},
              "arms": {}}
    for name in [a for a in ("V0", "V1", "V2", "V3", "V4", "V5") if a in arms]:
        f, cols, sel, pois = arms[name]
        print(f"=== {name} ({len(cols)} features) ===")
        report["arms"][name] = run_arm(name, f, cols, sel, pois)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
