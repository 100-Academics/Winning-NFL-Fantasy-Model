"""Clean-holdout logging + scoring for the LIVE season (2026).

Why this exists
---------------
NOTES.md (2026-09-30) established that the 2024-25 test split is NO LONGER a
clean holdout: we diagnosed the median-bias on it, so the mean-fit change is
validated on data the model-selection process saw. The one genuinely clean
holdout is **this season's live weeks, logged before kickoff** — a prediction
written to disk with a timestamp, scored against actuals only after the game.

This module is that mechanism, in two explicitly separated steps:

  1. **log**   — predict a (season, week) with the saved model and append the
                 predictions + a wall-clock timestamp to
                 ``models/holdout_log.jsonl``. Run BEFORE the games of that
                 week are played. For weeks already in the data the actuals
                 are already known, so the log is a *replay* of what we would
                 have predicted (still a valid clean prediction, since the
                 model + features were frozen before 2026); the timestamp is
                 recorded so we can audit it.

  2. **score** — read the log, join the actuals from
                 ``data/processed/features.parquet`` (populated by
                 ``--refresh`` / ``download_data`` after the games), and
                 report MAE / RMSE / Spearman per position and aggregate,
                 against the same standard-1-PPR scoring as ``src.bench`` and
                 the two naive baselines (last_game, trailing3) computed the
                 same way.

Design rules (keep the holdout honest):
  * The log is APPEND-ONLY. Re-logging a (season, week) appends a new entry
    with a fresh timestamp; scoring uses the LATEST entry per player.
  * Predictions here are the MAIN model only (no rookie-prior blend), so the
    holdout isolates the central-estimate change under test. The blend is
    applied on top by the predict CLI for display.
  * Count stats are clipped at 0 at prediction time (same rule as
    ``src.predict._clip_pred``) so the logged values are what we'd show.
  * ``score`` never writes to the log; it only reads it + reads actuals.

Usage
-----
    # BEFORE the games: log a week's predictions (the clean act)
    uv run python -m src.holdout --season 2026 --week 4 --log

    # AFTER the games (once actuals are in features.parquet): score it
    uv run python -m src.holdout --season 2026 --week 4 --score

    # both at once (log then score) — only sensible for an already-played week
    uv run python -m src.holdout --season 2026 --week 3 --log --score

Run:  uv run python -m src.holdout
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys

import joblib
import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.clean import PROCESSED_DIR, RAW_DIR, TARGET_COLS, attach_schedule_context
from src.model import MODELS_DIR, POS_TARGETS, HEADLINE
from src import features as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Standard 1-PPR (identical to src.bench / src.calibrate).
SCORE = {
    "passing_yards": 0.1, "rushing_yards": 0.1, "receiving_yards": 0.1,
    "passing_tds": 6.0, "rushing_tds": 6.0, "receiving_tds": 6.0,
    "receptions": 1.0, "passing_interceptions": -2.0,
}

COUNT_STATS = {"passing_tds", "rushing_tds", "receiving_tds", "receptions",
               "passing_interceptions"}
LOG_PATH = MODELS_DIR / "holdout_log.jsonl"
SCORE_PATH = MODELS_DIR / "holdout_score.json"


# --------------------------------------------------------------------------- #
# Prediction (main model only, count-stats clipped)
# --------------------------------------------------------------------------- #
def _clip(v: float, target: str) -> float:
    return 0.0 if (target in COUNT_STATS and v < 0.0) else v


def _load_payload() -> dict:
    p = MODELS_DIR / "models.joblib"
    if not p.exists():
        raise SystemExit(f"models not found at {p}; run `uv run python -m src.model` first.")
    return joblib.load(p)


def _load_features() -> pl.DataFrame:
    p = PROCESSED_DIR / "features.parquet"
    if not p.exists():
        raise SystemExit(
            f"features not found at {p}; run `uv run python -m src.features` first "
            f"(or `predict --refresh` for an upcoming week).")
    return pl.read_parquet(p)


def _build_scheduled_panel(season: int, week: int) -> pl.DataFrame:
    """Construct feature rows for an UPCOMING (unplayed) week, leakage-safe.

    The pipeline's panel only contains PLAYED weeks (``player_weekly_stats`` is
    an outcomes table). To predict a future week *before kickoff*, we feed the
    pipeline's OWN ``build_features_from`` a synthetic season: all PLAYED weeks
    of the season plus one DUMMY row at the target week. Because every feature
    is computed as ``value.shift(1)`` within the season (``.over([.., season])``),
    the target row's trailing features depend only on the played weeks — the
    dummy's own stats (a copy of the last played week) are shifted OUT of every
    window, so nothing leaks. We then keep just the target-week row.

    Uses strictly-prior data: roster (this season's played weeks), the week's
    schedule (pre-game), and trailing player/team/NextGen features from the
    season's played weeks. Returns one row per (player, week) for the games.
    """
    pw = pl.read_parquet(RAW_DIR / "player_weekly_stats.parquet")
    sched = pl.read_parquet(RAW_DIR / "schedules.parquet")
    team = pl.read_parquet(RAW_DIR / "team_weekly_stats.parquet")
    ngs_p = pl.read_parquet(RAW_DIR / "nextgen_passing.parquet")
    ngs_r = pl.read_parquet(RAW_DIR / "nextgen_receiving.parquet")
    ngs_ru = pl.read_parquet(RAW_DIR / "nextgen_rushing.parquet")

    pos4 = ["QB", "RB", "WR", "TE"]
    # full target set (attach_schedule_context selects all of these)
    TARGETS = list(TARGET_COLS)

    games = sched.filter((pl.col("season") == season) & (pl.col("week") == week))
    if games.height == 0:
        return pl.DataFrame()
    # teams actually scheduled this week (drives who is on the panel)
    sched_set = set(games["away_team"].to_list()) | set(games["home_team"].to_list())
    # team -> (game_id, opponent_team) for THIS week's game (both away and home sides)
    team_map = (pl.concat([
        games.select(pl.col("away_team").alias("team"),
                     pl.col("home_team").alias("opponent_team"), "game_id"),
        games.select(pl.col("home_team").alias("team"),
                     pl.col("away_team").alias("opponent_team"), "game_id"),
    ], how="diagonal")
        .unique(subset=["team"], keep="first"))

    def _add_dummy(df: pl.DataFrame, key: str, stats: list[str]) -> pl.DataFrame:
        """Append one dummy row at ``week`` per ``key`` (last-played stats + target-week game)."""
        played = df.filter(pl.col("week") < week)
        if played.height == 0:
            return df
        # identity cols to carry on the dummy row (team needed for game assignment,
        # player_display_name/position needed by the feature pipeline)
        ident = [c for c in ["player_display_name", "position", "position_group", "team"]
                 if c in played.columns and c != key]
        last = (played.sort([key, "week"]).unique(subset=[key], keep="last")
                .select([key] + ident + [s for s in stats if s in played.columns]))
        # assign the target-week game for each key's team
        if "team" in last.columns:
            last = last.join(team_map, on="team", how="left")
        last = last.with_columns(pl.lit(week).alias("week"), pl.lit(season).alias("season"))
        return pl.concat([df, last], how="diagonal")

    # --- player panel: played weeks + dummy target row (last-played stats as dummy)
    players = (pw.filter((pl.col("season") == season)
                         & (pl.col("week") < week)
                         & pl.col("position").is_in(pos4))
               .select(["season", "week", "player_id", "player_display_name",
                        "position", "position_group", "team", "opponent_team", "game_id"]
                       + TARGETS))
    panel = _add_dummy(players, "player_id", TARGETS)

    # --- team stats: played weeks + dummy target row per scheduled team
    teams = team.filter((pl.col("season") == season) & (pl.col("week") < week))
    teams = _add_dummy(teams, "team",
                       ["passing_yards", "rushing_yards", "receiving_yards"])

    # --- NextGen: played weeks + dummy target row (last-played stats) per player
    def _ngs(df: pl.DataFrame, stats: list[str]) -> pl.DataFrame:
        d = df.filter((pl.col("season") == season) & (pl.col("week") < week))
        if d.height == 0:
            return d
        keep = [s for s in stats if s in d.columns]
        d = _add_dummy(d.select(["season", "week", "player_display_name"] + keep),
                       "player_display_name", keep)
        return d

    ngs = {
        "passing": _ngs(ngs_p, ["completion_percentage", "avg_time_to_throw",
                                "aggressiveness", "avg_completed_air_yards"]),
        "receiving": _ngs(ngs_r, ["avg_cushion", "avg_separation",
                                  "percent_share_of_intended_air_yards", "catch_percentage"]),
        "rushing": _ngs(ngs_ru, ["efficiency", "rush_yards_over_expected_per_att",
                                 "avg_time_to_los"]),
    }

    # --- run through the SAME feature pipeline, keep the target-week rows
    feat = F.build_features_from(panel, teams, ngs)
    out = feat.filter(pl.col("week") == week)

    # --- attach pre-game schedule context via game_id (dummy rows carry this
    #     week's game_id/opponent_team, so attach_schedule_context can derive
    #     home_flag / signed_spread / implied total / rest / weather)
    out = attach_schedule_context(out, sched.filter(pl.col("season") == season))

    # keep only players on a scheduled team (drop anyone not playing this week)
    out = out.filter(pl.col("team").is_in(sched_set))
    return out


def predict_week(season: int, week: int) -> list[dict]:
    """Predict every (player, position) row for a (season, week), main model.

    Returns a list of per-player dicts: player_id, player_display_name,
    position, team, preds {stat: value (clipped)}. Rows must exist in
    features.parquet (played weeks) OR be constructable as a scheduled-week
    panel (upcoming weeks, via _build_scheduled_panel).
    """
    payload = _load_payload()
    feat_cols = payload["meta"]["feature_columns"]
    models = payload["models"]

    # Try the existing (played) features first.
    feat = _load_features()
    rows = feat.filter((pl.col("season") == season) & (pl.col("week") == week))
    if rows.height == 0:
        # Upcoming week: build a scheduled panel from roster + schedule + the
        # existing trailing features, then select the model's feature columns.
        rows = _build_scheduled_panel(season, week)
        if rows.height == 0:
            raise SystemExit(
                f"no (season={season}, week={week}) rows and could not build a "
                f"scheduled panel (need 2026 roster/schedule data).")
    # Ensure the model's feature columns exist (they should, from the pipeline).
    missing = [c for c in feat_cols if c not in rows.columns]
    if missing:
        raise SystemExit(f"scheduled panel missing model feature cols: {missing}")
    X = rows.select(feat_cols).fill_null(0.0).to_numpy().astype(np.float32)
    dicts = rows.to_dicts()

    out = []
    for i, r in enumerate(dicts):
        pos = r["position"]
        targets = POS_TARGETS.get(pos, [])
        if not targets:
            continue
        preds = {}
        for t in targets:
            m = models.get(f"{pos}/{t}")
            if m is None:
                continue
            preds[t] = _clip(float(m.predict(X[i:i+1])[0]), t)
        out.append({
            "player_id": r["player_id"],
            "player_display_name": r["player_display_name"],
            "position": pos,
            "team": r.get("team"),
            "preds": preds,
        })
    return out


# --------------------------------------------------------------------------- #
# LOG (append-only, timestamped)
# --------------------------------------------------------------------------- #
def log_week(season: int, week: int) -> int:
    """Append this week's predictions to the holdout log. Returns n rows."""
    preds = predict_week(season, week)
    if not preds:
        print(f"[holdout] nothing to log for {season} w{week} (no scoring rows).")
        return 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        for p in preds:
            entry = {"season": season, "week": week, "logged_at": now,
                     "player_id": p["player_id"],
                     "player_display_name": p["player_display_name"],
                     "position": p["position"], "team": p["team"],
                     "preds": p["preds"]}
            f.write(json.dumps(entry, default=str) + "\n")
    print(f"[holdout] logged {len(preds)} players for {season} w{week} "
          f"at {now} -> {LOG_PATH}")
    return len(preds)


# --------------------------------------------------------------------------- #
# SCORE (read log + actuals; never writes the log)
# --------------------------------------------------------------------------- #
def _spearman(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _stats(pred, actual) -> dict:
    err = np.abs(np.asarray(pred) - np.asarray(actual))
    return {"n": int(len(actual)),
            "mae": round(float(err.mean()), 3),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
            "spearman": round(_spearman(pred, actual), 4)}


def score_week(season: int, week: int) -> dict:
    """Score the logged predictions for a (season, week) against actuals.

    Uses the LATEST logged entry per player (append-only log). Actuals come
    from features.parquet. Also reports the naive baselines (last_game /
    trailing3) on the same rows, computed the same way as src.bench.
    """
    if not LOG_PATH.exists():
        raise SystemExit(f"no holdout log at {LOG_PATH}; run `--log` first.")
    # Load + keep the latest entry per (season, week, player_id).
    entries = {}
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("season") != season or e.get("week") != week:
                continue
            key = (e["season"], e["week"], e["player_id"])
            # append-only => later lines are newer; overwrite to keep latest.
            entries[key] = e
    if not entries:
        raise SystemExit(f"no logged entries for {season} w{week}; run `--log` first.")

    feat = _load_features()
    actual = feat.filter((pl.col("season") == season) & (pl.col("week") == week))
    if actual.height == 0:
        raise SystemExit(
            f"no actuals in features.parquet for {season} w{week} yet — "
            f"the games haven't been pulled. Refresh after they're played.")

    # Baselines (last game / trailing3) from features, same as src.bench.
    # Must be computed over the FULL SEASON (shift(1) needs the prior week's
    # row), then keyed by (season, week, player_id) so each scored row matches
    # the right baseline. Keying by player_id alone would let a later week
    # overwrite an earlier one; filtering to one week would null the shift.
    stat_names = sorted({t for ts in POS_TARGETS.values() for t in ts})
    feat_season = (feat.filter(pl.col("season") == season)
                   .sort(["player_id", "season", "week"]))
    for s in stat_names:
        if s in feat_season.columns:
            feat_season = feat_season.with_columns(
                pl.col(s).shift(1).over(["player_id", "season"]).alias(s + "_bg"),
                (pl.col(s).shift(1).over(["player_id", "season"])
                 .rolling_mean(3, min_samples=1).over(["player_id", "season"])
                 .alias(s + "_bg3")))
    # (season, week, player_id) -> row maps.
    def _key(r):
        return (season, int(r["week"]), r["player_id"])
    actual_by = { _key(r): r for r in actual.to_dicts() }
    base_by = { _key(r): r for r in feat_season.to_dicts() }

    per_pos: dict[str, dict[str, list[float]]] = {}
    agg: dict[str, list[float]] = {"model": [], "actual": [],
                                   "last_game": [], "trailing3": []}
    n_scored = 0
    for (season_, week_, pid), e in entries.items():
        a = actual_by.get((season_, week_, pid))
        if a is None:
            continue
        pos = e["position"]
        targets = POS_TARGETS.get(pos, [])
        if not targets:
            continue

        # --- actual PPR ---
        actual_pts = sum(SCORE.get(s, 0.0) * float(a.get(s) or 0.0) for s in targets)
        # --- logged model PPR ---
        model_pts = sum(SCORE.get(s, 0.0) * float(e["preds"].get(s) or 0.0) for s in targets)
        # --- baselines ---
        b = base_by.get((season_, week_, pid), {})
        last_pts = sum(SCORE.get(s, 0.0) * float(b.get(s + "_bg") or 0.0) for s in targets)
        tr3_pts = sum(SCORE.get(s, 0.0) * float(b.get(s + "_bg3") or 0.0) for s in targets)

        agg["model"].append(model_pts)
        agg["actual"].append(actual_pts)
        agg["last_game"].append(last_pts)
        agg["trailing3"].append(tr3_pts)
        for name, val in (("model", model_pts), ("actual", actual_pts),
                          ("last_game", last_pts), ("trailing3", tr3_pts)):
            per_pos.setdefault(pos, {}).setdefault(name, []).append(val)
        n_scored += 1

    out = {
        "season": season, "week": week, "scoring": "standard 1-PPR",
        "n_logged": len(entries), "n_scored": n_scored,
        "logged_at": e.get("logged_at"),
        "note": ("clean holdout: prediction logged pre-kickoff (or replayed "
                 "for an already-played week from the frozen pre-2026 model); "
                 "scored against actuals after the games. Main model only "
                 "(no rookie-prior blend)."),
        "positions": {},
    }
    for pos in POS_TARGETS:
        if pos not in per_pos:
            continue
        d = per_pos[pos]
        out["positions"][pos] = {
            "n": len(d["actual"]),
            "actual_mean_pts": round(float(np.mean(d["actual"])), 2),
            "model": _stats(d["model"], d["actual"]),
            "last_game": _stats(d["last_game"], d["actual"]),
            "trailing3": _stats(d["trailing3"], d["actual"]),
        }
    if n_scored:
        out["aggregate"] = {
            "n": len(agg["actual"]),
            "actual_mean_pts": round(float(np.mean(agg["actual"])), 2),
            "model": _stats(agg["model"], agg["actual"]),
            "last_game": _stats(agg["last_game"], agg["actual"]),
            "trailing3": _stats(agg["trailing3"], agg["actual"]),
        }
    return out


def _print_score(out: dict) -> None:
    print("=" * 72)
    print(f"HOLDOUT SCORE  {out['season']} w{out['week']}  "
          f"(PPR; model vs baselines, n={out.get('n_scored',0)})")
    print(f"logged_at: {out.get('logged_at')}")
    print("=" * 72)
    hdr = f"{'POS':<4}{'n':>6}{'actμ':>9}   {'MODEL':>22}{'LAST':>22}{'TR3':>22}"
    print(hdr)
    print("-" * 72)
    for pos in POS_TARGETS:
        if pos not in out.get("positions", {}):
            continue
        d = out["positions"][pos]
        def fmt(s):
            return f"{s['mae']:6.2f}/{s['rmse']:<5.1f}/{s['spearman']:+.3f}"
        print(f"{pos:<4}{d['n']:>6}{d['actual_mean_pts']:>9}   "
              f"  {fmt(d['model'])}  {fmt(d['last_game'])}  {fmt(d['trailing3'])}")
    a = out.get("aggregate")
    if a:
        def fmt(s):
            return f"{s['mae']:6.2f}/{s['rmse']:<5.1f}/{s['spearman']:+.3f}"
        print("-" * 72)
        print(f"{'ALL':<4}{a['n']:>6}{a['actual_mean_pts']:>9}   "
              f"  {fmt(a['model'])}  {fmt(a['last_game'])}  {fmt(a['trailing3'])}")
        print("(cols = MAE / RMSE / Spearman; lower MAE/RMSE + higher Spearman is better)")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--log", action="store_true",
                    help="append this week's predictions to the holdout log "
                         "(run BEFORE kickoff)")
    ap.add_argument("--score", action="store_true",
                    help="score the logged predictions against actuals "
                         "(run AFTER the games)")
    args = ap.parse_args(argv)
    if not args.log and not args.score:
        ap.error("specify --log and/or --score")

    if args.log:
        log_week(args.season, args.week)
    if args.score:
        out = score_week(args.season, args.week)
        _print_score(out)
        # Merge into the score file (one record per season-week).
        SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if SCORE_PATH.exists():
            try:
                existing = json.loads(SCORE_PATH.read_text())
            except Exception:
                existing = {}
        if not isinstance(existing, dict) or "weeks" not in existing:
            existing = {"weeks": {}}
        existing["weeks"][f"{args.season}_w{args.week:02d}"] = out
        SCORE_PATH.write_text(json.dumps(existing, indent=2, default=str))
        print(f"-> {SCORE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
