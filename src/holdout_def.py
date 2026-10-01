"""D/ST holdout log + score for the LIVE season (2026).

Companion to :mod:`src.holdout` (offense, per player). D/ST is a different
unit: one row per TEAM per week, ESPN Standard D/ST scoring, and the shipped
prediction is the MATCHUP baseline (rank by the opponent's Vegas implied team
total, lowest first — see the "Top-defense (D/ST) projection" A/B in
NOTES.md and notebooks/_def_*.py).

Two steps (same discipline as the offense holdout — log BEFORE kickoff, score
AFTER the games):

  1. **log**   — write this week's D/ST board (per team: projected points
                 allowed, ESPN tier points, total D/ST points) with a
                 wall-clock timestamp to ``models/holdout_log_def.jsonl``
                 (append-only). The board is deterministic from pre-kickoff
                 data only (Vegas lines + flat league means), so logging is
                 the clean act.

  2. **score** — read the log, compute ACTUAL D/ST points per team from the
                 raw data (schedule final score → points allowed;
                 ``team_weekly_stats`` → turnovers / sacks / D-TDs), and
                 report per-team + aggregate ranking quality (Spearman,
                 top-5 overlap, top-5 mean vs rest).

Design rules (keep the holdout honest):
  * The log is APPEND-ONLY; scoring uses the LATEST entry per (season, week,
    team) — same rule as the offense log.
  * ``score`` never writes to the log; it only reads it + reads raw actuals.
  * Both log and score compute the board with the SAME code path, so the
    logged prediction and the scored prediction are identical.

Usage
-----
    # BEFORE the games: log this week's D/ST board (the clean act)
    uv run python -m src.holdout_def --season 2026 --week 4 --log

    # AFTER the games (once actuals are in data/raw/): score it
    uv run python -m src.holdout_def --season 2026 --week 4 --score

Run:  uv run python -m src.holdout_def --help
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import polars as pl

from src.clean import RAW_DIR
from src.model import MODELS_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = MODELS_DIR / "holdout_log_def.jsonl"
SCORE_PATH = MODELS_DIR / "holdout_score_def.json"
METHOD = ("matchup-baseline-v1: rank by opponent's Vegas implied team total, "
          "lowest first; ESPN Standard D/ST scoring (+ flat league-mean sacks "
          "and turnovers for the absolute value)")

# Turnover components (all count as +2 D/ST points each in ESPN Standard).
TO_COLS = ["def_interceptions", "def_fumbles", "def_safeties",
           "def_punt_blocks", "def_pat_blocks", "def_fg_blocks"]


# --------------------------------------------------------------------------- #
# Scoring (ESPN Standard D/ST)
# --------------------------------------------------------------------------- #
def tier(allowed: float) -> int:
    """ESPN Standard D/ST points-allowed tier."""
    if allowed <= 6: return 10
    if allowed <= 13: return 7
    if allowed <= 17: return 4
    if allowed <= 23: return 1
    if allowed <= 30: return 0
    return -1


def dst_points(allowed: float, to: float, sacks: float, dst_td: float) -> float:
    return tier(allowed) + 2.0 * to + 1.0 * sacks + 6.0 * dst_td


def _league_means() -> tuple[float, float]:
    """(sacks/game, turnovers/game) means from 2016-25 history, for the flat
    components of the projected D/ST value. These are near-constants: they set
    the absolute value but do NOT change the ranking (every team gets the same
    flat addition)."""
    t = pl.read_parquet(RAW_DIR / "team_weekly_stats.parquet")
    t = t.filter(pl.col("season").is_in(range(2016, 2026)))
    for c in TO_COLS:
        t = t.with_columns(pl.col(c).fill_null(0.0))
    sacks = float(t["def_sacks"].fill_null(0.0).mean())
    to = float(t.select(pl.sum_horizontal(TO_COLS).alias("to"))["to"].mean())
    return sacks, to


# --------------------------------------------------------------------------- #
# Board (the shipped prediction) — deterministic from pre-kickoff data only
# --------------------------------------------------------------------------- #
def build_board(season: int, week: int,
                sacks_mean: float, to_mean: float) -> list[dict]:
    """The D/ST board for a (season, week), ranked by the OPPONENT's Vegas
    implied team total (lowest = best). ESPN Standard scoring on the projected
    points-allowed, plus the flat league-mean sack/turnover components.

    spread_line is from the AWAY team's perspective (positive = away underdog),
    so: away implied = (total - spread)/2, home implied = (total + spread)/2.
    A team's projected points-allowed is its OPPONENT's implied total.
    """
    sched = pl.read_parquet(RAW_DIR / "schedules.parquet").filter(
        (pl.col("season") == season) & (pl.col("week") == week))
    if sched.height == 0:
        raise SystemExit(f"no games in the schedule for {season} w{week}.")
    sched = sched.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64)
                                for c in ["spread_line", "total_line"]])

    rows: list[dict] = []
    for g in sched.to_dicts():
        if not g["total_line"] or not g["spread_line"]:
            continue
        total, spread = g["total_line"], g["spread_line"]
        # team = AWAY -> opponent is HOME, whose implied total is (t + s)/2
        rows.append(dict(team=g["away_team"], opp=g["home_team"],
                         proj_allwd=round((total + spread) / 2, 1)))
        # team = HOME -> opponent is AWAY, whose implied total is (t - s)/2
        rows.append(dict(team=g["home_team"], opp=g["away_team"],
                         proj_allwd=round((total - spread) / 2, 1)))
    if not rows:
        raise SystemExit(
            f"no Vegas lines for {season} w{week} — can't build the D/ST board.")

    for r in rows:
        r["allowed_pts"] = tier(r["proj_allwd"])
        r["d_st"] = round(r["allowed_pts"] + sacks_mean + 2.0 * to_mean, 1)
    # rank: higher D/ST points first; tie-break = fewer projected points allowed
    rows.sort(key=lambda x: (-x["d_st"], x["proj_allwd"]))
    for i, r in enumerate(rows, 1):
        r["board_rank"] = i
    return rows


# --------------------------------------------------------------------------- #
# Actuals (from raw data, after the games)
# --------------------------------------------------------------------------- #
def actual_dst(season: int, week: int) -> dict[str, dict]:
    """Actual D/ST points per team for a (season, week), from raw data.

    points-allowed = the team's OPPONENT's final score (schedules);
    turnovers / sacks / D-TDs = team_weekly_stats. ESPN Standard scoring.
    Returns {team: {allowed, to, sacks, dst_td, d_st}}.
    """
    t = pl.read_parquet(RAW_DIR / "team_weekly_stats.parquet").filter(
        (pl.col("season") == season) & (pl.col("week") == week))
    t = t.select(["team"] + TO_COLS + ["def_sacks", "def_tds"])
    for c in TO_COLS + ["def_sacks", "def_tds"]:
        t = t.with_columns(pl.col(c).fill_null(0.0))
    t = t.with_columns(pl.sum_horizontal(TO_COLS).alias("act_to"))
    t = t.select(["team", "act_to", "def_sacks", "def_tds"])

    sched = pl.read_parquet(RAW_DIR / "schedules.parquet").filter(
        (pl.col("season") == season) & (pl.col("week") == week))
    sched = sched.select(["away_team", "away_score", "home_team", "home_score"])
    sched = sched.with_columns([pl.col(c).cast(pl.Float64)
                                for c in ["away_score", "home_score"]])
    allowed = pl.concat([
        sched.select(["away_team", "home_score"]).rename(
            {"away_team": "team", "home_score": "allowed"}),
        sched.select(["home_team", "away_score"]).rename(
            {"home_team": "team", "away_score": "allowed"}),
    ], how="vertical")

    if t.height == 0:
        raise SystemExit(
            f"no team stats in data/raw for {season} w{week} — the games "
            f"haven't been pulled. Re-run `predict --refresh` (or "
            f"`src.download_data --force`) after they're played.")

    merged = t.join(allowed, on="team", how="left")
    out: dict[str, dict] = {}
    for r in merged.to_dicts():
        if r["allowed"] is None:
            continue
        a = float(r["allowed"]); to = float(r["act_to"])
        s = float(r["def_sacks"]); d = float(r["def_tds"])
        out[r["team"]] = dict(allowed=a, to=to, sacks=s, dst_td=d,
                              d_st=round(dst_points(a, to, s, d), 1))
    return out


# --------------------------------------------------------------------------- #
# LOG (append-only, timestamped)
# --------------------------------------------------------------------------- #
def log_week(season: int, week: int) -> int:
    """Append this week's D/ST board to the holdout log. Returns n teams."""
    sacks_mean, to_mean = _league_means()
    board = build_board(season, week, sacks_mean, to_mean)
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        for r in board:
            entry = dict(season=season, week=week, logged_at=now,
                         method=METHOD, **r)
            f.write(json.dumps(entry, default=str) + "\n")
    print(f"[holdout_def] logged {len(board)} teams for {season} w{week} "
          f"at {now} -> {LOG_PATH}")
    top = ", ".join(f'{r["team"]}({r["proj_allwd"]})' for r in board[:5])
    print(f"  top-5 board: {top}")
    return len(board)


# --------------------------------------------------------------------------- #
# SCORE (read log + actuals; never writes the log)
# --------------------------------------------------------------------------- #
def _spearman(a, b) -> float:
    import numpy as np
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def score_week(season: int, week: int) -> dict:
    """Score the logged D/ST board for (season, week) against actuals."""
    if not LOG_PATH.exists():
        raise SystemExit(f"no D/ST holdout log at {LOG_PATH}; run `--log` first.")
    entries: dict[str, dict] = {}
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("season") != season or e.get("week") != week:
                continue
            entries[e["team"]] = e  # append-only => later lines are newer
    if not entries:
        raise SystemExit(
            f"no logged D/ST board for {season} w{week}; run `--log` first.")

    actual = actual_dst(season, week)
    common = [tm for tm in entries if tm in actual]
    if not common:
        raise SystemExit(
            f"logged board and actuals share no teams for {season} w{week}.")

    rows = []
    for tm in common:
        p = entries[tm]; a = actual[tm]
        rows.append(dict(team=tm, opp=p.get("opp"),
                         board_rank=int(p["board_rank"]),
                         proj_allwd=float(p["proj_allwd"]),
                         pred_d_st=float(p["d_st"]),
                         act_allowed=a["allowed"], act_to=a["to"],
                         act_sacks=a["sacks"], act_dst_td=a["dst_td"],
                         act_d_st=a["d_st"]))
    # actual rank (1 = best D/ST output)
    order = sorted(rows, key=lambda r: (-r["act_d_st"], r["act_allowed"]))
    for i, r in enumerate(order, 1):
        r["actual_rank"] = i

    pred = [r["pred_d_st"] for r in rows]
    act = [r["act_d_st"] for r in rows]
    rho = _spearman(pred, act)

    # top-5 (by board) vs rest
    top5 = [r for r in rows if r["board_rank"] <= 5]
    rest = [r for r in rows if r["board_rank"] > 5]
    top5_mean = float(sum(r["act_d_st"] for r in top5) / len(top5)) if top5 else float("nan")
    rest_mean = float(sum(r["act_d_st"] for r in rest) / len(rest)) if rest else float("nan")
    top5_actual_ranks = sorted(r["actual_rank"] for r in top5)

    out = dict(
        season=season, week=week, scoring="ESPN Standard D/ST", method=METHOD,
        n_logged=len(entries), n_scored=len(rows),
        logged_at=entries[common[0]].get("logged_at"),
        spearman_pred_vs_act=round(rho, 4),
        top5_by_board=[r["team"] for r in sorted(top5, key=lambda x: x["board_rank"])],
        top5_mean_act_d_st=round(top5_mean, 2),
        rest_mean_act_d_st=round(rest_mean, 2),
        top5_actual_ranks=top5_actual_ranks,
        note=("clean holdout: D/ST board logged pre-kickoff (deterministic "
              "from Vegas lines + flat league means), scored against actuals "
              "after the games. Higher top5_mean + lower top5_actual_ranks = "
              "the board's top-5 outperformed the rest."),
        teams=sorted(rows, key=lambda r: r["board_rank"]),
    )
    return out


def _print_score(out: dict) -> None:
    print("=" * 78)
    print(f"D/ST HOLDOUT SCORE  {out['season']} w{out['week']}  "
          f"(ESPN Standard, n={out['n_scored']})")
    print(f"method: {out['method']}")
    print("=" * 78)
    print(f"{'RK':>3} {'DEF':5} {'vs':5} {'projAllwd':>10} {'predDST':>8} | "
          f"{'actAllwd':>9} {'TO':>3} {'sack':>4} {'dTD':>3} {'actDST':>7} {'actRK':>6}")
    print("-" * 78)
    for r in out["teams"]:
        print(f"{r['board_rank']:>3} {r['team']:5} {str(r['opp']):5} "
              f"{r['proj_allwd']:>10} {r['pred_d_st']:>8} | "
              f"{r['act_allowed']:>9} {r['act_to']:>3.0f} {r['act_sacks']:>4.0f} "
              f"{r['act_dst_td']:>3.0f} {r['act_d_st']:>7} {r['actual_rank']:>6}")
    print("-" * 78)
    print(f"Spearman(pred vs actual D/ST) = {out['spearman_pred_vs_act']:+.3f}")
    print(f"top-5 (by board): {', '.join(out['top5_by_board'])}")
    print(f"  top-5 mean actual D/ST = {out['top5_mean_act_d_st']:.2f}   "
          f"rest mean = {out['rest_mean_act_d_st']:.2f}   "
          f"(gap {out['top5_mean_act_d_st'] - out['rest_mean_act_d_st']:+.2f})")
    print(f"  top-5 actual ranks = {out['top5_actual_ranks']}  (lower = better)")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--log", action="store_true",
                    help="append this week's D/ST board to the holdout log "
                         "(run BEFORE kickoff)")
    ap.add_argument("--score", action="store_true",
                    help="score the logged board against actuals (run AFTER "
                         "the games)")
    args = ap.parse_args(argv)
    if not args.log and not args.score:
        ap.error("specify --log and/or --score")

    if args.log:
        log_week(args.season, args.week)
    if args.score:
        out = score_week(args.season, args.week)
        _print_score(out)
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
