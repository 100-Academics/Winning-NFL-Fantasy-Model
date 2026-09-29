"""Phase 5 — prediction CLI.

Turns the saved models (Phase 4) + features (Phase 3) into human-readable
per-player projections, e.g.

    Patrick Mahomes (QB, KC @ NO) will get ~2.1 passing TDs and ~268
    passing yards (198-340).

Usage:
    # specific players for a week
    python -m src.predict --season 2024 --week 5 --players "Patrick Mahomes" "Bryan Johnson"

    # a ranked board (top 15 per position) for a week
    python -m src.predict --season 2024 --week 5 --all

    # compare predictions against actuals (end-to-end sanity test)
    python -m src.predict --season 2024 --week 5 --all --compare --top 5

    # machine-readable
    python -m src.predict --season 2024 --week 5 --all --json

    # upcoming week: re-pull the season's data + rebuild features first
    python -m src.predict --season 2026 --week 6 --all --refresh

Design notes:
  * The (season, week) row must exist in the features file. For a *past* week
    that's already true. For an *upcoming* week, run with --refresh to pull the
    latest data and rebuild features first.
  * Features are the model's input; the current-week target columns (present in
    the same row) are used ONLY by --compare, never fed to the model.
  * Bands (p10-p90) are shown for the headline stats of each position.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import polars as pl

from src.model import (
    MODELS_DIR,
    POS_TARGETS,
    HEADLINE,
)
from src.clean import PROCESSED_DIR

# Human phrasing per position: (target, label)
PHRASE: dict[str, list[tuple[str, str]]] = {
    "QB": [("passing_tds", "passing TDs"),
           ("passing_yards", "passing yards"),
           ("passing_interceptions", "interceptions")],
    "RB": [("rushing_yards", "rushing yards"),
           ("rushing_tds", "rushing TDs"),
           ("receptions", "receptions"),
           ("receiving_yards", "receiving yards"),
           ("receiving_tds", "receiving TDs")],
    "WR": [("receptions", "receptions"),
           ("receiving_yards", "receiving yards"),
           ("receiving_tds", "receiving TDs")],
    "TE": [("receptions", "receptions"),
           ("receiving_yards", "receiving yards"),
           ("receiving_tds", "receiving TDs")],
}

# Headline stat used to rank a position's board.
RANK_STAT = {"QB": "passing_yards", "RB": "rushing_yards",
             "WR": "receiving_yards", "TE": "receiving_yards"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_payload() -> dict:
    import joblib
    path = MODELS_DIR / "models.joblib"
    if not path.exists():
        raise SystemExit(
            f"models not found at {path}. Run `python -m src.model` first (Phase 4)."
        )
    return joblib.load(path)


def _load_features() -> pl.DataFrame:
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise SystemExit(
            f"features not found at {path}. Run `python -m src.features` first (Phase 3)."
        )
    return pl.read_parquet(path)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _predict_row(rec: dict, payload: dict, feat_cols: list[str]) -> dict:
    """Predict all targets for a single (player, season, week) record (dict)."""
    pos = rec["position"]
    X = np.array([rec.get(c) for c in feat_cols], dtype=object)
    # fill nulls (None) -> 0.0
    X = np.array([0.0 if v is None else float(v) for v in X], dtype=np.float32).reshape(1, -1)
    out: dict = {"position": pos, "preds": {}, "bands": {}}
    for target in POS_TARGETS.get(pos, []):
        key = f"{pos}/{target}"
        m = payload["models"].get(key)
        if m is None:
            continue
        out["preds"][target] = float(m.predict(X)[0])
        bands = payload.get("bands", {}).get(key)
        if bands and target in HEADLINE.get(pos, set()):
            out["bands"][target] = (float(bands["p10"].predict(X)[0]),
                                    float(bands["p90"].predict(X)[0]))
    return out


def _fmt_num(x: float, stat: str) -> str:
    # Round yards to whole, counts to 1 decimal (e.g. "2.1 TDs").
    if stat.endswith(("yards",)):
        return f"{x:.0f}"
    return f"{x:.1f}"


def _phrase(pos: str, preds: dict, bands: dict) -> str:
    parts = []
    for target, label in PHRASE.get(pos, []):
        if target not in preds:
            continue
        val = preds[target]
        s = f"{_fmt_num(val, target)} {label}"
        if target in bands:
            lo, hi = bands[target]
            s += f" ({_fmt_num(lo, target)}-{_fmt_num(hi, target)})"
        parts.append(s)
    # join with commas, last with "and"
    if not parts:
        return "n/a"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _select_rows(feat: pl.DataFrame, season: int, week: int,
                 players: list[str] | None, all_players: bool,
                 top: int) -> pl.DataFrame:
    rows = feat.filter((pl.col("season") == season) & (pl.col("week") == week))
    if rows.height == 0:
        return rows
    if players:
        like = [p.lower() for p in players]
        # OR of lower-cased substring matches
        mask = pl.col("player_display_name").str.to_lowercase().str.contains(like[0], literal=True)
        for p in like[1:]:
            mask = mask | pl.col("player_display_name").str.to_lowercase().str.contains(p, literal=True)
        rows = rows.filter(mask)
        return rows
    # --all: return the full week; ranking + top-N is done by the caller AFTER
    # prediction (rank by the model's predicted headline stat, not trailing form).
    return rows


def _head_stat(entry: dict) -> float:
    stat = RANK_STAT.get(entry["position"], "passing_yards")
    return float(entry["preds"].get(stat, 0.0))


def _head_error_stats(entries: list[dict]) -> dict:
    """MAE/RMSE of the headline stat across entries (only those with actuals)."""
    errs = []
    for e in entries:
        if not e.get("actuals"):
            continue
        stat = RANK_STAT.get(e["position"], "passing_yards")
        if stat in e["actuals"]:
            errs.append(abs(e["preds"][stat] - e["actuals"][stat]))
    if not errs:
        return {"n": 0, "mae": None, "rmse": None}
    mae = float(np.mean(errs))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    return {"n": len(errs), "mae": round(mae, 2), "rmse": round(rmse, 2)}


def _build_prediction(rec: dict, payload: dict, feat_cols: list[str],
                      compare: bool) -> dict:
    p = _predict_row(rec, payload, feat_cols)
    entry = {
        "player": rec["player_display_name"],
        "position": rec["position"],
        "team": rec["team"],
        "opponent": rec["opponent_team"],
        "season": int(rec["season"]),
        "week": int(rec["week"]),
        "home": bool(rec["home_flag"]),
        "preds": p["preds"],
        "bands": p["bands"],
        "sentence": None,
    }
    if compare:
        actuals = {t: float(rec[t]) for t in p["preds"] if t in rec and rec[t] is not None}
        entry["actuals"] = actuals
        entry["error"] = {t: round(p["preds"][t] - a, 3)
                          for t, a in actuals.items()}
    return entry


# --------------------------------------------------------------------------- #
# Refresh (upcoming-week path)
# --------------------------------------------------------------------------- #
def _refresh(season: int) -> None:
    """Re-pull data for the current season (and the prior one, for context) and
    rebuild panel + features. Used when predicting a week not yet in the data."""
    print(f"[refresh] pulling seasons {season-1}..{season} and rebuilding features...",
          file=sys.stderr)
    from src.download_data import main as dl_main
    # pull the two most recent seasons we care about (prior season gives
    # cross-season trailing context; the current season gives this week's matchup)
    dl_main(["--seasons", str(season - 1), str(season), "--datasets",
             "core,context,advanced", "--force"])
    from src.features import build_features
    # build_features reads ALL seasons present in raw; idempotent.
    build_features()
    print("[refresh] done.", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True, help="season year, e.g. 2024")
    ap.add_argument("--week", type=int, required=True, help="week number, e.g. 5")
    ap.add_argument("--players", type=str, nargs="*", default=None,
                    help="player name(s) to predict (substring match)")
    ap.add_argument("--all", action="store_true",
                    help="predict a ranked board of all scoring players")
    ap.add_argument("--top", type=int, default=15,
                    help="with --all, top N per position (0 = all). Default 15.")
    ap.add_argument("--compare", action="store_true",
                    help="show actuals + error (requires the week to be in the data)")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull the season's data + rebuild features first "
                         "(for upcoming weeks)")
    args = ap.parse_args(argv)

    if not args.players and not args.all:
        ap.error("specify --players \"Name...\" and/or --all")

    if args.refresh:
        _refresh(args.season)

    payload = _load_payload()
    feat_cols = payload["meta"]["feature_columns"]
    feat = _load_features()
    rows = _select_rows(feat, args.season, args.week,
                        args.players if args.players else None,
                        args.all, args.top)

    if rows.height == 0:
        msg = (f"no (season={args.season}, week={args.week}) rows in the features "
               f"file. If this is an upcoming week, re-run with --refresh.")
        if args.players:
            msg += f" (also: none of {args.players} matched)"
        print(msg, file=sys.stderr)
        return 2

    entries = []
    for r in rows.iter_rows(named=True):
        entry = _build_prediction(r, payload, feat_cols, args.compare)
        entry["sentence"] = _phrase(entry["position"], entry["preds"], entry["bands"])
        entries.append(entry)

    if args.all:
        # rank by predicted headline stat; keep top N per position
        by_pos: dict[str, list[dict]] = {}
        for e in entries:
            by_pos.setdefault(e["position"], []).append(e)
        for p in by_pos:
            by_pos[p].sort(key=_head_stat, reverse=True)
        pos_order = [p for p in POS_TARGETS if p in by_pos]
        limit = args.top if args.top > 0 else len(by_pos)
        entries = [e for p in pos_order for e in by_pos[p][:limit]]
    else:
        # named players: keep them grouped by position, best first
        by_pos: dict[str, list[dict]] = {}
        for e in entries:
            by_pos.setdefault(e["position"], []).append(e)
        pos_order = [p for p in POS_TARGETS if p in by_pos]
        entries = [e for p in pos_order for e in sorted(by_pos[p], key=lambda e: _head_stat(e), reverse=True)]

    if args.json:
        if args.compare:
            entries.append({"headline_summary": _head_error_stats(entries)})
        print(json.dumps(entries, indent=2, default=str))
        return 0

    # human output
    if args.compare:
        summ = _head_error_stats(entries)
        if summ["n"]:
            print(f"\nWeek {args.week} {args.season} — headline-stat error "
                  f"across {summ['n']} players: MAE {summ['mae']}, RMSE {summ['rmse']}\n",
                  file=sys.stderr)
    for e in entries:
        venue = f"{e['team']} {'vs' if e['home'] else '@'} {e['opponent']}"
        print(f"\n{e['player']} ({e['position']}, {venue}) — Week {e['week']}, {e['season']}")
        print(f"  {e['sentence']}")
        if e.get("actuals") is not None:
            act = ", ".join(f"{_fmt_num(v, s)} {s.replace('_',' ')}"
                            for s, v in e["actuals"].items())
            err = ", ".join(f"{s.replace('_',' ')} {e['error'][s]:+.1f}"
                            for s in e["actuals"])
            print(f"  ACTUAL: {act}")
            print(f"  ERROR : {err}")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
