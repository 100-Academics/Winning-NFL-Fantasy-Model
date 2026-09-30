"""Head-to-head: OUR model  vs  FANTASYPROS consensus  (same 2024 test weeks).

Uses the cached FantasyPros free-tier weekly projections (top-10/position,
fetched by src.fetch_fantasypros) and compares them, stat-for-stat, against
our saved model's predictions on the SAME (season, week, position, player).
Both sides project the same raw stat components, so there's no scoring-scheme
ambiguity. We also report the same standard-1-PPR point conversion as bench.py.

Only players present in BOTH our features and the FP top-10 board for a given
(week, position) are compared — that's the fair head-to-head set (FP only
projects its top-10 per position on the free tier).

Run:  uv run python -m src.compare_fantasypros
"""
from __future__ import annotations

import json
import pathlib
import re

import joblib
import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.clean import PROCESSED_DIR
from src.model import POS_TARGETS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
FP_DIR = REPO_ROOT / "data" / "processed" / "fantasypros"

# FantasyPros stat key -> our stat column name (per-position).
STAT_MAP = {
    "QB": {"pass_yds": "passing_yards", "pass_tds": "passing_tds",
           "pass_ints": "passing_interceptions"},
    "RB": {"rush_yds": "rushing_yards", "rush_tds": "rushing_tds",
           "rec_rec": "receptions", "rec_yds": "receiving_yards",
           "rec_tds": "receiving_tds"},
    "WR": {"rec_rec": "receptions", "rec_yds": "receiving_yards",
           "rec_tds": "receiving_tds"},
    "TE": {"rec_rec": "receptions", "rec_yds": "receiving_yards",
           "rec_tds": "receiving_tds"},
}
# Standard 1-PPR (for the points-level readout; matches bench.py).
SCORE = {"passing_yards": .1, "rushing_yards": .1, "receiving_yards": .1,
         "passing_tds": 6, "rushing_tds": 6, "receiving_tds": 6,
         "receptions": 1, "passing_interceptions": -2}

SUFFIXES = [" II", " III", " IV", " Jr.", " Sr."]


def _norm(name: str) -> str:
    s = name.strip()
    changed = True
    while changed:
        changed = False
        for suf in SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return re.sub(r"\s+", " ", s).lower()


def _spearman(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _stats(pred, actual) -> dict:
    e = np.abs(np.asarray(pred) - np.asarray(actual))
    return {"n": int(len(actual)),
            "mae": round(float(e.mean()), 3),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 3),
            "spearman": round(_spearman(pred, actual), 4)}


def _load_fp_board(season: int, week: int, pos: str) -> dict | None:
    p = FP_DIR / f"{season}_w{week:02d}_{pos}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    out = {}
    for plr in d.get("players", []):
        st = plr.get("stats", {}) or {}
        out[_norm(plr["name"])] = st
    return out


def main() -> None:
    payload = joblib.load(MODELS_DIR / "models.joblib")
    feat_cols = payload["meta"]["feature_columns"]
    models = payload["models"]

    feat = (pl.read_parquet(PROCESSED_DIR / "features.parquet")
            .filter(pl.col("season") == 2024))
    # Determine which (week,pos) boards exist and are complete.
    weeks = sorted({int(re.match(r"2024_w(\d+)_", p.name).group(1))
                    for p in FP_DIR.glob("2024_w*_*.json")})
    complete = [w for w in weeks
                if all((FP_DIR / f"2024_w{w:02d}_{pos}.json").exists()
                       for pos in POS_TARGETS)]
    print(f"Using complete 2024 weeks: {complete}\n")
    # Per (week, position): load feature rows once, index by normalized name.
    rows_by_wkpos: dict[tuple[int, str], dict[str, dict]] = {}
    for pos in POS_TARGETS:
        for w in complete:
            sub = feat.filter((pl.col("week") == w) & (pl.col("position") == pos))
            rows_by_wkpos[(w, pos)] = {
                _norm(r["player_display_name"]): r
                for r in sub.to_dicts()
            }

    per_pos: dict[str, dict[str, dict[str, list[float]]]] = {}
    n_rows = 0
    for pos in POS_TARGETS:
        smap = STAT_MAP[pos]
        rows = rows_by_wkpos.get
        for w in complete:
            board = _load_fp_board(2024, w, pos)
            if not board:
                continue
            rows_for_wk = rows((w, pos))
            for fp_name, st in board.items():
                rec = rows_for_wk.get(fp_name)
                if rec is None:
                    continue
                for fp_key, our_stat in smap.items():
                    if fp_key not in st or st[fp_key] is None:
                        continue
                    m = models.get(f"{pos}/{our_stat}")
                    if m is None or rec.get(our_stat) is None:
                        continue
                    X = np.array([[0.0 if rec.get(c) is None else float(rec.get(c))
                                  for c in feat_cols]], dtype=np.float32)
                    our = float(m.predict(X)[0])
                    actual = float(rec[our_stat])
                    fp = float(st[fp_key])
                    acc = per_pos.setdefault(pos, {}).setdefault(our_stat, {})
                    acc.setdefault("ours", []).append(our)
                    acc.setdefault("fp", []).append(fp)
                    acc.setdefault("actual", []).append(actual)
                    n_rows += 1

    # Summarize: per (position, stat) our-MAE vs FP-MAE vs spearman.
    print(f"{'pos':<3}{'stat':<22}{'n':>5}  {'OURS':>10}{'FANTASYPROS':>14}"
          f"  {'ours-rho':>9}{'fp-rho':>8}")
    print("-" * 84)
    agg = {"ours": [], "fp": [], "actual": []}
    detail = {}
    for pos in POS_TARGETS:
        if pos not in per_pos:
            continue
        for stat in POS_TARGETS[pos]:
            if stat not in per_pos[pos]:
                continue
            a = per_pos[pos][stat]
            ours, fp, actual = a["ours"], a["fp"], a["actual"]
            so = _stats(ours, actual)
            sf = _stats(fp, actual)
            print(f"{pos:<3}{stat:<22}{so['n']:>5}"
                  f"  {so['mae']:>6.2f}/{so['rmse']:<3.1f}"
                  f" {sf['mae']:>6.2f}/{sf['rmse']:<4.1f}"
                  f" {so['spearman']:>9.3f}{sf['spearman']:>8.3f}")
            agg["ours"] += ours
            agg["fp"] += fp
            agg["actual"] += actual
            detail[f"{pos}/{stat}"] = {"n": so["n"], "ours": so, "fantasypros": sf}
    print("-" * 84)
    so = _stats(agg["ours"], agg["actual"])
    sf = _stats(agg["fp"], agg["actual"])
    print(f"{'AGG':<3}{'all stats':<22}{so['n']:>5}"
          f"  {so['mae']:>6.2f}/{so['rmse']:<3.1f}"
          f" {sf['mae']:>6.2f}/{sf['rmse']:<4.1f}"
          f" {so['spearman']:>9.3f}{sf['spearman']:>8.3f}")

    # Points-level (standard 1-PPR) readout per position.
    print(f"\nStandard 1-PPR weekly points (2024 test, head-to-head):")
    print(f"{'pos':<3}{'n':>5}  {'OURS pts':>12}{'FANTASYPROS':>14}{'rho ours':>9}{'rho FP':>8}")
    print("-" * 78)
    for pos in POS_TARGETS:
        if pos not in per_pos:
            continue
        rows = per_pos[pos]
        sstats = [s for s in rows if rows[s]]
        if not sstats:
            continue
        n = len(rows[sstats[0]]["ours"])
        o_pts = np.zeros(n)
        f_pts = np.zeros(n)
        a_pts = np.zeros(n)
        for s in sstats:
            o_pts += np.array(rows[s]["ours"]) * SCORE.get(s, 0)
            f_pts += np.array(rows[s]["fp"]) * SCORE.get(s, 0)
            a_pts += np.array(rows[s]["actual"]) * SCORE.get(s, 0)
        so = _stats(o_pts, a_pts)
        sf = _stats(f_pts, a_pts)
        print(f"{pos:<3}{so['n']:>5}  {so['mae']:>6.2f}/{so['rmse']:<3.1f}"
              f" {sf['mae']:>6.2f}/{sf['rmse']:<4.1f}"
              f" {so['spearman']:>9.3f}{sf['spearman']:>8.3f}")

    out = {"season": 2024, "weeks": complete, "n_rows": n_rows,
          "stat_level": detail, "note": "head-to-head on FP free-tier top-10/pos "
          "board; both sides project the same raw stat components"}
    (MODELS_DIR / "fantasypros_report.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {MODELS_DIR / 'fantasypros_report.json'}")


if __name__ == "__main__":
    main()
