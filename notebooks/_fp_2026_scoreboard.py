"""Cumulative LIVE 2026 head-to-head scoreboard: OUR model  vs  FANTASYPROS.

The "Is it good?" battery (2026-11-01) explicitly flagged the one untested
item: "FantasyPros head-to-head on live 2026 weeks ... not attempted." This is
that scoreboard, run on the SCORED live 2026 weeks (the only true out-of-sample
data for the shipped model — it was trained through 2025).

Method (apples-to-apples, mirrors src.compare_fantasypros for 2024):
  - For each scored week x position, compare the FANTASYPROS free-tier top-10
    board against OUR logged pre-kickoff predictions (the clean live record in
    models/holdout_log.jsonl) on the OVERLAPPING players (FP only projects its
    top-10/position on the free tier). Both sides are pre-week commitments, so
    it's a fair head-to-head.
  - Actual per-stat values come from data/processed/features.parquet (the same
    source the 2024 script uses), so the scoring is unambiguous.
  - Metrics: raw-stat MAE (apples-to-apples components), ESPN-Standard points
    MAE/RMSE/Spearman, and a per-week top-5 decision value, each per position
    and pooled, plus a per-week trend.

Run:  uv run python notebooks/_fp_2026_scoreboard.py
Out:  models/fantasypros_2026_report.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.clean import PROCESSED_DIR
from src.compare_fantasypros import STAT_MAP, _norm, _load_fp_board
from src.model import POS_TARGETS
from src.scoring import ESPN_STANDARD as SCORE

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
SEASON = 2026
TOP_K = 5  # decision-value slice


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _stats(pred, actual) -> dict:
    e = np.abs(np.asarray(pred, float) - np.asarray(actual, float))
    return {"n": int(len(actual)),
            "mae": round(float(e.mean()), 3),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 3),
            "spearman": round(_spearman(pred, actual), 4)}


def _scored_weeks() -> list[int]:
    d = json.loads((MODELS_DIR / "holdout_score.json").read_text())
    return sorted(int(k.split("_w")[1]) for k in d["weeks"]
                  if k.startswith(f"{SEASON}_w"))


def _logged_preds(week: int, pos: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in (MODELS_DIR / "holdout_log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("season") == SEASON and o.get("week") == week and o.get("position") == pos:
            out[_norm(o["player_display_name"])] = o.get("preds", {}) or {}
    return out


def _actuals(week: int, pos: str) -> dict[str, dict[str, float]]:
    sub = (pl.read_parquet(PROCESSED_DIR / "features.parquet")
           .filter((pl.col("season") == SEASON) & (pl.col("week") == week)
                   & (pl.col("position") == pos)))
    return {_norm(r["player_display_name"]): r for r in sub.to_dicts()}


def _topk_mean(actual, pred, k: int) -> float:
    """Mean ACTUAL of the top-k by ``pred`` (k capped at the set size)."""
    if not len(actual):
        return float("nan")
    k = min(k, len(actual))
    order = np.argsort(np.asarray(pred, float))[::-1][:k]
    return float(np.mean(np.asarray(actual, float)[order]))


def main() -> None:
    weeks = _scored_weeks()
    print(f"Scored live {SEASON} weeks: {weeks}\n")

    recs = []  # dict(pos, week, ours, fp, act)  — points per (pos,week,player)
    pos_stat: dict[str, dict[str, dict[str, list[float]]]] = {}
    for pos in POS_TARGETS:
        smap = STAT_MAP[pos]
        pos_stat[pos] = {}
        for w in weeks:
            board = _load_fp_board(SEASON, w, pos)
            if not board:
                continue
            logged = _logged_preds(w, pos)
            actuals = _actuals(w, pos)
            for name, st in board.items():
                if name not in logged or name not in actuals:
                    continue
                lp, act = logged[name], actuals[name]
                valid = []  # (our_stat, our_val, fp_val, act_val)
                for fp_key, our_stat in smap.items():
                    if st.get(fp_key) is None or our_stat not in lp \
                            or act.get(our_stat) is None:
                        continue
                    pos_stat[pos].setdefault(our_stat, {})
                    pos_stat[pos][our_stat].setdefault("ours", []).append(float(lp[our_stat]))
                    pos_stat[pos][our_stat].setdefault("fp", []).append(float(st[fp_key]))
                    pos_stat[pos][our_stat].setdefault("actual", []).append(float(act[our_stat]))
                    valid.append((our_stat, lp[our_stat], st[fp_key], act[our_stat]))
                if not valid:
                    continue
                o_pts = sum(SCORE.get(s, 0.0) * float(ov) for s, ov, fv, av in valid)
                f_pts = sum(SCORE.get(s, 0.0) * float(fv) for s, ov, fv, av in valid)
                a_pts = sum(SCORE.get(s, 0.0) * float(av) for s, ov, fv, av in valid)
                recs.append({"pos": pos, "week": w, "ours": o_pts, "fp": f_pts, "act": a_pts})

    # ---- per position ----
    per_pos = {}
    for pos in POS_TARGETS:
        precs = [r for r in recs if r["pos"] == pos]
        if not precs:
            continue
        ours = [r["ours"] for r in precs]
        fp = [r["fp"] for r in precs]
        act = [r["act"] for r in precs]
        o_mae, f_mae = [], []
        for s in POS_TARGETS[pos]:
            d = pos_stat[pos].get(s)
            if not d or not d.get("ours"):
                continue
            o_mae.append(float(np.mean(np.abs(np.array(d["ours"]) - np.array(d["actual"])))))
            f_mae.append(float(np.mean(np.abs(np.array(d["fp"]) - np.array(d["actual"])))))
        raw = {"n_stat_weeks": int(sum(len(pos_stat[pos][s]["ours"]) for s in POS_TARGETS[pos]
                                       if s in pos_stat[pos])),
               "ours_mae": round(float(np.mean(o_mae)), 3) if o_mae else None,
               "fp_mae": round(float(np.mean(f_mae)), 3) if f_mae else None,
               "we_win_raw": bool(o_mae and f_mae and np.mean(o_mae) < np.mean(f_mae))}
        sp = _stats(ours, act); sf = _stats(fp, act)
        o_top, f_top = [], []
        for w in weeks:
            wk = [r for r in precs if r["week"] == w]
            if len(wk) >= 2:
                o_top.append(_topk_mean([r["act"] for r in wk], [r["ours"] for r in wk], TOP_K))
                f_top.append(_topk_mean([r["act"] for r in wk], [r["fp"] for r in wk], TOP_K))
        per_pos[pos] = {
            "n": len(precs), "raw_stat": raw,
            "points_ours": sp, "points_fp": sf,
            "we_win_mae": bool(sp["mae"] < sf["mae"]),
            "we_win_rmse": bool(sp["rmse"] < sf["rmse"]),
            "we_win_spearman": bool(sp["spearman"] > sf["spearman"]),
            "topK_ours": round(float(np.mean(o_top)), 2) if o_top else None,
            "topK_fp": round(float(np.mean(f_top)), 2) if f_top else None,
            "we_win_topK": bool(o_top and f_top and np.mean(o_top) > np.mean(f_top))}

    # ---- per week (pooled over positions) ----
    per_week = {}
    for w in weeks:
        wk = [r for r in recs if r["week"] == w]
        if len(wk) < 2:
            continue
        per_week[f"w{w:02d}"] = {
            "n": len(wk),
            "ours_mae": round(float(np.mean(np.abs(np.array([r['ours'] for r in wk]) - np.array([r['act'] for r in wk])))), 3),
            "fp_mae": round(float(np.mean(np.abs(np.array([r['fp'] for r in wk]) - np.array([r['act'] for r in wk])))), 3),
            "ours_rho": round(_spearman([r["ours"] for r in wk], [r["act"] for r in wk]), 4),
            "fp_rho": round(_spearman([r["fp"] for r in wk], [r["act"] for r in wk]), 4)}

    # ---- pooled (all positions) ----
    ours = [r["ours"] for r in recs]
    fp = [r["fp"] for r in recs]
    act = [r["act"] for r in recs]
    pooled_pts = {"ours": _stats(ours, act), "fp": _stats(fp, act)}
    all_o, all_f, all_a = [], [], []
    for pos in POS_TARGETS:
        for s in POS_TARGETS[pos]:
            d = pos_stat[pos].get(s)
            if d and d.get("ours"):
                all_o += d["ours"]; all_f += d["fp"]; all_a += d["actual"]
    pooled_raw = {"n": len(all_o),
                  "ours_mae": round(float(np.mean(np.abs(np.array(all_o) - np.array(all_a)))), 3),
                  "fp_mae": round(float(np.mean(np.abs(np.array(all_f) - np.array(all_a)))), 3)}
    npos = len(per_pos)
    wins = {"raw_stat_mae": f"{sum(1 for p in per_pos.values() if p['raw_stat']['we_win_raw'])}/{npos}",
            "points_mae": f"{sum(1 for p in per_pos.values() if p['we_win_mae'])}/{npos}",
            "points_rmse": f"{sum(1 for p in per_pos.values() if p['we_win_rmse'])}/{npos}",
            "spearman": f"{sum(1 for p in per_pos.values() if p['we_win_spearman'])}/{npos}",
            f"top{TOP_K}_decision": f"{sum(1 for p in per_pos.values() if p['we_win_topK'])}/{npos}"}

    # ---- print ----
    print(f"{'pos':<3}{'metric':<11}{'n':>5}   {'OURS':>9}{'FANTASYPROS':>12}  win")
    print("-" * 52)
    for pos in POS_TARGETS:
        if pos not in per_pos:
            continue
        p = per_pos[pos]
        raw = p["raw_stat"]
        print(f"{pos:<3}{'rawMAE':<11}{raw['n_stat_weeks']:>5}   "
              f"{raw['ours_mae']:>9.2f}{raw['fp_mae']:>12.2f}  {'YES' if raw['we_win_raw'] else 'no'}")
        print(f"{pos:<3}{'ptsMAE':<11}{p['n']:>5}   "
              f"{p['points_ours']['mae']:>9.2f}{p['points_fp']['mae']:>12.2f}  {'YES' if p['we_win_mae'] else 'no'}")
        print(f"{pos:<3}{'ptsRMSE':<11}{p['n']:>5}   "
              f"{p['points_ours']['rmse']:>9.2f}{p['points_fp']['rmse']:>12.2f}  {'YES' if p['we_win_rmse'] else 'no'}")
        print(f"{pos:<3}{'ptsRho':<11}{p['n']:>5}   "
              f"{p['points_ours']['spearman']:>9.3f}{p['points_fp']['spearman']:>12.3f}  {'YES' if p['we_win_spearman'] else 'no'}")
        print(f"{pos:<3}{'top5act':<11}{p['n']:>5}   "
              f"{p['topK_ours']:>9.2f}{p['topK_fp']:>12.2f}  {'YES' if p['we_win_topK'] else 'no'}")
        print("-" * 52)
    print("\nPer-week trend (pooled over positions, points):")
    for wk, v in per_week.items():
        print(f"  {wk}  n={v['n']:>3}  MAE ours {v['ours_mae']:.2f} vs FP {v['fp_mae']:.2f}"
              f"   Rho ours {v['ours_rho']:+.3f} vs FP {v['fp_rho']:+.3f}")
    print("\nPOOLED (all positions, scored weeks):")
    print(f"  player-weeks: {len(recs)}   raw-stat rows: {pooled_raw['n']}")
    print(f"  raw-stat MAE  ours {pooled_raw['ours_mae']:.2f} vs FP {pooled_raw['fp_mae']:.2f}"
          f"   ({wins['raw_stat_mae']} pos)")
    print(f"  points  MAE   ours {pooled_pts['ours']['mae']:.2f} vs FP {pooled_pts['fp']['mae']:.2f}"
          f"   ({wins['points_mae']} pos)")
    print(f"  points  RMSE  ours {pooled_pts['ours']['rmse']:.2f} vs FP {pooled_pts['fp']['rmse']:.2f}"
          f"   ({wins['points_rmse']} pos)")
    print(f"  points  Rho   ours {pooled_pts['ours']['spearman']:+.3f} vs FP {pooled_pts['fp']['spearman']:+.3f}"
          f"   ({wins['spearman']} pos)")
    print(f"  top{TOP_K} decision  {wins[f'top{TOP_K}_decision']} pos  ("
          + ", ".join(f"{k}: {per_pos[k]['topK_ours']} vs {per_pos[k]['topK_fp']}"
                      for k in POS_TARGETS if k in per_pos) + ")")

    out = {"season": SEASON, "weeks": weeks, "top_k": TOP_K,
           "n_player_weeks": len(recs),
           "note": ("live OOS head-to-head; OURS = logged pre-kickoff predictions "
                    "(holdout_log.jsonl), FP = free-tier top-10/position board, "
                    "actuals = features.parquet; overlap set per (week,position); "
                    "ESPN Standard (non-PPR) points"),
           "per_position": per_pos, "per_week": per_week,
           "pooled": {"raw_stat": pooled_raw, "points": pooled_pts, "win_counts": wins}}
    (MODELS_DIR / "fantasypros_2026_report.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {MODELS_DIR / 'fantasypros_2026_report.json'}")


if __name__ == "__main__":
    main()
