"""Tier-1-lite: within-slice Spearman vs full-roster Spearman (padding check).

For every logged 2026 week, compute the full-roster rank Spearman (what
holdout --score reports) and the within-slice Spearman of the projected
starter pools (Top-12 QB/TE, Top-24 RB/WR). If the full-roster number is
heavily padded by bench rows, the top-slice number should be materially
lower. Read-only; no model changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
feat = pl.read_parquet(REPO / "data/processed/features.parquet")
from src.scoring import ESPN_STANDARD, fantasy_points

lines_all = [json.loads(l) for l in open(REPO / "models/holdout_log.jsonl")]
entries: dict[tuple[int, int], list[dict]] = {}
for e in lines_all:
    key = (e["season"], e["week"])
    entries.setdefault(key, []).append(e)


def spear(p, a) -> float:
    if np.std(p) == 0 or np.std(a) == 0:
        return float("nan")
    return float(spearmanr(p, a).statistic)


SLICES = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
lines = []
for (season, week) in sorted(entries):
    if season != 2026:
        continue
    recs = [r for r in entries[(season, week)]
            if r["position"] in SLICES]
    # dedupe players logged more than once: keep the latest log
    best: dict[str, dict] = {}
    for r in recs:
        k = r["player_id"]
        if k not in best or r["logged_at"] > best[k]["logged_at"]:
            best[k] = r
    recs = list(best.values())
    if not recs:
        continue
    act_rows = (feat.filter((pl.col("season") == season) & (pl.col("week") == week)
                            & pl.col("position").is_in(list(SLICES)))
                .select(["player_display_name"] + list(ESPN_STANDARD)))
    if act_rows.height == 0:  # unplayed week (e.g. W11 logged pre-kickoff)
        continue
    df = (pl.DataFrame(
             [{"player_display_name": r["player_display_name"],
               "position": r["position"],
               "pred": float(fantasy_points(r["preds"]))} for r in recs],
             orient="row")
          .join(
              pl.DataFrame(
                  [{"player_display_name": a["player_display_name"],
                    "actual": float(fantasy_points({k: a[k] for k in ESPN_STANDARD}))}
                   for a in act_rows.iter_rows(named=True)],
                  orient="row"),
              on="player_display_name", how="inner"))
    if df.height < 50:
        continue
    full = spear(df["pred"].to_numpy(), df["actual"].to_numpy())
    parts = [f"W{week:2d}  n={df.height:3d}  full={full:+.3f}"]
    for pos, k in SLICES.items():
        sub = df.filter(pl.col("position") == pos).sort("pred", descending=True).head(k)
        if sub.height >= 8:
            parts.append(f"{pos}top{k:2d}={spear(sub['pred'].to_numpy(), sub['actual'].to_numpy()):+.3f}")
    lines.append("  ".join(parts))

print("\n".join(lines))
