"""W4 head-to-head: OUR model vs FANTASYPROS consensus, scored on W4 ACTUALS.

Overlap set = (2026, w4, position, player) present in BOTH our W4 log and the
FP W4 top-10/pos board. For each, ESPN-standard (non-PPR) points for:
  our logged pred  |  FP consensus  |  actual
then Spearman + MAE vs actual for each side, overall and per position.
This is the real 'did we beat the consensus on the held-out week' answer.
"""
import json
import pathlib

import polars as pl
import numpy as np
from scipy.stats import rankdata

REPO = pathlib.Path(__file__).resolve().parent.parent
ESPN = {"passing_yards": 0.04, "rushing_yards": 0.10, "receiving_yards": 0.10,
        "passing_tds": 4, "rushing_tds": 6, "receiving_tds": 6,
        "passing_interceptions": -2}
FPSTAT = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards",
          "rec_yds": "receiving_yards", "pass_tds": "passing_tds",
          "rush_tds": "rushing_tds", "rec_tds": "receiving_tds",
          "pass_ints": "passing_interceptions", "rec_rec": "receptions"}
POS = ["QB", "RB", "WR", "TE"]


def spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def stats(pred, act):
    e = np.abs(np.asarray(pred) - np.asarray(act))
    return {"n": len(act), "mae": round(float(e.mean()), 3),
            "spearman": round(spearman(pred, act), 4)}


def norm(s):
    for suf in [" II", " III", " IV", " Jr.", " Sr."]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.split()).lower()


# ---- our W4 log (latest per player) ----
entries = {}
for line in (REPO / "models/holdout_log.jsonl").read_text().splitlines():
    if line.strip():
        e = json.loads(line)
        if (e["season"], e["week"]) == (2026, 4):
            entries[(e["position"], norm(e["player_display_name"]))] = e

# ---- FP W4 boards ----
fp = {}
for pos in POS:
    p = REPO / "data/processed/fantasypros" / f"2026_w04_{pos}.json"
    if not p.exists():
        continue
    for plr in json.loads(p.read_text()).get("players", []):
        fp[(pos, norm(plr["name"]))] = plr.get("stats", {})

# ---- actuals ----
feat = pl.read_parquet(REPO / "data/processed/features.parquet")
actual = {
    (r["position"], norm(r["player_display_name"])): r
    for r in feat.filter((pl.col("season") == 2026) & (pl.col("week") == 4)).to_dicts()
}


def espn_from_raw(raw: dict) -> float:
    return round(sum(ESPN.get(k, 0) * (v or 0) for k, v in raw.items()), 3)


rows = []
for key in set(entries) & set(fp):
    pos, name = key
    if key not in actual:
        continue
    a = actual[key]
    our = espn_from_raw(entries[key]["preds"])
    fpraw = {FPSTAT[k]: v for k, v in fp[key].items() if k in FPSTAT}
    fpp = espn_from_raw(fpraw)
    actp = espn_from_raw({k: a.get(k, 0) for k in ESPN if k in a})
    rows.append((pos, name, our, fpp, actp))

print(f"W4 head-to-head on {len(rows)} overlap players (ESPN standard, non-PPR)\n")
print(f"{'POS':<3}{'PLAYER':<22}{'OUR':>8}{'FP':>8}{'ACT':>8}")
for pos, name, o, f, a in sorted(rows, key=lambda r: (r[0], -r[4])):
    print(f"{pos:<3}{name:<22}{o:>8.2f}{f:>8.2f}{a:>8.2f}")

# aggregate + per-pos
print("\n=== SCORED vs actual (overlap set) ===")
print(f"{'SET':<4}{'n':>5}{'OUR mae':>9}{'OUR rho':>9}{'FP mae':>9}{'FP rho':>9}")
all_o, all_f, all_a = [], [], []
for pos in POS + ["ALL"]:
    sel = [r for r in rows if pos == "ALL" or r[0] == pos]
    if not sel:
        continue
    o = [r[2] for r in sel]; f = [r[3] for r in sel]; a = [r[4] for r in sel]
    so = stats(o, a); sf = stats(f, a)
    print(f"{pos:<4}{len(a):>5}{so['mae']:>9.2f}{so['spearman']:>+9.3f}"
          f"{sf['mae']:>9.2f}{sf['spearman']:>+9.3f}")
    if pos == "ALL":
        all_o, all_f, all_a = o, f, a

# who won the head-to-head on individual players
wins = sum(1 for r in rows if r[2] > r[3])
losses = sum(1 for r in rows if r[3] > r[2])
print(f"\nPlayer-by-player (closer to actual): ours {wins}, FP {losses}, "
      f"ties {len(rows)-wins-losses}")
out = {"season": 2026, "week": 4, "n_overlap": len(rows),
       "rows": [{"pos": p, "player": n, "our": o, "fp": f, "actual": a}
                for p, n, o, f, a in rows]}
(REPO / "models/w4_head2head_fp.json").write_text(json.dumps(out, indent=1))
print("\nwrote models/w4_head2head_fp.json")
