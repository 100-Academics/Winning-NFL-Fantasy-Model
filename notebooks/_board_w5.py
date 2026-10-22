"""2026 w5 board data (ESPN STANDARD non-PPR) from the LATEST logged entry.

Offense: top-10 per position (start/sit board), converted from the raw
stat components in models/holdout_log.jsonl to ESPN-standard points.
  passing    0.04 / yd,  4 / TD,  -2 / INT
  rushing    0.10 / yd,  6 / TD
  receiving  0.10 / yd,  6 / TD        (NO point per reception)

Defense: the full D/ST board (all teams) from models/holdout_log_def.jsonl,
  board order = rank by opponent's Vegas implied total, lowest first.

Writes models/w5_board.json for the board widget.
"""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
ESPN = {"passing_yards": 0.04, "rushing_yards": 0.1, "receiving_yards": 0.1,
        "passing_tds": 4, "rushing_tds": 6, "receiving_tds": 6,
        "passing_interceptions": -2}
POS = ["QB", "RB", "WR", "TE"]

def pts(preds): return round(sum(ESPN.get(k, 0) * v for k, v in preds.items()), 2)

# ---- offense (latest entry per (season, week, player_id)) ----
entries = {}
for line in (REPO / "models/holdout_log.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        e = json.loads(line)
        entries[(e["season"], e["week"], e["player_id"])] = e
w5 = [e for (s, wk, pid), e in entries.items() if (s, wk) == (2026, 5)]
by = defaultdict(list)
for e in w5:
    by[e["position"]].append(e)

positions = {}
for pos in POS:
    rows = sorted([(pts(e["preds"]), e["player_display_name"], e["team"])
                   for e in by.get(pos, [])], reverse=True)
    out = []
    for i, (p, n, t) in enumerate(rows[:10], 1):
        out.append({"rank": i, "name": n, "team": t, "pts": p})
    positions[pos] = out

# ---- defense (one log entry PER TEAM; sort by board_rank) ----
def_rows = []
for line in (REPO / "models/holdout_log_def.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        e = json.loads(line)
        if e["season"] == 2026 and e["week"] == 5:
            def_rows.append(e)
def_rows.sort(key=lambda r: r["board_rank"])
defense = []
for i, r in enumerate(def_rows, 1):
    defense.append({"rank": i, "team": r["team"], "opp": r["opp"],
                    "proj_allwd": r["proj_allwd"], "pred_d_st": r["d_st"]})

out = {"season": 2026, "week": 5, "scoring": "ESPN Standard (non-PPR)",
       "limit": "top-10/pos offense + full D/ST board",
       "n_offense_logged": len(w5), "n_defense_logged": len(defense),
       "positions": positions, "defense": defense}
(REPO / "models/w5_board.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

# console summary
print(f"OFFENSE w5: {len(w5)} players logged | DEFENSE w5: {len(defense)} teams logged")
for pos in POS:
    print(f"  {pos}: " + ", ".join(f"{r['name']}({r['team']}) {r['pts']}" for r in positions[pos][:5]))
print("  D/ST top-5:", ", ".join(f"{d['team']}(vs {d['opp']}) {d['pred_d_st']}" for d in defense[:5]))
print("\nwrote models/w5_board.json")
