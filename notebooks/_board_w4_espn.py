"""2026 w4 top-10 per position in ESPN STANDARD scoring (non-PPR).

ESPN standard (default, non-PPR):
  passing    0.04 pt / yd,  4 pt / TD,  -2 pt / INT
  rushing    0.10 pt / yd,  6 pt / TD
  receiving  0.10 pt / yd,  6 pt / TD
  (NO point per reception)

Reads the LATEST logged entry per (season, week, player) from the holdout log
(appended by src.holdout --log after the QB-rushing retrain + NGS fix),
converts raw stat components to ESPN-standard points, prints top-10/position.
"""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
ESPN = {"passing_yards":0.04,"rushing_yards":0.1,"receiving_yards":0.1,
        "passing_tds":4,"rushing_tds":6,"receiving_tds":6,
        "passing_interceptions":-2}
POS = ["QB","RB","WR","TE"]

def pts(preds): return sum(ESPN.get(k,0)*v for k,v in preds.items())

entries={}
for line in (REPO/"models/holdout_log.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        e=json.loads(line); entries[(e["season"],e["week"],e["player_id"])]=e

w4 = [e for (s,wk,pid),e in entries.items() if (s,wk)==(2026,4)]
by = defaultdict(list)
for e in w4: by[e["position"]].append(e)

out=["2026 WEEK 4 — TOP 10 PER POSITION (ESPN STANDARD, non-PPR, live model)  0.04/pyd 4/pTD",""]
for pos in POS:
    lst=sorted([(pts(e["preds"]), e["player_display_name"], e["team"]) for e in by.get(pos,[])],reverse=True)
    out.append(f"===== {pos} =====")
    for i,(p,n,t) in enumerate(lst[:10],1):
        out.append(f"  {i:2d}  {p:5.1f}  {n} ({t})")
    out.append("")
print("\n".join(out))
