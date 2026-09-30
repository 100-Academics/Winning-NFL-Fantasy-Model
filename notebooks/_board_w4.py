"""2026 w4 top-10 per position, standard 1-PPR fantasy conversion.

Reads the LATEST logged entry per (season, week, player) from the append-only
holdout log, converts the raw stat components to standard 1-PPR points, and
prints top-10 per position. This is the "live board" the model should show now
that the NGS scheduled-panel bug is fixed.

Scoring (standard 1-PPR, same as src.bench/src.holdout):
  yards (rush/pass/rec)   1 pt / 10 yds
  TD (rush/pass/rec)       6 pts each
  receptions               1 pt each
  interception            -2 pts
"""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
SCORE = {"passing_yards":0.1,"rushing_yards":0.1,"receiving_yards":0.1,
         "passing_tds":6,"rushing_tds":6,"receiving_tds":6,"receptions":1,
         "passing_interceptions":-2}
POS = ["QB","RB","WR","TE"]

def ppr(preds): return sum(SCORE.get(k,0)*v for k,v in preds.items())

# latest logged entry per (season,week,player)
entries={}
for line in (REPO/"models/holdout_log.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        e=json.loads(line)
        entries[(e["season"],e["week"],e["player_id"])]=e

w4 = [e for (s,wk,pid),e in entries.items() if (s,wk)==(2026,4)]
by = defaultdict(list)
for e in w4:
    by[e["position"]].append(e)

out = ["2026 WEEK 4 — TOP 10 PER POSITION (standard 1-PPR, live model)", ""]
for pos in POS:
    lst = sorted([(ppr(e["preds"]), e["player_display_name"], e["team"]) for e in by.get(pos,[])],
                 reverse=True)
    out.append(f"===== {pos} =====")
    for i,(p,n,t) in enumerate(lst[:10],1):
        out.append(f"  {i:2d}  {p:5.1f}  {n} ({t})")
    out.append("")
print("\n".join(out))
