"""Emit 2026 w4 top-10-per-position board (ESPN standard) as JSON for the graph."""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
ESPN = {"passing_yards":0.04,"rushing_yards":0.1,"receiving_yards":0.1,
        "passing_tds":4,"rushing_tds":6,"receiving_tds":6,"passing_interceptions":-2}
POS = ["QB","RB","WR","TE"]
def pts(p): return round(sum(ESPN.get(k,0)*v for k,v in p.items()),2)

entries={}
for line in (REPO/"models/holdout_log.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        e=json.loads(line); entries[(e["season"],e["week"],e["player_id"])]=e
w4=[e for (s,wk,pid),e in entries.items() if (s,wk)==(2026,4)]
by=defaultdict(list)
for e in w4: by[e["position"]].append(e)
out={"title":"2026 Week 4 — Top 10 by Position","scoring":"ESPN Standard (non-PPR)","positions":{}}
for pos in POS:
    lst=sorted([(pts(e["preds"]), e["player_display_name"], e["team"]) for e in by.get(pos,[])],reverse=True)[:10]
    out["positions"][pos]=[{"rank":i+1,"name":n,"team":t,"pts":p} for i,(p,n,t) in enumerate(lst)]
print(json.dumps(out,indent=2))
