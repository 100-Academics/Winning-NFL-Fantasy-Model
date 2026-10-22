"""Emit 2026 w4 FantasyPros consensus top-10/pos (STD points) as JSON + board.

FP ships pre-computed consensus `points` (scoring=STD, non-PPR), so we use it
directly — apples-to-apples with our ESPN-standard board (both non-PPR).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data/processed/fantasypros"
POS = ["QB", "RB", "WR", "TE"]
out = {"season": 2026, "week": 4, "scoring": "FantasyPros consensus (STD, non-PPR)",
       "limit": "top-10/pos (free tier)", "positions": {}}

for pos in POS:
    d = json.loads((CACHE / f"2026_w04_{pos}.json").read_text(encoding="utf-8"))
    rows = []
    for p in d["players"][:10]:
        rows.append({"name": p["name"], "team": p["team_id"],
                     "pts": round(float(p["stats"].get("points") or 0.0), 2)})
    rows.sort(key=lambda r: r["pts"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    out["positions"][pos] = rows

print(json.dumps(out, indent=1))
(CACHE / "2026_w04_board.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\nwrote", CACHE / "2026_w04_board.json", file=__import__("sys").stderr)
