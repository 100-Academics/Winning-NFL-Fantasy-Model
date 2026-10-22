"""Fetch 2026 W5 FantasyPros consensus top-10/pos (free tier) + compare to OUR W5 board.

Reuses the same API / auth / cache as src.fetch_fantasypros, but for 2026 w5.
Writes the board (ESPN-standard points) to models/w5_fpboard.json and prints a
head-to-head overlap vs our model's W5 log (both non-PPR, same top-10/pos set).
"""
import json
import pathlib
import time

import requests

REPO = pathlib.Path(__file__).resolve().parent.parent
CACHE = REPO / "data/processed/fantasypros"
API = "https://api.fantasypros.com/public/v2/json/nfl/{season}/projections"
POS = ["QB", "RB", "WR", "TE"]
SEASON, WEEK = 2026, 5

ESPN = {"pass_yds": 0.04, "rush_yds": 0.10, "rec_yds": 0.10,
        "pass_tds": 4, "rush_tds": 6, "rec_tds": 6, "pass_ints": -2}


def _key():
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("FANTASYPROS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no FANTASYPROS_API_KEY in .env")


def _fetch(session, key, pos):
    url = API.format(season=SEASON)
    for attempt in range(12):
        r = session.get(url, params={"position": pos, "week": WEEK}, timeout=40)
        if r.status_code == 429:
            w = min(5 * (attempt + 1), 30)
            print(f"  429 {pos}; sleep {w}s")
            time.sleep(w)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"fetch failed {pos}")


def fp_pts(st):
    return round(sum(ESPN.get(k, 0) * (v or 0) for k, v in st.items()), 2)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    key = _key()
    s = requests.Session()
    s.headers.update({"Accept": "application/json", "x-api-key": key})

    fp = {}
    for pos in POS:
        path = CACHE / f"{SEASON}_w{WEEK:02d}_{pos}.json"
        if path.exists() and path.stat().st_size > 0:
            data = json.loads(path.read_text())
            print(f"[cached] {pos}: {len(data.get('players', []))} players")
        else:
            data = _fetch(s, key, pos)
            path.write_text(json.dumps(data))
            print(f"[fetched] {pos}: {len(data.get('players', []))} players")
        rows = []
        for p in data.get("players", [])[:10]:
            rows.append({"name": p["name"], "team": p["team_id"],
                         "pts": fp_pts(p.get("stats", {}))})
        rows.sort(key=lambda r: r["pts"], reverse=True)
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        fp[pos] = rows
    out = {"season": SEASON, "week": WEEK,
           "scoring": "FantasyPros consensus (STD, non-PPR)",
           "limit": "top-10/pos (free tier)", "positions": fp}
    (REPO / "models/w5_fpboard.json").write_text(json.dumps(out, indent=1))
    print("\nwrote models/w5_fpboard.json")

    # ---- head-to-head vs OUR W5 log (latest entry per player) ----
    entries = {}
    for line in (REPO / "models/holdout_log.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            if (e["season"], e["week"]) == (SEASON, WEEK):
                entries[(e["position"], e["player_display_name"].lower())] = e

    def our_pts(preds):
        O = {"passing_yards": 0.04, "rushing_yards": 0.10, "receiving_yards": 0.10,
             "passing_tds": 4, "rushing_tds": 6, "receiving_tds": 6,
             "passing_interceptions": -2}
        return round(sum(O.get(k, 0) * v for k, v in preds.items()), 2)

    print("\n=== W5 head-to-head: OURS vs FANTASYPROS (same top-10/pos, ESPN std) ===")
    print(f"{'POS':<3}{'PLAYER':<22}{'OUR':>7}{'FP':>7}  (who's higher)")
    for pos in POS:
        ours_by = {e["player_display_name"].lower(): e for (p, n), e in entries.items() if p == pos}
        for r in fp[pos][:10]:
            k = r["name"].lower()
            ours = our_pts(ours_by[k]["preds"]) if k in ours_by else None
            mark = ""
            if ours is not None:
                mark = "ours" if ours >= r["pts"] else "FP"
                print(f"{pos:<3}{r['name']:<22}{ours:>7.2f}{r['pts']:>7.2f}  {mark}")
            else:
                print(f"{pos:<3}{r['name']:<22}{'—':>7}{r['pts']:>7.2f}  (not on our board)")


if __name__ == "__main__":
    main()
