"""Fetch FantasyPros consensus weekly projections (free public API, top-10/pos).

Caches raw JSON under data/processed/fantasypros/ (gitignored) so re-runs are
free and we never re-burn the free-tier rate limit. The API key is read from
the gitignored .env (FANTASYPROS_API_KEY), never hard-coded.

Free-tier shape (per the OpenAPI spec + live probe):
  GET https://api.fantasypros.com/public/v2/json/nfl/{season}/projections
      ?position={QB|RB|WR|TE}&week={0..17}
  -> {season, week, count, positions, scoring, experts,
      players: [{fpid, name, position_id, team_id, stats:{...}}, ...]}

`stats` is a dict of weekly projection values. week=0 is preseason; week>=1 is
that NFL week. The free tier returns only the top-10 players per position
(`limit: 10`, `public_api_limited: true`).

Run:  uv run python -m src.fetch_fantasypros
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "processed" / "fantasypros"
API = "https://api.fantasypros.com/public/v2/json/nfl/{season}/projections"
SEASONS = [2024, 2025]
WEEKS = list(range(1, 18))
POSITIONS = ["QB", "RB", "WR", "TE"]


def _load_api_key() -> str:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        raise SystemExit("no .env at repo root; set FANTASYPROS_API_KEY")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("FANTASYPROS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("FANTASYPROS_API_KEY not found in .env")


def _cache_path(season: int, week: int, pos: str) -> pathlib.Path:
    return CACHE_DIR / f"{season}_w{week:02d}_{pos}.json"


def _fetch(session: requests.Session, key: str, season: int, week: int,
           pos: str) -> dict:
    url = API.format(season=season)
    last_err = None
    for attempt in range(10):
        try:
            r = session.get(url, params={"position": pos, "week": week}, timeout=40)
            if r.status_code == 429:  # rate limited
                wait = min(5 * (attempt + 1), 30)
                print(f"  429 on {season} w{week} {pos}; sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed {season} w{week} {pos}: {last_err!r}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _load_api_key()
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "x-api-key": key})

    total = len(SEASONS) * len(WEEKS) * len(POSITIONS)
    done = 0
    for season in SEASONS:
        for week in WEEKS:
            for pos in POSITIONS:
                path = _cache_path(season, week, pos)
                if path.exists() and path.stat().st_size > 0:
                    done += 1
                    continue
                data = _fetch(session, key, season, week, pos)
                path.write_text(json.dumps(data))
                n = len(data.get("players", []))
                done += 1
                print(f"[{done}/{total}] {season} w{week:02d} {pos}: {n} players")
                time.sleep(0.3)
    print(f"\nDone. {done}/{total} cached under {CACHE_DIR}")


if __name__ == "__main__":
    main()
