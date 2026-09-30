"""CollegeFootballData (CFBD) API client — Phase 5.5 rookie prior.

Wraps the `cfbd` OpenAPI client with:
  * auth from `.env` (CFBD_API_KEY) — the key is NEVER committed,
  * a disk cache under `data/raw/cfbd/` so repeated runs don't re-download,
  * polars-DataFrame returns (the raw client returns lists of model objects).

cfbd 4.5.2 auth gotcha (cost us time): the key must be set under the
identifier "Authorization" with a "Bearer" prefix —
    cfg.api_key["Authorization"] = key
    cfg.api_key_prefix["Authorization"] = "Bearer"
Setting cfg.api_key["api_key"] does NOT work (that identifier isn't used).

Endpoints used:
  DraftApi.get_draft_picks(year)          -> list[DraftPick]
  PlayersApi.get_player_season_stats(year)-> list[PlayerSeasonStat] (long format)
  PlayersApi.get_player_usage(year)       -> list[PlayerUsage]

Join keys:
  DraftPick.nfl_athlete_id  is NOT in nflverse's ID space.
  DraftPick.college_athlete_id == PlayerSeasonStat.player_id == PlayerUsage.id.
  Bridge to the main model is by player NAME (verified: first-round 2024 picks
  match nflverse display_name with identical draft_round/draft_pick).
"""
from __future__ import annotations

import os
import pathlib
import time

import polars as pl

import cfbd

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "cfbd"


def _load_key() -> str:
    """Read CFBD_API_KEY from .env (or the environment). Never committed."""
    env = os.environ.get("CFBD_API_KEY")
    if env:
        return env
    env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("CFBD_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit(
        "CFBD_API_KEY not found. Add it to .env (see .env.example) — "
        "free key at https://cfbd.com/developers/"
    )


def _client() -> tuple:
    key = _load_key()
    cfg = cfbd.Configuration()
    cfg.api_key["Authorization"] = key
    cfg.api_key_prefix["Authorization"] = "Bearer"
    ac = cfbd.ApiClient(cfg)
    return ac


def _to_df(items) -> pl.DataFrame:
    """List of cfbd model objects -> polars DataFrame (flatten nested dicts)."""
    if items is None or len(items) == 0:
        return pl.DataFrame()
    recs = []
    for it in items:
        if hasattr(it, "to_dict"):
            d = it.to_dict()
        elif hasattr(it, "__dict__"):
            d = dict(it)
        else:
            d = dict(it)
        # flatten one level (e.g. usage dict, hometown_info)
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}_{k2}"] = v2
            else:
                flat[k] = v
        recs.append(flat)
    return pl.DataFrame(recs)


def _cached(name: str, force: bool = False) -> pl.DataFrame | None:
    path = RAW_DIR / name
    if (not force) and path.exists():
        return pl.read_parquet(path)
    return None


def _cache(name: str, df: pl.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RAW_DIR / name)


def get_draft_picks(year: int, force: bool = False) -> pl.DataFrame:
    """CFBD draft picks for a draft year (carries NFL + college IDs + pre-draft grade)."""
    cached = _cached(f"draft_picks_{year}.parquet", force)
    if cached is not None:
        return cached
    ac = _client()
    t0 = time.time()
    items = cfbd.DraftApi(ac).get_draft_picks(year=year)
    df = _to_df(items)
    _cache(f"draft_picks_{year}.parquet", df)
    print(f"[cfbd] draft_picks {year}: {df.height} rows ({time.time()-t0:.1f}s)")
    return df


def get_player_season_stats(year: int, force: bool = False) -> pl.DataFrame:
    """College player season stats for a season (long format: category/stat_type/stat)."""
    cached = _cached(f"player_season_stats_{year}.parquet", force)
    if cached is not None:
        return cached
    ac = _client()
    t0 = time.time()
    items = cfbd.PlayersApi(ac).get_player_season_stats(year=year)
    df = _to_df(items)
    _cache(f"player_season_stats_{year}.parquet", df)
    print(f"[cfbd] player_season_stats {year}: {df.height} rows ({time.time()-t0:.1f}s)")
    return df


def get_player_usage(year: int, force: bool = False) -> pl.DataFrame:
    """College player usage (target share / dominator-style rates) for a season."""
    cached = _cached(f"player_usage_{year}.parquet", force)
    if cached is not None:
        return cached
    ac = _client()
    t0 = time.time()
    items = cfbd.PlayersApi(ac).get_player_usage(year=year)
    df = _to_df(items)
    _cache(f"player_usage_{year}.parquet", df)
    print(f"[cfbd] player_usage {year}: {df.height} rows ({time.time()-t0:.1f}s)")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pull + cache CFBD data (draft picks / college stats).")
    ap.add_argument("--draft-years", type=int, nargs="+", default=list(range(2016, 2026)))
    ap.add_argument("--stat-years", type=int, nargs="+", default=list(range(2014, 2025)))
    ap.add_argument("--usage-years", type=int, nargs="+", default=list(range(2014, 2025)))
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()
    for y in args.draft_years:
        get_draft_picks(y, args.force)
    for y in args.stat_years:
        get_player_season_stats(y, args.force)
    for y in args.usage_years:
        get_player_usage(y, args.force)
    print("\nCFBD pull complete.")
