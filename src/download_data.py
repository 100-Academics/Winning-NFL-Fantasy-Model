"""Reproducible historical data acquisition for the NFL fantasy model.

Pulls weekly player/team stats and supporting context tables from the
nflverse-data releases (via the ``nflreadpy`` package) and mirrors them as
parquet files under ``data/raw/``. Raw pulls are treated as immutable — all
downstream transforms go through later scripts into ``data/processed/``.

The download is *resumable*: a dataset whose parquet already exists (and is
non-empty) is skipped unless ``--force`` is passed. So if a run is interrupted,
just re-run the same command and it picks up where it left off.

Usage
-----
# Default: core stats + context tables, seasons 2016..2026
    uv run python -m src.download_data

# Explicit seasons / a subset of datasets
    uv run python -m src.download_data --seasons 2021 2022 2023
    uv run python -m src.download_data --datasets core,context
    uv run python -m src.download_data --datasets all      # includes advanced (NGS, etc.)
    uv run python -m src.download_data --force             # re-download everything

Dataset groups
--------------
core       player weekly stats, team weekly stats, schedules, players (static)
context    weekly rosters, snap counts, injuries
advanced   Next Gen Stats (passing/receiving/rushing), participation
all        everything above
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import nflreadpy as nfl
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"

# Default season window (2016-2026 inclusive).
DEFAULT_SEASONS = list(range(2016, 2027))


@dataclass
class Dataset:
    """A named table to download, mapped to a nflreadpy loader."""

    name: str
    filename: str
    group: str
    loader: str  # attribute name on the ``nfl`` module
    kwargs: dict


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
DATASETS: list[Dataset] = [
    # --- core -------------------------------------------------------------- #
    Dataset("player_weekly", "player_weekly_stats.parquet", "core",
            "load_player_stats", {"summary_level": "week"}),
    Dataset("team_weekly", "team_weekly_stats.parquet", "core",
            "load_team_stats", {"summary_level": "week"}),
    Dataset("schedules", "schedules.parquet", "core",
            "load_schedules", {}),
    Dataset("players", "players.parquet", "core",
            "load_players", {}),
    # --- context ----------------------------------------------------------- #
    Dataset("weekly_rosters", "weekly_rosters.parquet", "context",
            "load_rosters_weekly", {}),
    Dataset("snap_counts", "snap_counts.parquet", "context",
            "load_snap_counts", {}),
    Dataset("injuries", "injuries.parquet", "context",
            "load_injuries", {}),
    # --- advanced ---------------------------------------------------------- #
    Dataset("nextgen_passing", "nextgen_passing.parquet", "advanced",
            "load_nextgen_stats", {"stat_type": "passing"}),
    Dataset("nextgen_receiving", "nextgen_receiving.parquet", "advanced",
            "load_nextgen_stats", {"stat_type": "receiving"}),
    Dataset("nextgen_rushing", "nextgen_rushing.parquet", "advanced",
            "load_nextgen_stats", {"stat_type": "rushing"}),
    Dataset("participation", "participation.parquet", "advanced",
            "load_participation", {}),
]

GROUPS = {
    "core": {"player_weekly", "team_weekly", "schedules", "players"},
    "context": {"weekly_rosters", "snap_counts", "injuries"},
    "advanced": {"nextgen_passing", "nextgen_receiving",
                 "nextgen_rushing", "participation"},
    "all": {d.name for d in DATASETS},
}

# Datasets that take a `seasons` argument. `load_players` is static (no seasons).
_TAKES_SEASONS = {d.name for d in DATASETS if d.name != "players"}


def _select_datasets(names: list[str]) -> list[Dataset]:
    """Resolve group names (core/context/advanced/all) and explicit dataset
    names into an ordered list of Dataset objects, deduplicated."""
    wanted: set[str] = set()
    for n in names:
        n = n.strip().lower()
        if n in GROUPS:
            wanted |= GROUPS[n]
        elif n in {d.name for d in DATASETS}:
            wanted.add(n)
        else:
            raise SystemExit(
                f"Unknown dataset/group {n!r}. "
                f"Valid groups: {sorted(GROUPS)}; "
                f"datasets: {sorted(d.name for d in DATASETS)}"
            )
    out = [d for d in DATASETS if d.name in wanted]
    if not out:
        raise SystemExit("No datasets selected.")
    return out


def _call_loader(ds: Dataset, seasons: list[int]) -> pl.DataFrame:
    fn = getattr(nfl, ds.loader)
    kw = dict(ds.kwargs)
    if ds.name in _TAKES_SEASONS:
        kw["seasons"] = seasons
    return fn(**kw)


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)  # atomic-ish: never leave a half-written .parquet


def _is_done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def download(ds: Dataset, seasons: list[int], raw_dir: Path, force: bool) -> bool:
    """Download one dataset. Returns True if it was (re)downloaded, False if skipped."""
    path = raw_dir / ds.filename
    if _is_done(path) and not force:
        print(f"  [skip] {ds.filename:<28} already present ({path.stat().st_size:,} bytes)")
        return False
    print(f"  [get ] {ds.filename:<28} seasons={seasons if ds.name in _TAKES_SEASONS else 'static'}",
          flush=True)
    t0 = time.time()
    df = _call_loader(ds, seasons)
    _write_parquet(df, path)
    print(f"        {df.shape[0]:>9,} rows x {df.shape[1]:>3} cols  "
          f"({time.time() - t0:6.1f}s, {path.stat().st_size / 1e6:7.2f} MB)", flush=True)
    return True


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS,
                   help=f"Season years to pull (default: {DEFAULT_SEASONS})")
    p.add_argument("--datasets", type=str, default="core,context",
                   help="Comma-separated dataset names or groups "
                        "(core, context, advanced, all). Default: core,context")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help=f"Output directory (default: {DEFAULT_RAW_DIR})")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the parquet already exists")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seasons = sorted(set(args.seasons))
    datasets = _select_datasets(args.datasets.split(","))
    raw_dir = args.raw_dir
    try:
        from importlib.metadata import version as _dist_version
        ver = _dist_version("nflreadpy")
    except Exception:
        ver = ""
    print(f"nflverse data acquisition via nflreadpy {ver}".rstrip())
    print(f"  seasons   : {seasons}")
    print(f"  datasets  : {[d.name for d in datasets]}")
    print(f"  raw dir   : {raw_dir}\n")

    done, skipped, failed = [], [], []
    for ds in datasets:
        try:
            if download(ds, seasons, raw_dir, args.force):
                done.append(ds.filename)
            else:
                skipped.append(ds.filename)
        except Exception as exc:  # keep going; report at the end
            failed.append((ds.filename, repr(exc)))
            print(f"  [FAIL] {ds.filename}: {exc!r}", flush=True)

    print("\n" + "=" * 64)
    print(f"Downloaded : {len(done)}  {done}")
    print(f"Skipped    : {len(skipped)}  {skipped}")
    print(f"Failed     : {len(failed)}")
    for name, err in failed:
        print(f"    - {name}: {err}")
    print("=" * 64)
    if failed:
        print("Re-run the same command to retry the failed datasets "
              "(already-downloaded ones will be skipped).")
        return 1
    print("Phase 2 data acquisition complete. Next: Phase 3 cleaning + features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
