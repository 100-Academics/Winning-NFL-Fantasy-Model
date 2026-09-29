"""Phase 5.5 — rookie prior feature table.

Builds a per-player table of *pre-season* inputs that a rookie's first NFL
season depends on. Everything is known BEFORE the rookie's first NFL game
(no lookahead):

  * draft capital : draft_round, draft_pick, UDFA flag, pre_draft_grade (CFBD)
  * age           : age_at_draft (birth_date -> draft year)
  * athleticism   : combine 40 / vertical / broad_jump / cone / shuttle / bench
                    (nflverse load_combine, joined by name + draft year)
  * college       : the player's most recent college season's rate stats
                    (YPA/YPC/YPR/PCT) + volume (yards/TDs/att), from CFBD
                    player_season_stats, reached via the DRAFT bridge
                    (CFBD DraftPick.college_athlete_id -> stats.player_id).

Bridge to CFBD is the DRAFT (year + player name): CFBD's DraftPick carries the
college_athlete_id that indexes its season stats. nflverse's players table gives
the NFL side (draft_round/pick, birth_date, name). We match on normalized name
(verified: first-round 2024 picks match exactly with identical round+pick).

Run:
    uv run python -m src.rookie_features
        -> data/processed/rookie_features.parquet
"""
from __future__ import annotations

import pathlib
import time

import polars as pl

from src import cfbd_client

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

COLLEGE_YEARS = list(range(2014, 2025))   # a 2016 rookie's senior yr was 2015
DRAFT_YEARS = list(range(2016, 2026))
COMBINE_COLS = ["forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]

# (category, stat_type) -> feature name, for the college stats we keep.
COLLEGE_STATS = [
    ("passing", "YDS", "col_pass_yds"), ("passing", "TD", "col_pass_td"),
    ("passing", "YPA", "col_pass_ypa"), ("passing", "PCT", "col_pass_pct"),
    ("passing", "ATT", "col_pass_att"), ("passing", "INT", "col_pass_int"),
    ("rushing", "YDS", "col_rush_yds"), ("rushing", "TD", "col_rush_td"),
    ("rushing", "YPC", "col_rush_ypc"), ("rushing", "CAR", "col_rush_car"),
    ("receiving", "YDS", "col_recv_yds"), ("receiving", "TD", "col_recv_td"),
    ("receiving", "YPR", "col_recv_ypr"), ("receiving", "REC", "col_recv_rec"),
]


def _norm(s) -> str:
    """Normalize a name for matching: lower, drop suffixes, one space."""
    if s is None:
        return ""
    s = str(s).lower()
    for suf in [" jr.", " sr.", " ii", " iii", " iv", " i.", " ii.", " iii."]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.replace(".", " ").replace("-", " ").split())


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _nfl_players() -> pl.DataFrame:
    # players.parquet's ID column is gsis_id (== the panel's player_id)
    return pl.read_parquet(RAW_DIR / "players.parquet").select(
        [pl.col("gsis_id").alias("player_id"), "display_name", "position", "birth_date",
         "draft_year", "draft_round", "draft_pick", "draft_team"]
    ).with_columns(pl.col("display_name").map_elements(_norm).alias("name_key"))


def _combine() -> pl.DataFrame:
    import nflreadpy as nfl
    rows = []
    for y in DRAFT_YEARS:
        try:
            items = nfl.load_combine(y)
        except Exception:
            continue
        df = items if isinstance(items, pl.DataFrame) else pl.DataFrame(items)
        keep = [c for c in ["player_name", *COMBINE_COLS] if c in df.columns]
        if "player_name" not in df.columns:
            continue
        df = df.select(keep).with_columns(
            pl.col("player_name").map_elements(_norm).alias("name_key"),
            pl.lit(y).alias("draft_year"))
        rows.append(df)
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows, how="vertical")


def _cfbd_draft() -> pl.DataFrame:
    rows = [cfbd_client.get_draft_picks(y) for y in DRAFT_YEARS]
    rows = [r for r in rows if r.height]
    if not rows:
        raise SystemExit("No CFBD draft data cached — run `python -m src.cfbd_client` first.")
    df = pl.concat(rows, how="vertical")
    return df.with_columns(pl.col("name").map_elements(_norm).alias("name_key"))


def _college_stat_lookup() -> pl.DataFrame:
    """One row per college player_id with their most-recent-season key stats (wide)."""
    frames: dict[str, list] = {name: [] for _, _, name in COLLEGE_STATS}
    for y in COLLEGE_YEARS:
        s = cfbd_client.get_player_season_stats(y)
        if not s.height:
            continue
        for cat, st, name in COLLEGE_STATS:
            sub = s.filter((pl.col("category") == cat) & (pl.col("stat_type") == st))
            if not sub.height:
                continue
            frames[name].append(sub.select(
                [pl.col("player_id"), pl.col("stat").alias("v")]
            ).with_columns(pl.lit(y).alias("season")))
    parts = []
    for _, _, name in COLLEGE_STATS:
        fr = frames[name]
        if not fr:
            parts.append(pl.DataFrame(schema={"player_id": pl.Int64, name: pl.Float64}))
            continue
        d = pl.concat(fr, how="vertical")
        d = d.sort(["player_id", "season"]).unique(subset=["player_id"], keep="last")
        parts.append(d.select([pl.col("player_id"), pl.col("v").alias(name)]))
    base = parts[0]
    for d in parts[1:]:
        base = base.join(d, on="player_id", how="full", coalesce=True)
    return base


def _college_usage() -> pl.DataFrame:
    parts = []
    for y in COLLEGE_YEARS:
        u = cfbd_client.get_player_usage(y)
        if not u.height:
            continue
        parts.append(u.select([
            pl.col("id"),
            pl.col("usage_overall").alias("col_usage_overall"),
            pl.col("usage__pass").alias("col_usage_pass"),
            pl.col("usage_third_down").alias("col_usage_3rd"),
        ]))
    if not parts:
        return pl.DataFrame(schema={"id": pl.Int64, "col_usage_overall": pl.Float64,
                                    "col_usage_pass": pl.Float64, "col_usage_3rd": pl.Float64})
    return pl.concat(parts, how="vertical").unique(subset=["id"], keep="last")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_rookie_features() -> pl.DataFrame:
    nfl = _nfl_players()
    combine = _combine()
    draft = _cfbd_draft()
    college_stats = _college_stat_lookup()
    college_usage = _college_usage()

    # bridge: nflverse players + CFBD draft (by normalized name)
    base = nfl.join(
        draft.select(["name_key", "pre_draft_grade", "college_team",
                      "college_conference", "college_athlete_id", "round", "pick"]),
        on="name_key", how="left", suffix="_cfbd"
    )
    base = base.with_columns([
        pl.col("college_athlete_id").alias("college_id"),
        (pl.col("draft_pick").is_not_null()).alias("was_drafted"),
        (pl.col("draft_pick").is_null()).alias("udfa"),
    ])

    # college stats + usage via the CFBD college id
    base = base.join(college_stats.rename({"player_id": "college_id"}),
                     on="college_id", how="left")
    base = base.join(college_usage.rename({"id": "college_id"}),
                     on="college_id", how="left")

    # combine via name + draft year (combine cols don't collide with base, so no suffix)
    if combine.height:
        base = base.join(combine.select(["name_key", "draft_year", *COMBINE_COLS]),
                         on=["name_key", "draft_year"], how="left")
        base = base.rename({c: f"comb_{c}" for c in COMBINE_COLS if c in base.columns})
    else:
        base = base.with_columns([pl.lit(None, pl.Float64).alias(f"comb_{c}")
                                  for c in COMBINE_COLS])

    # age at draft (birth_date is a string date; cast to date for dt.year)
    base = base.with_columns(
        pl.col("birth_date").str.to_date("%Y-%m-%d").alias("_bd")
    )
    base = base.with_columns(
        pl.when(pl.col("draft_year").is_not_null() & pl.col("_bd").is_not_null())
        .then(pl.col("draft_year") - pl.col("_bd").dt.year())
        .otherwise(None).cast(pl.Int32).alias("age_at_draft")
    ).drop("_bd")

    out_cols = [
        "player_id", "display_name", "position",
        "draft_year", "draft_round", "draft_pick", "draft_team",
        "was_drafted", "udfa", "age_at_draft",
        "pre_draft_grade", "college_team", "college_conference",
        *(n for _, _, n in COLLEGE_STATS),
        "col_usage_overall", "col_usage_pass", "col_usage_3rd",
        *[f"comb_{c}" for c in COMBINE_COLS],
    ]
    return base.filter(
        pl.col("position").is_in(["QB", "RB", "WR", "TE"])
        & (pl.col("was_drafted") | pl.col("udfa"))
    ).select(out_cols)


def main() -> None:
    t0 = time.time()
    df = build_rookie_features()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "rookie_features.parquet"
    df.write_parquet(out)
    n = df.height
    nd = int(df.filter(pl.col("was_drafted")).height)
    nu = n - nd
    has_grade = len(df["pre_draft_grade"].drop_nulls())
    has_college = df.filter(
        pl.col("col_pass_yds").is_not_null() | pl.col("col_rush_yds").is_not_null()
        | pl.col("col_recv_yds").is_not_null()).height
    has_comb = (len(df["comb_forty"].drop_nulls()) if "comb_forty" in df.columns else 0)
    print(f"\nrookie prior features: {n} skill players ({nd} drafted, {nu} UDFA)")
    print(f"  with pre_draft_grade : {has_grade}")
    print(f"  with college stats   : {has_college}")
    print(f"  with combine 40      : {has_comb}")
    print(f"  -> {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
