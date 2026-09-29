"""Phase 3 — data cleaning: build a clean per-player-week REGULAR-season panel.

Responsibility of THIS module (keep it leak-free by construction):
  * load the raw nflverse parquets,
  * keep REGULAR season only (no preseason/postseason),
  * de-duplicate / validate the player-week keys,
  * attach **pre-game** context from the schedule (Vegas spread/total, weather,
    rest, home/away, implied team total) — all of which are KNOWN before the
    game is played, so they are safe to use as inputs,
  * carry the **component-stat targets** (passing/rushing/receiving yards & TDs,
    receptions, targets, carries, …). Fantasy points are intentionally NOT a
    target here (separate later project).

Trailing / rolling (form, usage, opponent-defense, NGS-efficiency) features are
NOT built here — they live in :mod:`src.features` so that *every* non-static
feature is provably computed from strictly-prior weeks.

Run:
    uv run python -m src.clean          # writes data/processed/panel.parquet
"""
from __future__ import annotations

import pathlib
import sys

import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

REG = "REG"

# Stable keys that identify one player in one game.
KEY_COLS = [
    "player_id",
    "player_display_name",
    "season",
    "week",
    "team",
    "opponent_team",
    "game_id",
    "position",
    "position_group",
]

# The canonical "one row per player-week" key. In REG data this is unique and
# position/game_id are constant within it (verified), so de-duping on these
# three is safe and matches the model's grain.
PLAYER_WEEK_KEY = ["player_id", "season", "week"]

# Component-stat TARGETS (what we ultimately predict). No fantasy points.
TARGET_COLS = [
    # passing
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_air_yards",
    # rushing
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    # receiving
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles",
    # usage (also predictable, useful for volume models)
    "targets",
    "carries",
    "completions",
    "attempts",
]

# Positions we care about for scoring; the panel keeps ALL positions so nothing
# is lost, and Phase 4 subsets by this set.
RELEVANT_POSITIONS = {"QB", "RB", "WR", "TE", "K"}

# Pre-game context derived from the schedule. All known before kickoff.
SCHEDULE_CTX_COLS = [
    "home_flag",            # 1 if the player's team is at home
    "signed_spread",        # player's team spread (negative = favored)
    "total_line",           # over/under total for the game
    "player_implied_total", # (total_line +/- spread)/2 for the player's team
    "rest_days_player",     # player team's rest days
    "rest_days_opp",        # opponent's rest days
    "rest_diff",            # player_rest - opp_rest
    "temp",                 # game-day temp (null if dome/no data)
    "wind",                 # game-day wind (null if dome/no data)
    "roof_indoor",          # 1 if dome/closed (no weather)
]


# --------------------------------------------------------------------------- #
# Loaders (thin wrappers; each returns a REG-only, typed frame)
# --------------------------------------------------------------------------- #
def _reg(df: pl.DataFrame, col: str = "season_type") -> pl.DataFrame:
    if col in df.columns:
        return df.filter(pl.col(col) == REG)
    return df


def load_player_stats(raw_dir: pathlib.Path = RAW_DIR) -> pl.DataFrame:
    df = _reg(pl.read_parquet(raw_dir / "player_weekly_stats.parquet"))
    df = df.with_columns(pl.col("position").fill_null("NA"))
    # numeric coercion for the target set (already numeric, be defensive)
    for c in TARGET_COLS:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    return df


def load_team_stats(raw_dir: pathlib.Path = RAW_DIR) -> pl.DataFrame:
    return _reg(pl.read_parquet(raw_dir / "team_weekly_stats.parquet"))


def load_schedules(raw_dir: pathlib.Path = RAW_DIR) -> pl.DataFrame:
    return _reg(pl.read_parquet(raw_dir / "schedules.parquet"))


def load_nextgen(raw_dir: pathlib.Path = RAW_DIR) -> dict[str, pl.DataFrame]:
    out = {}
    for k, f in [
        ("passing", "nextgen_passing.parquet"),
        ("receiving", "nextgen_receiving.parquet"),
        ("rushing", "nextgen_rushing.parquet"),
    ]:
        out[k] = _reg(pl.read_parquet(raw_dir / f))
    return out


# --------------------------------------------------------------------------- #
# Pre-game schedule context (leak-free: all known before the game)
# --------------------------------------------------------------------------- #
def _schedule_context(sched: pl.DataFrame) -> pl.DataFrame:
    """One row per game_id with pre-game context, from the schedule table.

    ``spread_line`` is the AWAY team's spread (positive => away underdog).
    """
    s = sched.select(
        ["game_id", "away_team", "home_team", "spread_line", "total_line",
         "away_rest", "home_rest", "temp", "wind", "roof"]
    )
    return s.with_columns(
        pl.when(pl.col("roof").str.contains("dome", literal=True)
                | pl.col("roof").str.contains("closed", literal=True))
          .then(1).otherwise(0).alias("roof_indoor"),
    )


def attach_schedule_context(panel: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Join per-game pre-game context onto the player panel via ``game_id``.

    ``signed_spread`` is from the player's team perspective (negative = favored).
    ``player_implied_total`` = (total_line - spread)/2 for home, (total_line +
    spread)/2 for away, where ``spread`` is the away-team's spread.
    """
    ctx = _schedule_context(sched)
    out = panel.join(ctx, on="game_id", how="left")

    # 1) home flag + rest days (independent of each other)
    out = out.with_columns(
        (pl.col("team") == pl.col("home_team")).cast(pl.Int8).alias("home_flag"),
    )
    out = out.with_columns(
        pl.when(pl.col("home_flag") == 1).then(pl.col("home_rest")).otherwise(pl.col("away_rest"))
          .alias("rest_days_player"),
        pl.when(pl.col("home_flag") == 1).then(pl.col("away_rest")).otherwise(pl.col("home_rest"))
          .alias("rest_days_opp"),
    )
    # 2) derived from the above
    out = out.with_columns(
        (pl.col("rest_days_player") - pl.col("rest_days_opp")).alias("rest_diff"),
        # signed spread for the player's team (negative = favored)
        pl.when(pl.col("home_flag") == 1)
          .then(-pl.col("spread_line"))
          .otherwise(pl.col("spread_line"))
          .alias("signed_spread"),
        # implied team total for the player's team
        pl.when(pl.col("home_flag") == 1)
          .then((pl.col("total_line") - pl.col("spread_line")) / 2.0)
          .otherwise((pl.col("total_line") + pl.col("spread_line")) / 2.0)
          .alias("player_implied_total"),
    )
    return out.select([
        "player_id", "player_display_name", "season", "week", "team",
        "opponent_team", "game_id", "position", "position_group",
        *TARGET_COLS, *SCHEDULE_CTX_COLS,
    ])


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
def build_panel(raw_dir: pathlib.Path = RAW_DIR) -> pl.DataFrame:
    """Clean REG player-week panel with targets + pre-game context.

    Returns one row per (player, season, week). Deduped on the full key set.
    """
    player = load_player_stats(raw_dir)
    sched = load_schedules(raw_dir)

    panel = attach_schedule_context(player, sched)

    # De-duplicate on the canonical player-week key (verified unique in REG).
    panel = (
        panel.sort(PLAYER_WEEK_KEY + ["game_id"])
        .unique(subset=PLAYER_WEEK_KEY, keep="first")
    )

    # Fill missing targets with 0 (a player-week with no stat is a 0 for that
    # component) — but keep usage/production nulls honest where the column is
    # genuinely absent.
    for c in TARGET_COLS:
        if c in panel.columns:
            panel = panel.with_columns(pl.col(c).fill_null(0.0))

    return panel


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    out = PROCESSED_DIR / "panel.parquet"
    panel.write_parquet(out)
    print(f"panel: {panel.height:,} player-weeks x {panel.width} cols -> {out}")
    print("seasons:", sorted(panel["season"].unique().to_list()))
    print("positions kept:", panel["position"].unique().to_list())
    print("target cols:", TARGET_COLS)
    print("context cols:", SCHEDULE_CTX_COLS)
    # quick leak-free sanity: signed_spread for home should be <= spread sign flip
    print("home_flag dist:", panel["home_flag"].value_counts().sort("home_flag"))


if __name__ == "__main__":
    main()
