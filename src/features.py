"""Phase 3 — leakage-safe feature engineering.

Everything here is computed from **strictly prior** weeks (never the current
week, never future weeks). The leak-free recipe is:

    x_prior  = value.shift(1) over [key..., season]     # drop the current week
    trailing = x_prior.rolling_mean(window, min_samples=1) over [key..., season]

``.over([..., season])`` also guarantees the window does NOT cross a season
boundary (week 1 of 2025 never looks at 2024). This was validated against
hand-computed values in ``notebooks/explore_phase3.py``.

Feature families (all trailing / prior-week):
  * player_form  — trailing 3/5 of each component stat, plus season-to-date rate
  * usage        — trailing 3/5 of targets, carries, attempts, receptions;
                   season-to-date targets/carries (volume proxy)
  * momentum     — current-week value minus last-week value (week-over-week)
  * team_offense — trailing 3/5 of the player's OWN team total points & yards
  * opp_defense  — trailing 3/5 of the OPPONENT team total points & yards (allowed)
  * nextgen      — trailing 3 of NGS efficiency (comp%, air yds, cushion,
                   separation, yds over expected) — NGS values are post-game,
                   so they are treated as trailing, not same-week inputs

Run:
    uv run python -m src.features        # reads panel, writes features.parquet
"""
from __future__ import annotations

import pathlib

import polars as pl

from src.clean import (
    RELEVANT_POSITIONS,
    PROCESSED_DIR,
    RAW_DIR,
    build_panel,
    load_nextgen,
    load_team_stats,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

KEYS = ["player_id", "season"]  # window is per (player, season), ordered by week

# Component stats that are ALSO meaningful as trailing "form" features.
FORM_STATS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "targets", "carries", "completions", "attempts",
]

WINDOWS = [3, 5]


# --------------------------------------------------------------------------- #
# Rolling helpers (all strictly-prior)
# --------------------------------------------------------------------------- #
def _trailing(key_cols: list[str], col: str, window: int) -> pl.Expr:
    """Last ``window`` games BEFORE the current one, mean (nulls skipped)."""
    return (
        pl.col(col).shift(1).over(key_cols + ["season"])
        .rolling_mean(window, min_samples=1)
        .over(key_cols + ["season"])
    )


def _s2d_rate(key_cols: list[str], col: str) -> pl.Expr:
    """Season-to-date MEAN of prior games for ``col`` (0 if no prior games).

    Uses a wide rolling_mean over the shifted series so week 1 (no prior games)
    yields null -> 0 rather than 0/0 = NaN.
    """
    big = 20  # max regular-season games is 18; 20 covers the whole season
    prior = pl.col(col).shift(1).over(key_cols + ["season"])
    return (
        prior.rolling_mean(big, min_samples=1)
        .over(key_cols + ["season"]).fill_null(0.0)
    )


def _wow(key_cols: list[str], col: str) -> pl.Expr:
    """Week-over-week MOMENTUM, prior-only: last game minus the game before.

    ``value[W-1] - value[W-2]`` — deliberately does NOT use the current week
    ``value[W]`` (that is the target we're trying to predict). A leak-free
    "form momentum" signal.
    """
    return (
        pl.col(col).shift(1).over(key_cols + ["season"])
        - pl.col(col).shift(2).over(key_cols + ["season"])
    )


def _add_trailing(df: pl.DataFrame, key_cols: list[str], cols: list[str]) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for c in cols:
        for w in WINDOWS:
            exprs.append(_trailing(key_cols, c, w).alias(f"{c}_tr{w}"))
        exprs.append(_s2d_rate(key_cols, c).alias(f"{c}_s2d"))
    return df.with_columns(exprs)


# --------------------------------------------------------------------------- #
# Player-level features (from the panel's own stats)
# --------------------------------------------------------------------------- #
def _player_features(panel: pl.DataFrame) -> pl.DataFrame:
    # CRITICAL: Polars `.over()` windows follow physical row order within each
    # group, so the frame MUST be week-sorted inside each (player, season)
    # before any shift/rolling — otherwise "trailing" is not actually prior.
    p = panel.sort(KEYS + ["week"])
    p = _add_trailing(p, KEYS, FORM_STATS)
    # usage volume season-to-date (distinct from form)
    for c in ["targets", "carries"]:
        big = 20
        prior = pl.col(c).shift(1).over(KEYS + ["season"])
        p = p.with_columns(
            prior.fill_null(0.0).rolling_sum(big, min_samples=1)
            .over(KEYS + ["season"]).fill_null(0.0).alias(f"{c}_s2d_total")
        )
    # week-over-week momentum on the headline stats
    for c in ["passing_yards", "rushing_yards", "receiving_yards",
              "receptions", "targets", "carries"]:
        p = p.with_columns(_wow(KEYS, c).alias(f"{c}_wow"))
    # games played so far this season (recency / availability signal)
    big = 20
    p = p.with_columns(
        pl.col("week").is_not_null()
        .shift(1).over(KEYS + ["season"])
        .rolling_sum(big, min_samples=1).over(KEYS + ["season"])
        .cast(pl.Float64).fill_null(0.0).alias("games_played_s2d")
    )
    return p


# --------------------------------------------------------------------------- #
# Team-offense & opponent-defense (from team stats, trailing)
# --------------------------------------------------------------------------- #
def _team_totals(team: pl.DataFrame) -> pl.DataFrame:
    """Trailing (prior-week) strength of a team's OFFENSE.

    Uses total yards (passing + rushing + receiving) rather than a derived
    points total — the team-stat TD columns also count opponent scoring, so a
    clean score formula isn't reliable, and yards are a more robust "allowed"
    proxy. All values are trailing (strictly prior weeks) within the season.
    """
    t = team.select(["season", "week", "team",
                     "passing_yards", "rushing_yards", "receiving_yards"])
    t = t.with_columns(
        (pl.col("passing_yards") + pl.col("rushing_yards") + pl.col("receiving_yards"))
        .cast(pl.Float64).fill_null(0.0).alias("total_yards")
    )
    t = t.sort(["team", "season", "week"])  # window order must be week-sorted
    t = _add_trailing(t, ["team", "season"], ["total_yards"])
    keep = ["season", "week", "team"]
    for w in WINDOWS:
        keep.append(f"total_yards_tr{w}")
    keep.append("total_yards_s2d")
    return t.select(keep).unique(subset=["season", "week", "team"])


def _attach_team_and_opp(panel: pl.DataFrame, team: pl.DataFrame) -> pl.DataFrame:
    tt = _team_totals(team)
    # own team offense
    own = tt.rename({"team": "team"}).select(
        ["season", "week", "team"] + [c for c in tt.columns if c not in ("season", "week", "team")]
    )
    out = panel.join(own, on=["season", "week", "team"], how="left", suffix="_team")
    # opponent defense (same trailing stats, but the opponent's team)
    opp = tt.rename({"team": "opponent_team"}).select(
        ["season", "week", "opponent_team"] + [c for c in tt.columns if c not in ("season", "week", "team")]
    )
    out = out.join(opp, on=["season", "week", "opponent_team"], how="left", suffix="_opp")
    return out


# --------------------------------------------------------------------------- #
# NextGen efficiency (trailing; NGS values are post-game outcomes)
# --------------------------------------------------------------------------- #
def _ngs_trailing(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    keys = ["player_display_name", "season"]
    df = df.sort(keys + ["week"])  # window order must be week-sorted
    out = df.with_columns(
        [_trailing(keys, c, 3).alias(f"ngs_{c}_tr3") for c in cols]
    )
    return out.select(
        ["season", "week", "player_display_name"] + [f"ngs_{c}_tr3" for c in cols]
    ).unique(subset=["season", "week", "player_display_name"])


def _attach_nextgen(panel: pl.DataFrame, ngs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    out = panel
    # passing (QB)
    if ngs["passing"].height:
        cols = ["completion_percentage", "avg_time_to_throw", "aggressiveness",
                "avg_completed_air_yards"]
        g = ngs["passing"].select(
            ["season", "week", "player_display_name"] +
            [c for c in cols if c in ngs["passing"].columns]
        )
        g = _ngs_trailing(g, [c for c in cols if c in g.columns])
        out = out.join(g, on=["season", "week", "player_display_name"], how="left")
    # receiving (WR/TE)
    if ngs["receiving"].height:
        cols = ["avg_cushion", "avg_separation", "percent_share_of_intended_air_yards",
                "catch_percentage"]
        g = ngs["receiving"].select(
            ["season", "week", "player_display_name"] +
            [c for c in cols if c in ngs["receiving"].columns]
        )
        g = _ngs_trailing(g, [c for c in cols if c in g.columns])
        out = out.join(g, on=["season", "week", "player_display_name"], how="left")
    # rushing (RB)
    if ngs["rushing"].height:
        cols = ["efficiency", "rush_yards_over_expected_per_att", "avg_time_to_los"]
        g = ngs["rushing"].select(
            ["season", "week", "player_display_name"] +
            [c for c in cols if c in ngs["rushing"].columns]
        )
        g = _ngs_trailing(g, [c for c in cols if c in g.columns])
        out = out.join(g, on=["season", "week", "player_display_name"], how="left")
    return out


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def build_features_from(panel: pl.DataFrame, team: pl.DataFrame,
                        ngs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Build leakage-safe features from pre-loaded inputs.

    Split out from :func:`build_features` so tests can perturb ``panel`` and
    rebuild without re-reading every raw file.
    """
    # Keep the scoring positions we model (QB/RB/WR/TE/K).
    feat = panel.filter(pl.col("position").is_in(sorted(RELEVANT_POSITIONS)))
    feat = _player_features(feat)
    feat = _attach_team_and_opp(feat, team)
    feat = _attach_nextgen(feat, ngs)
    return feat


def build_features(raw_dir: pathlib.Path = RAW_DIR) -> pl.DataFrame:
    panel = build_panel(raw_dir)
    team = load_team_stats(raw_dir)
    ngs = load_nextgen(raw_dir)
    return build_features_from(panel, team, ngs)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feat = build_features()
    out = PROCESSED_DIR / "features.parquet"
    feat.write_parquet(out)
    n_features = feat.width - (
        # keys + targets + context are not "model features"; count them roughly
        9  # player_id, display_name, season, week, team, opp, game_id, position, position_group
    )
    print(f"features: {feat.height:,} rows x {feat.width} cols -> {out}")
    print(f"scoring positions: {feat['position'].unique().to_list()}")
    print(f"approx model-feature count: {n_features}")
    # show a few feature null-rates to confirm trailing worked
    sample = ["passing_yards_tr3", "passing_yards_s2d", "total_points_tr3_team",
              "total_points_tr3_opp", "targets_s2d_total"]
    sample = [c for c in sample if c in feat.columns]
    for c in sample:
        print(f"  {c:<22} nulls={feat[c].is_null().sum():>7} of {feat.height}")


if __name__ == "__main__":
    main()
