"""No-lookahead (leakage) tests for the Phase 3 feature builder.

Core invariant: a feature for week ``W`` must NOT depend on the current week's
outcome (the target). Perturbing ``passing_yards`` (a target) at week ``W``
must leave every trailing / season-to-date / team / momentum feature for week
``W`` unchanged.

Run:
    uv run pytest tests/ -q
"""
from __future__ import annotations

import polars as pl

from src.clean import RELEVANT_POSITIONS  # noqa: F401  (ensures import works)
from src.features import build_features_from

# Feature families that are computed from trailing (prior) weeks only. Any of
# these that change when we perturb the current week's target is a LEAK.
TRAILING_FEATURE_PREFIXES = (
    "_tr3", "_tr5", "_s2d", "_s2d_total",
)
# Momentum features (shift(1) - shift(2)) — also prior-only.
MOMENTUM_SUFFIX = "_wow"

# Raw target columns (the current-week outcome). Perturbing these must not
# change the trailing features.
TARGET_COLS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds", "rushing_fumbles",
    "receptions", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "targets", "carries", "completions", "attempts",
]


def _synthetic_panel() -> pl.DataFrame:
    """Two players, two seasons, weeks 1-4 each, with a stable schedule context.

    Deterministic, no joins needed (team stats + NGS supplied separately).
    """
    rows = []
    for season in (2023, 2024):
        for pid, name in (("P1", "Alpha One"), ("P2", "Beta Two")):
            for week in (1, 2, 3, 4):
                # deterministic pseudo-stats
                base = (week * 10) + (50 if pid == "P1" else 0)
                rows.append({
                    "player_id": pid,
                    "player_display_name": name,
                    "season": season,
                    "week": week,
                    "team": "A" if pid == "P1" else "B",
                    "opponent_team": "B" if pid == "P1" else "A",
                    "game_id": f"{season}_{week:02d}_{'AB' if pid == 'P1' else 'BA'}",
                    "position": "QB" if pid == "P1" else "RB",
                    "position_group": "QB" if pid == "P1" else "RB",
                    # targets / stats
                    "passing_yards": float(base + 50 if pid == "P1" else 0),
                    "passing_tds": float(1 if week % 2 else 0),
                    "passing_interceptions": 0.0,
                    "passing_air_yards": float(base / 2),
                    "rushing_yards": float(20 if pid == "P2" else 0),
                    "rushing_tds": float(0.5 if week == 3 and pid == "P2" else 0),
                    "rushing_fumbles": 0.0,
                    "receptions": float(3 if pid == "P2" else 1),
                    "receiving_yards": float(40 if pid == "P2" else 10),
                    "receiving_tds": 0.0,
                    "receiving_fumbles": 0.0,
                    "targets": float(8 if pid == "P1" else 5),
                    "carries": float(12 if pid == "P2" else 0),
                    "completions": float(20 if pid == "P1" else 0),
                    "attempts": float(30 if pid == "P1" else 0),
                    # static pre-game context (constant so it can't mask a leak)
                    "home_flag": 1, "signed_spread": -3.0, "total_line": 44.0,
                    "player_implied_total": 23.5, "rest_days_player": 7,
                    "rest_days_opp": 7, "rest_diff": 0,
                    "temp": 60.0, "wind": 8.0, "roof_indoor": 0,
                })
    return pl.DataFrame(rows)


def _empty_team() -> pl.DataFrame:
    return pl.DataFrame({
        "season": pl.Series([], dtype=pl.Int32),
        "week": pl.Series([], dtype=pl.Int32),
        "team": pl.Series([], dtype=pl.String),
        "passing_yards": pl.Series([], dtype=pl.Int32),
        "rushing_yards": pl.Series([], dtype=pl.Int32),
        "receiving_yards": pl.Series([], dtype=pl.Int32),
    })


def _empty_ngs() -> dict[str, pl.DataFrame]:
    empty = pl.DataFrame({
        "season": pl.Series([], dtype=pl.Int32),
        "week": pl.Series([], dtype=pl.Int32),
        "player_display_name": pl.Series([], dtype=pl.String),
    })
    return {"passing": empty, "receiving": empty, "rushing": empty}


def _trailing_feature_cols(feat: pl.DataFrame) -> list[str]:
    cols = []
    for c in feat.columns:
        if c.endswith(TRAILING_FEATURE_PREFIXES) or c.endswith(MOMENTUM_SUFFIX):
            cols.append(c)
    return cols


def test_perturbing_current_week_target_does_not_change_trailing_features():
    panel = _synthetic_panel()
    team = _empty_team()
    ngs = _empty_ngs()

    feat = build_features_from(panel, team, ngs)
    tr_cols = _trailing_feature_cols(feat)
    assert tr_cols, "expected to find trailing feature columns"

    # Pick a concrete (player, season, week) in the middle of the season so
    # there ARE prior weeks (otherwise trailing is null anyway).
    target_week = 3
    target_player = "P1"
    target_season = 2024

    before = feat.filter(
        (pl.col("player_id") == target_player)
        & (pl.col("season") == target_season)
        & (pl.col("week") == target_week)
    ).select(tr_cols)

    # Perturb the CURRENT week's targets for that player (big change).
    mask = (
        (pl.col("player_id") == target_player)
        & (pl.col("season") == target_season)
        & (pl.col("week") == target_week)
    )
    perturbed = panel.with_columns([
        pl.when(mask).then(pl.col("passing_yards") + 5000).otherwise(pl.col("passing_yards")).alias("passing_yards"),
        pl.when(mask).then(pl.col("passing_tds") + 25).otherwise(pl.col("passing_tds")).alias("passing_tds"),
        pl.when(mask).then(pl.col("receptions") + 30).otherwise(pl.col("receptions")).alias("receptions"),
        pl.when(mask).then(pl.col("targets") + 40).otherwise(pl.col("targets")).alias("targets"),
    ])

    # Confirm the perturbation actually took effect on the target.
    assert (perturbed.filter(mask)["passing_yards"][0]
            > panel.filter(mask)["passing_yards"][0]), \
        "sanity: perturbation should have changed the target"

    feat_p = build_features_from(perturbed, team, ngs)
    after = feat_p.filter(
        (pl.col("player_id") == target_player)
        & (pl.col("season") == target_season)
        & (pl.col("week") == target_week)
    ).select(tr_cols)

    # Every trailing feature at the target week must be identical.
    changed = []
    for c in tr_cols:
        b, a = before[c][0], after[c][0]
        if b is None and a is None:
            continue
        if b is None or a is None:
            changed.append((c, b, a)); continue
        if abs(float(b) - float(a)) > 1e-9:
            changed.append((c, b, a))
    assert not changed, (
        f"LEAK: perturbing week {target_week}'s targets changed trailing "
        f"features for that same week: {changed}"
    )


def test_future_week_does_not_change_prior_week_features():
    """Perturbing a LATER week must not change an EARLIER week's trailing features."""
    panel = _synthetic_panel()
    team = _empty_team()
    ngs = _empty_ngs()

    feat = build_features_from(panel, team, ngs)
    tr_cols = _trailing_feature_cols(feat)

    prior_player, prior_season, prior_week = "P1", 2024, 2
    later_player, later_season, later_week = "P1", 2024, 4

    before = feat.filter(
        (pl.col("player_id") == prior_player)
        & (pl.col("season") == prior_season)
        & (pl.col("week") == prior_week)
    ).select(tr_cols)

    mask = (
        (pl.col("player_id") == later_player)
        & (pl.col("season") == later_season)
        & (pl.col("week") == later_week)
    )
    perturbed = panel.with_columns([
        pl.when(mask).then(pl.col("passing_yards") + 9999).otherwise(pl.col("passing_yards")).alias("passing_yards"),
        pl.when(mask).then(pl.col("receptions") + 99).otherwise(pl.col("receptions")).alias("receptions"),
    ])
    feat_p = build_features_from(perturbed, team, ngs)
    after = feat_p.filter(
        (pl.col("player_id") == prior_player)
        & (pl.col("season") == prior_season)
        & (pl.col("week") == prior_week)
    ).select(tr_cols)

    changed = []
    for c in tr_cols:
        b, a = before[c][0], after[c][0]
        if (b is None) == (a is None) and not (b is None):
            if abs(float(b) - float(a)) > 1e-9:
                changed.append((c, b, a))
    assert not changed, f"LEAK: perturbing week {later_week} changed week {prior_week} features: {changed}"
