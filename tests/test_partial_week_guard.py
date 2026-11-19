"""Partial-week log guard (regression: Week 8 Thursday-night edge case).

``src.holdout._target_week_fully_played`` decides which feature source
``predict_week`` uses: a fully-played week keeps its real feature rows (the
replay path); an unplayed or PARTIALLY-played week must fall back to the
scheduled panel, whose trailing features are strictly prior-week. Before the
guard, ``rows.height == 0`` was the only test, so a week whose Thursday game
had already landed in features.parquet logged the played rows and silently
skipped the rest of the slate.

Run:
    uv run pytest tests/ -q
"""
from __future__ import annotations

import polars as pl

from src.holdout import _target_week_fully_played


def _sched(rows: list[tuple[int, int, int | None, int | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        orient="row",
        schema=["season", "week", "home_score", "away_score"],
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


def test_all_games_finished_is_fully_played():
    assert _target_week_fully_played(
        _sched([(2026, 8, 21, 17), (2026, 8, 14, 7)])) is True


def test_one_game_missing_score_is_partial():
    assert _target_week_fully_played(
        _sched([(2026, 8, 21, 17), (2026, 8, None, None)])) is False


def test_no_games_scheduled_is_not_fully_played():
    assert _target_week_fully_played(_sched([])) is False
