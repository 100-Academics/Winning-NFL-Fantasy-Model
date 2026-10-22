"""Central fantasy-scoring weights (one place, per user directive).

The user's league is **ESPN Standard (non-PPR)**:

    passing    0.04 pt / yd,  4 pt / TD,  -2 pt / INT
    rushing    0.10 pt / yd,  6 pt / TD
    receiving  0.10 pt / yd,  6 pt / TD
    (NO point per reception)

Historical note: `src/bench`, `src/calibrate`, `src/holdout`, and
`src/compare_fantasypros` each carried their own copy of a "standard 1-PPR"
table that actually used 0.10 pt / passing yard + 6 pt / passing TD (a
heavy-QB scheme matching no common format), plus +1 PPR per reception.
That inflated QB points vs the user's real league (~+15-20 pts/week on a
250-yard game: 25 vs 10) and made the pooled cross-position holdout
Spearman measure a different ranking space than the user's start/sit
decisions. 2026-10-22: consolidated here and switched to the user's actual
ESPN Standard scoring. The 1-PPR holdout scores are archived in
`models/holdout_score_1ppr_archive.json` (per-position Spearman is identical
under either scheme — only pooled cross-position numbers and absolute
MAE/RMSE change).
"""
from __future__ import annotations

ESPN_STANDARD: dict[str, float] = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "passing_interceptions": -2.0,
    "rushing_yards": 0.10,
    "rushing_tds": 6.0,
    "receiving_yards": 0.10,
    "receiving_tds": 6.0,
}

SCORING_LABEL = "ESPN Standard (non-PPR)"


def fantasy_points(preds: dict[str, float], weights: dict[str, float] = ESPN_STANDARD) -> float:
    """Convert raw stat components to fantasy points under ``weights``."""
    return sum(weights.get(k, 0.0) * float(v or 0.0) for k, v in preds.items())
