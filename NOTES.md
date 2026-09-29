# Working notes

## Scope

- We predict **per-player per-week stat components** (passing/rushing/receiving
  yards & TDs, receptions, targets, carries, …). **Fantasy points are NOT a
  target here** — that's a separate, later project that converts these
  components. (User directive, 2026-09-29.)
- nflverse ships `fantasy_points`/`fantasy_points_ppr` in the weekly stats —
  fine to keep for sanity-checking, but don't build the model around them.

## Tooling

- **Python environment: use `uv`.** No pip/venv/conda by hand — for any new
  dependency, edit `pyproject.toml` then run `uv sync` (or `uv add <pkg>`).
  Run scripts with `uv run python ...`. The venv is `.venv/` (gitignored),
  interpreter pinned to CPython 3.11 via `requires-python = ">=3.11"`.
  (User directive, 2026-09-29.)

## Data & modeling decisions

- Data source: **nflverse** (nflreadpy is the current Python package; it
  supersedes the older `nfl_data_py`). nflverse distributes parquet/CSV on
  GitHub releases, so mirror locally into `data/raw/`.
  - nflverse packages: nflfastR (PBP since 1999), nflseedR (simulations),
    nfl4th (4th-down analysis), nflreadr (downloads), nflplotR (viz).
- Baselines to beat / use as sanity checks: **FantasyPros consensus**
  rankings & projections; ESPN, Yahoo, Sleeper projections.
- **Vegas lines AND weather ARE in nflverse** — the `schedules` release
  carries `spread_line`, `total_line`, `away/home_moneyline`, `over/under_odds`,
  plus `temp`, `wind`, `roof`, `surface`, `away/home_rest`. Use spread/total
  as inputs (don't try to beat them).
- Features that move the needle: **Vegas lines** (spread/total → implied team
  totals), **weather** (temp/wind for passing games + kickers), **injury
  reports/inactives**, snap counts, route participation, **target share**,
  **red-zone usage**.
- **nflverse player names are abbreviated** (e.g. `P.Mahomes`); use
  `player_display_name` for human-facing output, `player_id` (e.g. `00-0033873`)
  as the stable join key.
- Pro Football Reference: supplementary historical + college stats;
  rate-limits scraping aggressively — use sparingly.
- **Lookahead leakage is the #1 risk**: every feature must be computable
  strictly before the week being predicted.
- **Polars `.over()` windows follow PHYSICAL ROW ORDER within each group, not
  the `week` column.** Always `.sort([key..., "season", "week"])` BEFORE
  applying `shift`/`rolling` `.over(...)`, or "trailing" silently uses the wrong
  rows. (This bit us — the panel is not week-sorted, so the first feature pass
  was leaking. Fixed by sorting first in every window site.)
- **NextGen Stats (NGS) weekly values are post-game outcomes** (computed from
  that week's games), so treat them as *trailing* (prior weeks) inputs, never
  same-week — same leak rule as everything else.
- **Team "points" can't be cleanly derived** from the team-stat TD columns:
  those TD columns also count *opponent* scoring, so a score formula
  (7×TD + 3×FG + …) doesn't reconcile to the final score. Use **team total
  yards** as the offense/defense-strength proxy instead (robust, no
  reconciliation).
- **Momentum must be prior-only**: `value[W-1] - value[W-2]`, NOT
  `value[W] - value[W-1]` (the latter uses the current week = the target).
- **nflverse `game_id` is a clean join key** across player/team/schedules (0
  orphans). Team `(season,week,team)` is unique in team stats. Join on
  `game_id`, not on `(season,week)` (that fans out across the week's games).
- **Model (Phase 4):** per (position, target) use
  `sklearn.ensemble.HistGradientBoostingRegressor(loss="quantile")` — fast
  (~0.7s/fit), no extra deps. Split train 2016–22 / val 2023 / test 2024–25.
  All 14 position-stats beat the last-game & trailing-3 baselines on test MAE;
  Spearman 0.60–0.77 on headline yardage stats. Headline stats get p10/p90 band
  models (74–89% of actuals inside the 80% band). Saved to `models/models.joblib`
  + `models/eval_report.json`.
- **Gotcha:** `df.select([...]).to_numpy()` on a single-column frame returns
  shape (n,1) → broadcast-bugs against (n,) predictions when computing MAE
  (n×n). Use `.to_series().to_numpy()` for y-vectors, or ravel.
