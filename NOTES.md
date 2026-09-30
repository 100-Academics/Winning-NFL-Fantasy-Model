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
- **Prediction CLI (Phase 5):** `src/predict.py` (also the `predict` console
  script). Predicts any (season, week) present in `data/processed/features.parquet`.
  `--players` does a case-insensitive substring match on
  `player_display_name`. `--all` ranks a board by the model's **predicted**
  headline stat (not trailing form) and slices top N per position. `--compare`
  uses the same row's actual target columns (present only for in-data weeks) and
  prints a headline MAE/RMSE summary. `--refresh` re-pulls the current + prior
  season and rebuilds features for upcoming weeks.
  - **Gotcha:** in this polars build `str.contains(..., ignore_case=True)` is
    unsupported, and `pl.any([...])` is a column-aggregation (not an OR).
    Match by lower-casing both sides + `literal=True`, and OR masks with `|`.
  - **Gotcha:** a `(season, week)` row with no production for the target (e.g. a
    player out that week) is still a valid prediction row — don't treat "player
    not found" as a data error when the week itself has other rows.

## Head-to-head vs the named baselines (FantasyPros / ESPN)

- **FantasyPros public API (free tier)** — the primary "beat" target from
  NOTES baselines. Auth is the **`x-api-key`** header (NOT `Bearer`; the spec
  says the gateway hashes the Bearer token and wants `key=value`). Spec at
  `https://api.fantasypros.com/public/v2/docs/fantasypros_v2_public.yml`.
  Endpoint: `GET .../public/v2/json/nfl/{season}/projections?position={QB|RB|WR|TE}&week={0..17}`
  (week=0 = preseason). Free tier = **top-10 per position only**
  (`limit:10`, `public_api_limited:true`), **rate-limited** (429s are common —
  back off and retry). `stats` is a **dict** of the SAME raw components our
  model predicts (`pass_yds`,`pass_tds`,`rec_yds`,`rec_tds`,...) → clean
  apples-to-apples, no scoring ambiguity. Archived weeks for 2024 & 2025 are
  available. Key in gitignored `.env` as `FANTASYPROS_API_KEY`.
  - `src/fetch_fantasypros.py` pulls + caches boards to
    `data/processed/fantasypros/` (gitignored). `src/compare_fantasypros.py`
    does the head-to-head (2024 w1–12, n≈120/pos).
  - **Result (2024 test, vs FP consensus top-10, n≈120/pos, 480 pts-rows):**
    * Raw stat components (apples-to-apples): aggregate MAE **10.86 vs 11.73 —
      ours better**; we win MAE on ~10 of 14 stats (all yards stats, both TD
      yardage, TE receptions). FP wins the sparse count stats (QB pass INT,
      RB/WR receiving TDs) — those are near-zero events.
    * Weekly standard-1-PPR points (pooled n=480): MAE **7.13 vs 7.15**
      (near-tie), RMSE 9.77 vs 9.18 (FP slightly better), but **weekly rank
      Spearman 0.655 vs 0.601 — ours better** (also better at QB/WR/TE; FP
      leads the RB weekly rank).
    * Bottom line: **we beat the FantasyPros consensus on the raw components
      and on weekly ranking; we're statistically tied on weekly point totals.**
      Modest but real — the "beat the consensus" claim holds on signal/quality,
      not on a blowout. (Top-10 weekly rank is noisier than the full-roster
      split in `src/bench.py`.)
- **ESPN** — SWID **does** work, but only against the NEW host
  `lm-api-reads.fantasy.espn.com` with a **league id** (e.g. `.../leagues/899513?view=kona_player_info&scoringPeriodId=W`);
  the old `fantasy.espn.com` host is bot-walled (returns HTML) and needs no
  SWID but also none works there. Returns per-league fantasy **points**
  (not raw components) → less apples-to-apples. SWID in `.env` as `ESPN_SWID`.
- **Sleeper / Yahoo** — Sleeper's projection board now returns empty for every
  season/week (endpoint changed); `stats.sleeper.app` doesn't resolve on this
  box. Yahoo needs an `X-API-Key`. Not currently usable as an archived
  baseline.

## Calibration / signed-bias (why "MAE looks fine" is not enough)

- **MAE / RMSE / Spearman are all UNSIGNED — they hide systematic over/under-
  projection, which is exactly what distorts flex decisions.** Even with equal
  per-position MAE, if the model over-projects one flex position and
  under-projects another, the cross-position ranking used to fill flex is wrong.
  Diagnose signed bias, not just MAE. `src/calibrate.py` does this on the SAME
  2024–25 test split as `bench` (imports `SCORE`/`_ppr` from `src.bench`, so
  its per-position actual means match `bench_report.json` exactly).
- **Finding (test 2024–25):** the model UNDER-projects the flex positions
  heavily and QB is nearly unbiased — i.e. the model is **worse calibrated than
  both naive baselines**:
  - per-position signed bias (PPR): **RB −40.1%, TE −32.6%, WR −29.6%**
    (under) vs **QB −1.2%** (≈unbiased). Baselines: last_game −0.5 to −2.0,
    trailing3 −0.6 to −2.4 (QB much worse there) — so the model's RB/WR/TE
    under-projection is 3–6× larger than the baselines'.
  - **Root cause: RB/WR/TE TD predictions are 0.0 for ~100% of rows**
    (RB/WR/TE actuals: 20%/18%/15% of weeks score a TD). The sparse TD
    components collapse to 0, removing ~6 pts from every TD, plus the yardage
    stats are under-projected ~17–36%. (README already flags "conservative TD
    projections" — this quantifies it.)
  - **Top-of-board (flex stars) is under-projected too:** top-5% RB actual is
    **1.6×** predicted, WR **1.36×**, TE **1.34×**; but the top QB is slightly
    OVER-projected (actual **0.93×** predicted). So the global top tier looks
    fine (top-decile ratio 1.01) only because the RB/WR/TE under-estimate and
    the top-QB over-estimate cancel — that cancellation is what makes the
    aggregate MAE look acceptable.
  - **Flex distortion:** RB/WR/TE carry a −10.6-pt spread of relative bias,
    and the model over/under-projects them by up to 40% of their mean. A flex
    pick between them is systematically biased toward whatever is least
    under-projected (here WR, −29.6%) and away from RB (−40.1%).
- **How to fix (not yet done):** the bias is a *calibration* problem, not a
  ranking one (Spearman is fine). Options: (a) calibrate the per-position PPR
  output with an isotonic/Platt/quantile mapping fit on val; (b) stop
  collapsing TDs to 0 — model TD *probability* (binary) separately and add
  `6 × P(TD)`; (c) per-position bias correction (regress actual on predicted,
  apply slope/intercept) fit on val. Measure with `src/calibrate.py`: success
  = per-position rel-bias near 0 AND top-5% flex ratio near 1.0, without
  sacrificing the (already-good) Spearman.

## Reproduction

- `uv run python -m src.bench` — our model vs naive baselines in PPR space
  (2024–25 full roster). → `models/bench_report.json`.
- `uv run python -m src.calibrate` — signed bias per position + projected-vs-
  actual by predicted decile + per-stat bias + flex distortion (same split).
  → `models/calibration_report.json` + `notebooks/charts/calibration_*.png`.
- `uv run python -m src.fetch_fantasypros` then `uv run python -m src.compare_fantasypros`
  — head-to-head vs FantasyPros consensus. → `models/fantasypros_report.json`.
