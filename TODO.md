# TODO: NFL Player Stat Projection Model

Goal: predict per-player per-week **component stats**, e.g.
"[Player xyz] will get 4.5 touchdowns and 202 yards."

We predict the raw stat components (passing/rushing/receiving yards & TDs,
receptions, targets, carries, etc.) and do NOT compute fantasy points here —
that is a separate, later project.

## Phase 1 — Project setup

- [x] Set up Python environment (venv or conda), pin Python 3.11+
- [x] Install core libs: pandas, numpy, scikit-learn, matplotlib, jupyter
- [x] Create directory layout: `data/raw`, `data/processed`, `src`, `notebooks`, `models`
- [x] Add `.gitignore` (exclude `data/raw`, `models/`, venv dirs)
- [x] Add `requirements.txt` or `pyproject.toml` and keep it updated

## Phase 2 — Historical data acquisition

- [x] Choose a data source. Candidates:
  - ~~nfl_data_py (free, wraps nflfastR play-by-play + weekly rosters/stats)~~ — **DEPRECATED** in favour of nflreadpy (confirmed via nfl_data_py README)
  - **nflreadpy (current recommendation) — SELECTED.** Python port of nflreadr; downloads the same nflverse-data releases, Polars-backed, pip/uv-installable.
  - Sleeper/FantasyData/SportsDataIO APIs (alternative stat/roster sources, some paid)
- [x] Install and pull historical weekly player stats (target: 2016–2025 seasons, or earlier)
  - columns needed: player, team, position, week, season, passing_yards, passing_tds,
    rushing_yards, rushing_tds, receiving_yards, receiving_tds, receptions, targets,
    carries — **all present** in `player_weekly_stats.parquet` (150 cols).
    Note: nflverse names are abbreviated (e.g. `P.Mahomes`); use
    `player_display_name` for human-facing output.
- [x] Pull supporting context data:
  - game schedules/results (opponent, home/away, Vegas spread/total) — `schedules.parquet` (opponent/home/away + scores, AND Vegas `spread_line`/`total_line`/moneylines + weather `temp`/`wind`/`roof`/`surface`)
  - player usage: snap counts, target share, red-zone touches — `snap_counts.parquet`, `target_share` in player stats
  - injuries/roster status changes (nflfastR has some; may need supplements) — `injuries.parquet`, `weekly_rosters.parquet`
- [x] Write a `src/download_data.py` script so the pull is reproducible (resumable, `--force`, groups core/context/advanced)
- [x] Save raw pulls to `data/raw/<season>_weekly.parquet` (parquet, not CSV — much smaller/faster)
  - actual files: `player_weekly_stats.parquet`, `team_weekly_stats.parquet`, `schedules.parquet`,
    `players.parquet`, `weekly_rosters.parquet`, `snap_counts.parquet`, `injuries.parquet`

## Phase 3 — Data cleaning & feature engineering

- [x] Clean and validate: dedupe player names across seasons, handle team changes,
  fill/handle injury weeks, drop irrelevant positions (e.g. OL/DT) or keep for analysis
  - `src/clean.py` → `data/processed/panel.parquet`: REG-only, de-duped on
    `(player_id, season, week)` (verified unique), nulls→0 for targets,
    pre-game schedule context attached via `game_id`.
- [x] Define the prediction target per player-week — the **stat components**:
  passing yards/tds, rushing yards/tds, receiving yards/tds, receptions, targets,
  carries. (Fantasy points are a separate, later project — NOT a target here.)
  - Panel keeps ALL positions; the feature builder subsets to QB/RB/WR/TE/K.
- [x] Build rolling features (avoid lookahead leakage! only use data from prior weeks):
  - trailing 3-game and 5-game averages of usage and production → `_tr3`/`_tr5`
  - season-to-date totals and per-game rates → `_s2d` (mean of prior games, 0 if none)
  - opponent defense strength vs position (rolling yards allowed) → `total_yards_tr3_opp` etc.
  - team implied total from Vegas line, home/away flag → `player_implied_total`, `signed_spread`, `home_flag`
  - target share / air-yard share trends, red-zone opportunity rate → NGS trailing
    (`ngs_completion_percentage_tr3`, `ngs_avg_cushion_tr3`, `ngs_avg_separation_tr3`, `ngs_efficiency_tr3`, …)
- [x] Write `src/features.py` with a `build_features(...) -> DataFrame` function
  (61,827 scoring rows × 96 cols, 0 NaNs)
- [x] Sanity-check: no feature uses same-week or future information (add a test)
  - `tests/test_no_lookahead.py`: perturbing a week's targets leaves that week's
    trailing/momentum features unchanged; future-week perturb doesn't touch
    prior weeks. 2 pass; proven non-vacuous (removing the `shift(1)` makes it fail).

## Phase 4 — Model

- [x] Baseline first: predict trailing 3-game average (or last game) as the naive forecast.
  Measure MAE/RMSE per stat — the model must beat this to be worth anything.
  - `src/model.py` builds two baselines per position-stat: `last_game` (player's
    own prior-week value) and `trailing3`.
- [x] Split data chronologically: train on earlier seasons, validate on a held-out
  mid-season, test on the most recent season(s). NEVER random splits (time leakage).
  - **train 2016–2022 | val 2023 | test 2024–2025.**
- [x] Train candidate models per target stat (or one multi-output model):
  - Gradient boosting: XGBoost / LightGBM / sklearn HistGradientBoosting
    → **`HistGradientBoostingRegressor(loss="quantile")`** (sklearn, no new deps).
  - Quantile regression (gradient boosting with quantile loss) for "4.5 tds"-style
    medians — gives a defensible central estimate + uncertainty bands
    → headline stats also get p10/p90 band models (calibrated: 74–89% of actuals
      land inside the 80% band).
- [x] Tune hyperparameters on the validation split (small grid search is fine)
  - grid over max_iter {150,300,500} × lr {0.03,0.06} × depth {3,5} × l2 {1,5};
    `--quick` flag for a fast smoke test. Full grid ≈ 2m50s for all 14 models.
- [x] Evaluate on the test split: MAE per stat, and a ranking backtest
  (would ordering players by predicted stat have ranked the actual leaders better
  than a naive baseline?)
  - **All 14/14 position-stats beat BOTH baselines on MAE.** Ranking (Spearman)
    is strong on the headline stats: RB rushing 0.77, WR rec 0.76/0.71,
    TE rec 0.69/0.66, QB passing 0.60. TD/INT counts (sparse, near-zero) have
    NaN Spearman — expected, low-signal.
- [x] Save trained models to `models/` (joblib) with a version/metadata file
  - `models/models.joblib` (all 14 models + p10/p90 bands + meta) and
    `models/eval_report.json` (per-stat metrics, best params, baselines).
  - Verified: reload from disk → predictions identical (max diff 0.0), MAE 52.78
    matches the report.

## Phase 5 — Prediction output & CLI

- [x] Write `src/predict.py`: input = week/season + player list (or "all"), output =
  formatted lines like "Patrick Mahomes will get 2.1 touchdowns and 268 yards."
  - map model outputs back to the human phrasing per stat component
    → per-position phrasing (`PHRASE`): QB "X passing TDs … Y passing yards … Z
    interceptions", RB/WR/TE "X rushing/receiving yards, … TDs, … receptions".
  - include a confidence note (quantile spread) when available
    → headline stats show `(p10-p90)` band, e.g. "236 passing yards (136-323)".
- [x] Add a simple CLI entry point (`python -m src.predict --week 5 --season 2026`)
  - `python -m src.predict --season 2024 --week 5 --players "Patrick Mahomes"`
  - `predict --season 2024 --week 5 --all --top 15` (ranked board, top N per position,
    ranked by the model's **predicted** headline stat)
  - `predict --season 2024 --week 5 --all --compare --json`
  - also registered as the `predict` console script in `pyproject.toml`.
- [x] Handle upcoming-week inputs: fetch current rosters + Vegas lines for the week,
  build features from latest available data
  - `--refresh` re-pulls the current season (and the prior, for trailing context)
    via `src.download_data` then rebuilds features via `src.features.build_features`.
    Bounded test: 2025 re-pull + feature rebuild ≈ 2.5s, features stay intact.
- [x] End-to-end test: run prediction for a past week and compare against actuals
  - `--compare` prints ACTUAL + ERROR per stat and a headline-stat MAE/RMSE
    summary. Verified weeks 5 & 10 (2024): top-10 board headline MAE ≈ 28–35,
    consistent with the Phase 4 test-split MAE.

## Phase 6 — Polish & documentation

- [x] README: setup, data pull, training, prediction usage
  - Full rewrite with verified numbers (14/14 beat baseline, MAE table), CLI
    examples, `uv` setup instructions, and tests section.
- [x] Notebook with exploratory analysis + model evaluation charts
  - `notebooks/explore_model.py` → `notebooks/charts/`: test MAE vs baselines
    (grouped bars), predicted-vs-actual scatter (QB passing_yards, WR
    receiving_yards), and per-target ranking quality (Spearman).
- [x] Record known limitations (injury uncertainty, rookies, weather, OL changes)
  - README "Known limitations": injuries, rookies (rookie path under dev),
    conservative TD projections, shrinkage in receiving yards, OL changes,
    Vegas-as-input.

## Notes / gotchas

- Lookahead leakage is the #1 way this kind of model looks great offline and fails
  in real use. Everything in features must be computable BEFORE the week being predicted.
- Always compare against the naive baseline; a model that can't beat "last 3 games
  average" adds noise, not signal.
- Keep raw data immutable; all transformations go through scripts into `data/processed`.
