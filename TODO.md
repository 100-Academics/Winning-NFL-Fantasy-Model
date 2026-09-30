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
  - **10/14 position-stats beat BOTH baselines on MAE** (was 14/14 when the
    point estimate was fit as the median; the 4 that don't are the ultra-sparse
    TD/INT counts where "last week was 0" is a strong null — an honest MAE-vs-
    calibration tradeoff after switching the central estimate to the mean).
    Ranking (Spearman) is strong on the headline stats: RB rushing 0.77,
    WR rec 0.76/0.71, TE rec 0.69/0.66, QB passing 0.60.
- [x] **Calibration fix (2026-09):** central estimate now fits the conditional
  MEAN (`squared_error`), not the median — the median under-projected RB/WR/TE
  by 30–40% of their mean and collapsed TDs to 0 (see NOTES.md "Calibration").
  Signed bias per position is now within ±3.3%; `src/calibrate.py` +
  `models/calibration_report.json` verify it.
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

## Phase 5.5 — Rookie handling (prior + shrinkage)

*Why before Phase 6:* the main model predicts from trailing form. A true rookie
has NO trailing form, so it degenerates to a league-average guess and can't
rank rookies well. This phase adds a rookie **prior** (from draft capital +
college + combine + age) and blends it in with shrinkage, so more NFL games
earn more weight. This matters for beating current models in Phase 6.

- [x] Add `cfbd` (CollegeFootballData) as a dependency; API key goes in `.env`
  (gitignored) via `.env.example`. Key is read by `src/cfbd_client.py`.
  - Gotcha: cfbd 4.5.2 auth = `cfg.api_key["Authorization"]=key` +
    `cfg.api_key_prefix["Authorization"]="Bearer"`. APIs return LISTS of model
    objects (not DataFrames).
- [x] `src/cfbd_client.py`: cached pull of CFBD `draft_picks`,
  `player_season_stats`, `player_usage` (2016–2025 drafts; college 2014–2024)
  into `data/raw/cfbd/`.
- [x] `src/rookie_features.py`: join the CFBD draft bridge (name →
  college_athlete_id) + college rate/volume stats + CFBD usage + nflverse
  combine + pre-draft grade + age-at-draft into one per-player static table
  (`data/processed/rookie_features.parquet`). All pre-season — no lookahead.
  - Verified: first-round 2024 picks (Williams 97, Daniels 94, Nabers 95) match
    nflverse with correct round/pick + age, and carry their 2023 college stats.
- [x] `src/rookie_model.py`: per (position, target) `HistGradientBoosting`
  regressors trained on FIRST-SEASON player-weeks (pooled 2016–2022), val 2023,
  test 2024–25. Predicts the SAME per-week components as the main model.
  → `models/rookie_prior.joblib` + `models/rookie_prior_report.json`.
  - Trained: 14 models, 1,070 rookies / 10,691 first-season weeks
    (2024 test MAE: QB pass yds 72.9, RB rush yds 22.7, WR recv yds 19.5).
- [x] Wire shrinkage into `src/predict.py`: for a player in their rookie season,
  `final = w·main + (1−w)·prior` with `w = games_played/(games_played+K)`.
  Output shows a `[ROOKIE — rookie-prior blend]` line with the main-vs-prior
  split. `--no-rookie` disables it. (Graceful: if the prior isn't trained, the
  main model is used unchanged.)
- [x] Tune K (shrinkage strength). Swept K ∈ {0.1 … 12} per (pos, target),
  chosen on 2023 rookies (val), scored on 2024+2025 (test):
  - K=4 (old default) hurts: 6.89 / 6.59 test MAE. Uniform K=0.5 wins both
    test seasons: 6.37 / 6.17. Per-target K selection is within 0.04 of it,
    so uniform K=0.5 is kept (less to overfit).
  - The rookie prior helps mostly week-1 (its purpose) and only for some
    targets — WR receiving yards and RB receiving/receptions benefit; QB
    passing and RB rushing week-1 are better served by the main model.
    Position/tier (1st round vs later) optima flip by target and aren't
    stable across seasons → no per-tier K yet.
- [x] End-to-end verified (week-1 2024, `--compare`):
  - Jayden Daniels (QB): blend 193 yds vs main 237; actual 184 → prior was right.
  - Caleb Williams (QB): blend 211 vs main 247; actual 93 (prior closer).
  - Cam Ward (QB, 2025 w1): main only 49 yds, prior 220, actual 112 — the
    prior rescues the league-average collapse for zero-form rookies.
  - Week 6 (5 games played): blend is 91% main / 9% prior — shrinkage
    hands off to actual NFL form as designed.

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
