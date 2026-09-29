# TODO: NFL Fantasy Prediction Model

Goal: predict per-player fantasy performance, e.g.
"[Player xyz] will get 4.5 touchdowns and 202 yards."

## Phase 1 — Project setup

- [ ] Set up Python environment (venv or conda), pin Python 3.11+
- [ ] Install core libs: pandas, numpy, scikit-learn, matplotlib, jupyter
- [ ] Create directory layout: `data/raw`, `data/processed`, `src`, `notebooks`, `models`
- [ ] Add `.gitignore` (exclude `data/raw`, `models/`, venv dirs)
- [ ] Add `requirements.txt` or `pyproject.toml` and keep it updated

## Phase 2 — Historical data acquisition

- [ ] Choose a data source. Candidates:
  - nfl_data_py (free, wraps nflfastR play-by-play + weekly rosters/stats) — recommended starting point
  - nflfastR raw data feeds (downloadable season play-by-play CSVs)
  - Sleeper/FantasyData/SportsDataIO APIs (fantasy points, some paid)
- [ ] Install and pull historical weekly player stats (target: 2016–2025 seasons, or earlier)
  - columns needed: player, team, position, week, season, passing_yards, passing_tds,
    rushing_yards, rushing_tds, receiving_yards, receiving_tds, receptions, targets,
    carries, fantasy_points (PPR + standard)
- [ ] Pull supporting context data:
  - game schedules/results (opponent, home/away, Vegas spread/total)
  - player usage: snap counts, target share, red-zone touches
  - injuries/roster status changes (nflfastR has some; may need supplements)
- [ ] Write a `src/download_data.py` script so the pull is reproducible
- [ ] Save raw pulls to `data/raw/<season>_weekly.parquet` (parquet, not CSV — much smaller/faster)

## Phase 3 — Data cleaning & feature engineering

- [ ] Clean and validate: dedupe player names across seasons, handle team changes,
  fill/handle injury weeks, drop irrelevant positions or keep for flex decisions
- [ ] Define the prediction target per player-week:
  - fantasy_points (primary), plus component targets: passing yards/tds, rushing
    yards/tds, receiving yards/tds — the output format needs each component
- [ ] Build rolling features (avoid lookahead leakage! only use data from prior weeks):
  - trailing 3-game and 5-game averages of usage and production
  - season-to-date totals and per-game rates
  - opponent defense strength vs position (rolling yards/tds/fantasy allowed)
  - team implied total from Vegas line, home/away flag
  - target share / air-yard share trends, red-zone opportunity rate
- [ ] Write `src/features.py` with a `build_features(weekly_df) -> DataFrame` function
- [ ] Sanity-check: no feature uses same-week or future information (add a test)

## Phase 4 — Model

- [ ] Baseline first: predict trailing 3-game average (or last game) as the naive forecast.
  Measure MAE/RMSE per stat — the model must beat this to be worth anything.
- [ ] Split data chronologically: train on earlier seasons, validate on a held-out
  mid-season, test on the most recent season(s). NEVER random splits (time leakage).
- [ ] Train candidate models per target stat (or one multi-output model):
  - Gradient boosting: XGBoost / LightGBM / sklearn HistGradientBoosting
  - Quantile regression (gradient boosting with quantile loss) for "4.5 tds"-style
    medians — gives a defensible central estimate + uncertainty bands
- [ ] Tune hyperparameters on the validation split (small grid search is fine)
- [ ] Evaluate on the test split: MAE per stat, and backtest as fantasy ranking
  (would picking starters by predicted points have beaten baselines?)
- [ ] Save trained models to `models/` (joblib) with a version/metadata file

## Phase 5 — Prediction output & CLI

- [ ] Write `src/predict.py`: input = week/season + player list (or "all"), output =
  formatted lines like "Patrick Mahomes will get 2.1 touchdowns and 268 yards."
  - map model outputs back to the human phrasing per stat component
  - include a confidence note (quantile spread) when available
- [ ] Add a simple CLI entry point (`python -m src.predict --week 5 --season 2026`)
- [ ] Handle upcoming-week inputs: fetch current rosters + Vegas lines for the week,
  build features from latest available data
- [ ] End-to-end test: run prediction for a past week and compare against actuals

## Phase 6 — Polish & documentation

- [ ] README: setup, data pull, training, prediction usage
- [ ] Notebook with exploratory analysis + model evaluation charts
- [ ] Record known limitations (injury uncertainty, rookies, weather, OL changes)

## Notes / gotchas

- Lookahead leakage is the #1 way this kind of model looks great offline and fails
  in real use. Everything in features must be computable BEFORE the week being predicted.
- Always compare against the naive baseline; a model that can't beat "last 3 games
  average" adds noise, not signal.
- Keep raw data immutable; all transformations go through scripts into `data/processed`.
