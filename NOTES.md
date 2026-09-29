# Working notes

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
- Features that move the needle: **Vegas lines** (spread/total → implied
  team totals; use as inputs, don't try to beat them), **weather**
  (Open-Meteo, free historical), **injury reports/inactives**, snap counts,
  route participation, **target share**, **red-zone usage**.
- Pro Football Reference: supplementary historical + college stats;
  rate-limits scraping aggressively — use sparingly.
- **Lookahead leakage is the #1 risk**: every feature must be computable
  strictly before the week being predicted.
