# Winning-NFL-Fantasy-Model

A prediction model for **NFL player performance**: it estimates the **stat
components** a player is likely to post in a given week — e.g.
"Patrick Mahomes will get ~2.1 touchdowns and ~268 passing yards."

We predict the raw stats (passing/rushing/receiving yards & TDs, receptions,
targets, carries, …). **Fantasy points are a separate, later project** that
converts these components — this repo does not score them.

## What it does

- Consumes historical NFL data (nflverse, 2016–2025) plus per-game context
  (opponent, Vegas line, weather, injuries, usage).
- Produces per-player per-week projections of individual stat components,
  with uncertainty bands.

## Status

- **Phase 1 (setup)** and **Phase 2 (data acquisition)** are done — the raw
  data is mirrored to `data/raw/` and a reproducible downloader lives in
  `src/download_data.py`.
- Phases 3–6 (cleaning/features, modeling, prediction CLI, polish) are still
  to come. See `TODO.md` for the full plan.
