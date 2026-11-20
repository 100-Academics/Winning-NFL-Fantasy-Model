<div align="center">

# 🏈 Winning-NFL-Fantasy-Model

**Project every NFL player's stat line — then rank the whole field — before kickoff.**

*ESPN Standard (non-PPR) · nflverse 2016–2025 · clean holdouts, no leakage*

</div>

<br>

<div align="center">

**The product — a full-roster weekly board** · Week 11, 2026 (top of the board)

![Week 11, 2026 board — hero](docs/week11_board_hero.png)

[View the full board — all positions + the 26-team D/ST →](docs/RESULTS.md)

</div>

<br>

## 📊 At a glance

| | **2025 · clean holdout** | **2026 · live, pre-kickoff** |
|:--|:--|:--|
| **Beats both naive baselines** | **12 / 16** stat models | **7 / 9** weeks |
| **QB passing yards MAE** | **53.8** (vs 81.0 / 75.5) | — |
| **Ranking (Spearman ρ)** | positive on all 16 | **0.67 – 0.76** |

> [!TIP]
> 2025 was **never used in training**, and 2026 predictions are written to disk
> **before kickoff** and scored after the games — never re-fit on a week you're
> about to score. That's the only test that counts.

---

## What it does

Projects the raw **stat components** a player is likely to post in a week —
passing / rushing / receiving yards & TDs, receptions, targets, carries — with
**p10–p90 uncertainty bands**. Trained on nflverse (2016–2025) plus per-game
context: opponent, Vegas line, weather, injuries, usage.

- 📈 **Results** — full accuracy, charts & the *why* → [RESULTS.md](docs/RESULTS.md)
- 📓 **Working notes** — decisions + A/B log → [NOTES.md](NOTES.md)
- 🤝 **Agents** — team coordination → [AGENTS.md](AGENTS.md)

## ⚙️ Setup

Python **3.11+** and [uv](https://docs.astral.sh/uv/) — no pip/conda by hand.

```bash
uv sync --python 3.11
uv run python -m src.download_data --datasets all   # mirror nflverse -> data/raw/
```

## 🧪 Pipeline

```bash
uv run python -m src.clean     # -> panel.parquet  (REG-only, deduped)
uv run python -m src.features  # -> features.parquet  (leak-free)
uv run python -m src.model     # trains + evaluates -> models/
```

The split is **chronological** — train 2016–2023 · val 2024 · test 2025 · 2026 is the live holdout (`src.holdout`).

## 🚀 Predict

```bash
# one player
uv run python -m src.predict --season 2024 --week 5 --players "Patrick Mahomes"

# full ranked board (top 15 per position)
uv run python -m src.predict --season 2024 --week 5 --all

# compare vs actuals — end-to-end sanity check
uv run python -m src.predict --season 2024 --week 5 --all --compare --top 5

# upcoming week: refresh data + features first
uv run python -m src.predict --season 2026 --week 6 --all --refresh
```

```
Patrick Mahomes (QB, KC vs NO) — 1.0 passing TDs (0.0–2.8), 236 passing yards (136–323)
```

## ✅ Tests

```bash
uv run pytest tests/
```

`test_no_lookahead.py` proves the trailing features use only strictly-prior weeks — non-vacuous (remove the `shift(1)` and it fails).

## ⚠️ Limitations

Injury uncertainty · rookies (CollegeFootballData path in progress) · TD projections are directional (fit the mean, lean on p10–p90) · receiving yards shrink to the position mean · mid-season OL changes not modeled · Vegas lines are inputs, not targets. → [full detail in NOTES.md](NOTES.md)
