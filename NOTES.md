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
  `sklearn.ensemble.HistGradientBoostingRegressor` — fast (~0.7s/fit), no extra
  deps. Split train 2016–22 / val 2023 / test 2024–25.
  - **Central estimate = conditional MEAN (`loss="squared_error"`); bands =
    quantile (p10/p90).** The original build fit `loss="quantile", quantile=0.5`
    (the median) for the point estimate. For right-skewed, zero-inflated stats
    the median < the mean, so the model systematically UNDER-projected
    (measured 2024–25: RB −40%, TE −33%, WR −30% of mean; RB/WR/TE TDs = 0 for
    100% of rows). Fitting the mean fixes this (per-position bias now +0…+3%,
    better than the last-game/trailing-3 baselines) while preserving Spearman
    (0.60–0.77) and slightly improving RMSE. This is principled, not a patch:
    fantasy points are a LINEAR combo of stats, so E[points]=Σ score·E[stat]
    only holds with the mean; a median model breaks that linearity and distorts
    cross-position (flex) ranking even when per-stat MAE looks fine.
  - **Cost:** 4 ultra-sparse count stats (receiving TDs, QB INTs) no longer
    beat the "last week was 0" null on raw MAE (was 14/14, now 10/14). That's
    the honest MAE-vs-calibration tradeoff — we optimize the estimate, not MAE.
  - Saved to `models/models.joblib` + `models/eval_report.json`.
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
  - **Re-run after the mean-fit fix (2026-09-30, same 2024 w1–12 board):**
    * Raw components: aggregate MAE **10.81 vs 11.73** (was 10.86) — still
      better; RMSE **24.1 vs 25.6** (ours now better; was worse).
    * Weekly 1-PPR points, per position (MAE / RMSE): **QB 9.21/11.7 vs
      9.73/12.3; RB 6.23/7.8 vs 6.60/8.1; WR 5.81/8.1 vs 6.36/8.3; TE
      4.74/5.9 vs 5.93/7.2** — we now beat FP on **both** MAE and RMSE at
      every position (previously FP had the RMSE edge). The old RMSE gap was
      the median-model under-projection leaking into the squared error;
      fitting the mean closed it.
    * Caveat: this is the *same* 2024 test weeks we diagnosed the bias on, so
      the mean-fit change is validated on in-sample-for-diagnosis data — not a
      fresh holdout (see the "holdout status" note below).
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
  - **Root cause: the point estimate was fit as the MEDIAN**
    (`loss="quantile", quantile=0.5` ≡ `absolute_error`). For right-skewed,
    zero-inflated stats the median sits below the mean, so it *must*
    under-project — the effect scales with zero-inflation (TDs 80–94% zeros →
    −100%, predicted as 0.0 for ~all rows; RB receiving 41% zeros → −37%;
    QB passing 7% zeros → ~0%). (README already flagged "conservative TD
    projections" — this is why.)
  - **Top-of-board (flex stars) was under-projected too:** top-5% RB actual
    was **1.6×** predicted, WR **1.36×**, TE **1.34×**; the top QB slightly
    OVER (0.93×). So the global top tier looked fine (top-decile ratio 1.01)
    only because the RB/WR/TE under-estimate and the top-QB over-estimate
    *canceled* — that cancellation is what made the aggregate MAE acceptable.
  - **Flex distortion:** RB/WR/TE carried a −10.6-pt spread of relative bias,
    over/under-projected by up to 40% of their mean — a flex pick between them
    was systematically biased.
- **FIX (done, this branch):** changed the central-estimate loss from the
  median to the **conditional mean** (`loss="squared_error"`) in BOTH
  `src/model.py` and `src.rookie_model.py`; p10/p90 bands stay quantile.
  This is the principled fix (not an isotonic/Platt afterthought or a hand-
  fit bias offset): fantasy value is a LINEAR combo of stats, so
  `E[points] = Σ scoreᵢ·E[statᵢ]` only holds with the mean — a median model
  breaks that linearity (median of a sum ≠ sum of medians) and distorts
  cross-position/flex ranking even when per-stat MAE is fine. After the fix:
  per-position signed bias **QB +2.7% / RB +3.3% / WR +1.5% / TE −1.8%**
  (all within ±3.3%, better than both naive baselines); TDs no longer collapse
  to 0 (rookie QBs ~1 TD, TEs ~0.1–0.4); top decile genuinely calibrated
  (ratio 0.99, not a cancellation); Spearman preserved (0.77→0.769) and RMSE
  slightly better (6.69→6.16). Honest cost: 4 ultra-sparse count stats
  (receiving TDs, QB INTs) no longer beat the "last week was 0" null on raw
  MAE (14/14 → 10/14) — we optimize the calibrated estimate, not MAE.
  Verify any change with `uv run python -m src.calibrate`: success =
  per-position rel-bias ≈ 0 AND top-5% flex ratio ≈ 1.0, Spearman preserved.

## Per-position verification of the mean-fit fix (2026-09-30)

`notebooks/diag_post_mean_fix.py` re-runs the diagnostics on the CURRENT
(mean-fit) model, broken out **per position and per season** (the pooled
number can hide season-specific drift), plus band coverage and negative
predictions. Output → `models/diag_post_mean_fix.json`. Findings:

- **Bias by season (PPR, per position):** no season-specific blowup.
  - QB +1.4% (2024) / +4.0% (2025); RB +1.8% / +4.8%; WR **−1.4% (2024) /
    +4.5% (2025)**; TE −0.4% / −3.0%. The WR 2024→2025 sign flip is the one
    to watch — it's small (≈0.1–0.3 pts) but the only position whose direction
    changed. TE stays slightly under both years (the one flex position that
    still reads marginally conservative).
- **Per-stat (pooled 2024-25):** the residual over-projections concentrate in
  RB/TE *receiving* (RB rec_yds +9.6%, RB receptions +6.6%, TE receptions
  −6.2%) and QB passing yards +4.2% — all modest. The TD components are
  unbiased (within ±6%) and no longer collapse to zero.
- **Deciles (PPR, per position, all 10):**
  - **Top decile (flex stars) is well calibrated** at every position —
    ratios 0.98 (QB) / 1.04 (RB) / 1.02 (WR) / 1.02 (TE). Not a cancellation.
  - **Bottom decile (waiver/streaming) is OVER-projected:** actual/pred
    **RB 0.51, WR 0.54** (the model over-estimates low-usage RB/WR by ~2×),
    TE 0.82, QB 0.99. This is the mean-objective-on-zero-inflated effect the
    brief predicted — it's real but small in absolute terms (RB bottom-decile
    actual mean ≈ 1.1 pts, so the over-estimate is ≈ 1.0 pts). For start/sit
    it means a bottom-board RB/WR is more likely to be *worse* than
    projected; don't lean on the low end of the band as a floor.
  - Middle deciles (D3–D6) are essentially flat (residuals within ±0.5).
- **p10/p90 band coverage (headline stats, per position):** nominal 80%.
  - **Yards stats are UNDER-covered** — the band is too narrow: QB pass_yds
    74%, RB rush_yds 86%, TE rec_yds 85% (the rest 85–92%). Under-coverage
    means "the actual fell outside the band" more often than 20% — the band
    is **over-confident** on yards. If used for start/sit, treat the yards
    band as optimistic, especially QB.
  - **TD/reception stats are OVER-covered** (89–94%) — the band is too wide
    there (correct for zero-inflated, but wider than nominal).
  - **The mean sits sensibly inside the band** in 40–52% of the width for
    most stats (no systematic inversion: `p10>mean` and `p90<mean` are 0%
    everywhere except QB pass_tds 0.5% / RB rush_tds 16.5% / WR rec_tds
    47.6% — the last two are the sparse-TD stats where the quantile band
    degenerates to a near-zero width on most rows; see below).
  - **Bottom line for risk-based start/sit:** the *central* estimate is
    well-calibrated per position (good), but the **yards bands are too
    tight** (under-cover) and the **TD bands are too loose / degenerate** —
    don't read the band edges as a literal 80% interval for those.
- **Negative count predictions (clipping):** the squared-error mean objective
  emits small negatives on ~0.5% of rows for the sparse TD/INT stats
  (worst: TE rec_tds 12/2510, min −0.13; QB pass_int 6/1328, min −0.30;
  QB pass_tds 6/1328, min −0.06). Clipping at zero changes stat-MAE by
  <0.001 — negligible. **Done at the output surface** (`src/predict.py`
  `_clip_pred`): count stats and band bounds are clipped at 0 in the
  prediction CLI (and after the rookie-prior blend, which can re-introduce a
  tiny negative). The stored model is untouched, so `bench`/`calibrate`
  diagnostics and the band models are unchanged. A Poisson/Tweedie objective
  is the principled next step if negatives or the loose TD bands become a
  problem (it keeps the mean property while constraining to ≥ 0), but it's
  not needed for correctness today.
- **Holdout status:** the 2024–25 test split is **no longer a clean holdout**
  — we diagnosed the median-bias on it, so the mean-fit fix is validated on
  data the model-selection process saw. Treat 2024–25 numbers as
  *in-sample-for-diagnosis*, not untouched. The genuinely clean holdout is
  **this season's (2026) live weeks, logged before kickoff.** Score those
  with `src.predict --compare` as the season progresses and keep that log —
  that's the only number that tells us the mean-fit change generalizes.

## Live 2026 holdout — bugs found & fixes (2026-09-30)

Week-4 "live board" looked systematically low (top QB 11.9, RB-heavy, no
elite stars). Root cause + fix, plus the A/Bs we ran so we don't redo them:

### FIXED — NextGen (NGS) features were null in the upcoming-week path
The live board is built by `src/holdout._build_scheduled_panel` (weeks not yet
in `features.parquet`). NGS trailing was built from the NGS file, which only
has rows through the last *played* week, so the unplayed target week had no NGS
row to join → every `ngs_*` feature nulled to 0. NGS is the dominant QB/WR
efficiency signal, so this crushed elite players (Mahomes passing yds 272 → 44;
the whole board collapsed). **Fix:** in `_ngs()`, add a dummy NGS row at the
target week per player (reusing `_add_dummy`), so `ngs_*_tr3` = mean of the
played weeks — exactly what the played-week path produces. Committed `63a7830`.
Verified: NGS went 454/454-null → populated for every player with NGS data.

### FIXED — QB rushing was not modeled at all
`POS_TARGETS["QB"]` was passing-only, so Allen/Lamar/Hurts rush TDs never
counted (Allen's 6 rush TDs in w1-3 = ~47 fantasy pts, unmodeled). **Fix:**
added `rushing_yards` + `rushing_tds` to `POS_TARGETS["QB"]`, retrained
(`src/model.py`). QB rushing_yards model **beats baseline** (test MAE 11.22 vs
13.37, ρ +0.576); QB rushing_tds sits at baseline (rare event — expected).
Allen → #4, Lamar → #3 in the ESPN-standard board.

### NOT applied — prior-season shrinkage (the "pocket passers too high" fix)
Hypothesis: QBs rank too high off 3 games of 2026; blend toward prior-season
per-game rate (w = games/(games+K)). **A/B on the 2025 test holdout,
early-season rows (games_played_s2d ≤ 4) — it HURTS, not helps:**

| pos | raw ρ | @K=1 | @K=3 | @K=5 |
|-----|-------|------|------|------|
| QB  | +0.702| +0.638| +0.551| +0.504|
| RB  | +0.725| +0.735| +0.712| +0.699|
| WR  | +0.640| +0.655| +0.634| +0.618|
| TE  | +0.572| +0.575| +0.538| +0.516|

Why: Shough is QB1 because he is the **top-volume 2026 passer** (44 att/g,
306 yd/g vs Mahomes 33/271). The model is reading a real usage signal; pulling
him toward his 2025 *backup* year regresses QBs and only marginally helps
RB/WR. **Decision: NOT applied.** Repro: `notebooks/_q1_ab.py` (has a K sweep).
If a user still wants a "veteran prior," this is the mechanism to toggle.

### NOT applied — true opponent-defense features
Hypothesis: add real defense (yards/sacks/TDs allowed, trailing-3) as features
to the player models. **A/B on the 2025 test holdout (retrain each
position×target with the 3 new features): basically the same.** Every delta is
noise-level — MAE ≤ 0.06, ρ ≤ 0.02. Tally 4 keep / 2 drop / 10 same, and the
"keeps" (QB passing_yards −0.12 MAE, QB passing_int +0.011 ρ) are offset by the
"drops" (QB rushing_yards, WR receiving_yards). No position improves
meaningfully. **Decision: NOT added** — opponent defense is a weak predictor of
a *specific* player's fantasy output (known). Repro: `notebooks/_def_exp.py`.
Note: the model *already* has the proxy `total_yards_*_opp` + `rest_*_opp`
features, so this was a marginal improvement, not a gap.

### Scoring
The user's league is **ESPN Standard (non-PPR)**: passing 0.04/yd, 4/pass-TD,
−2/INT; rushing/receiving 0.10/yd, 6/TD; **no point per reception.** The
model predicts raw components, so scoring is a pure conversion at the output
surface — keep weights in one place. (The 1-PPR `SCORE` tables in
`src/bench`/`src/holdout`/`src/calibrate` are for the baseline comparisons, not
the user's league.)

### Top-defense (D/ST) projection — new, week 4
`notebooks/_topdef.py`: for each week-4 team, project points-allowed (½ own
trailing allowed + ½ opponent trailing offense) + turnovers (own trailing D TO
+ opponent projected INTs from the saved QB model), scored on ESPN D/ST tiers
(+2/TO). Week-4 top 5: SEA, ARI, SF, PIT, NO. **Caveat:** the 2026 season in
this dataset is high-scoring (many 50–65 pt games), so points-allowed tiers run
low and turnover volume is the main differentiator — treat absolute D/ST values
as season-relative. No specific D players projected (per user: defense as a
whole).

## Reproduction

- `uv run python -m src.bench` — our model vs naive baselines in PPR space
  (2024–25 full roster). → `models/bench_report.json`.
- `uv run python -m src.calibrate` — signed bias per position + projected-vs-
  actual by predicted decile + per-stat bias + flex distortion (same split).
  → `models/calibration_report.json` + `notebooks/charts/calibration_*.png`.
- `uv run python notebooks/diag_post_mean_fix.py` — per-position, per-season
  bias + all deciles + p10/p90 band coverage + negative count predictions
  (the post-mean-fix verification). → `models/diag_post_mean_fix.json`.
- `uv run python -m src.fetch_fantasypros` then `uv run python -m src.compare_fantasypros`
  — head-to-head vs FantasyPros consensus. → `models/fantasypros_report.json`.
