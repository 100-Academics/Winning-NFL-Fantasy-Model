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

### Top-defense (D/ST) projection — REBUILT, week 4 (2026-09-30)
The first D/ST board (`_topdef.py`, week-4 top 5 SEA/ARI/SF/PIT/NO) was BAD and
has been replaced. Diagnosis + fix, A/B'd so we don't redo it:

- **The old board was missing sacks, missed D/ST TDs, and over-projected
  turnovers** (2.1–3.4/game vs ~1.0–1.4 forced league-wide), so it was being
  carried by turnover *luck*. (Agent diagnosis — confirmed.)
- **A/B (2024+25 actual D/ST points, n≈570/yr; 2025 is the clean holdout per
  user):**
  - **Baseline — rank D/ST by the OPPONENT's Vegas implied team total, LOWEST
    first — is the best ranker: Spearman +0.349 (2024) / +0.339 (2025).**
    Opponent implied total is the dominant predictor of points-allowed
    (corr +0.413 to actual allowed; D/ST pts corr +0.345). Sacks (−0.174) and
    turnovers (−0.119) are weak, noisy, same-sign-correlated noise.
  - **My own-defense model (points-allowed + shrunk TO/sacks/DTD) does NOT
    beat it: a grid of 240 weights tops out at +0.336 Spearman** (vs baseline
    +0.345; `_def_grid.py`) — i.e. adding own-defense signal is at best a wash,
    at worst slightly worse. Adding a **prior-season defense-quality prior**
    also fails (2024 worse at every weight; 2025 a wash, `_def_blend.py`).
  - **Why the gut ("HOU/BAL/DEN are elite") disagrees with the board:** D/ST
    fantasy value is the **MATCHUP** (who they play), not the team. In 2025
    HOU is baseline-rank ~326 (worst) *because it plays top offenses* — the
    defense is still elite but its fantasy output is low. That's correct, not a
    bug.
- **Shipped method (`_topdef.py`):** rank by opponent Vegas implied total
  (lowest = best); ESPN Standard points-allowed tier on that projected
  allowed; + flat league-mean sacks (~2.36, +1) and hard-shrunk TO (~1.0, +2)
  for a realistic absolute value (these are near-constants, ranking-invariant).
  Tie-break within a tier: fewer projected points allowed = better.
  **Week-4 top 5: MIN (14.0), BAL (16.0), SEA (17.8), PIT (18.0), GB (18.0)** —
  all facing the weakest projected offenses. (Old board had HOU/ARI/NO high off
  turnover luck; they're now mid/bottom.)
- **Caveat (unchanged):** 2026 in this data is high-scoring, so points-allowed
  tiers run low — treat absolute D/ST values as season-relative. No specific D
  players projected (per user: defense as a whole).
- **Clean D/ST holdout (added 2026-09-30):** the D/ST board is now logged and
  scored the same disciplined way as the offense holdout, via
  `src/holdout_def.py` (append-only `models/holdout_log_def.jsonl` +
  `models/holdout_score_def.json`). Log BEFORE kickoff (deterministic from
  Vegas lines + flat league means — no model, nothing to leak), score AFTER the
  games (actuals from `schedules` final score → points allowed + `team_weekly_stats`
  TO/sacks/D-TD). W4 board already logged 2026-09-30T21:02.
  - When W4 finishes: `uv run python -m src.holdout_def --season 2026 --week 4 --score`.
  - Note: single weeks are noisy (W3 board ranked top-5 at actual ranks
    6/7/20/28/29, gap −1.51) — that's the weekly variance the agent flagged;
    the method is validated on 2024+25, not one week. Keep a per-week log and
    judge by the trend across weeks, not a single one.
- Repro: `_def_ab.py` (model vs baseline), `_def_grid.py` (weight sweep),
  `_def_blend.py` (prior-season prior), `_def_truth.py` (ground-truth corr).
  `uv run python notebooks/_topdef.py [season] [week]` for the board.

## Live 2026 holdout — W4 scored, W5 logged (2026-10-08)
- **Week 4 (scored 2026-10-08, both tracks clean — logged pre-kickoff 09-30,
  scored against pulled actuals):**
  - **Offense (n=338):** model **Spearman +0.786**, MAE 4.87 — beats trailing-3
    (+0.753, MAE 4.99) and last-game (+0.684, MAE 5.95) on ranking; lowest MAE
    of the three. **Trend w1→w4: +0.682 → +0.759 → +0.786 → +0.786** (peaked,
    holding — the mean-fit model is generalizing live, not just in-sample).
  - **D/ST (n=32):** Spearman **+0.378**; top-5 (by board) MIN/SEA/BAL/GB/PIT
    landed actual ranks **3/6/7/13/20**, top-5 mean 7.80 vs rest 5.11 (**gap
    +2.69**). A strong week — big jump from W3 (gap −1.51, ranks 6/7/20/28/29).
    The matchup baseline (rank by opponent's Vegas implied total) is paying off.
  - Score files: `models/holdout_score.json` (+`2026_w04`), `models/holdout_score_def.json`
    (+`2026_w04`).
- **Week 5 (logged pre-kickoff 2026-10-08T16:05 — the clean act; games this
  weekend Oct 10–12):** offense **442 players**, D/ST **30 teams** (2 byes that
  week). W5 boards in `notebooks/w5_board.html` + `models/w5_board.json`
  (offense top-10/pos in ESPN standard, full 30-team D/ST). Score after the
  games: `uv run python -m src.holdout --season 2026 --week 5 --score` and
  `uv run python -m src.holdout_def --season 2026 --week 5 --score`.
- **REFRESH PITFALL (the one that would have cost us the model):** the model
  is trained on **2016–2026**, but `src.predict --refresh` hard-codes
  `--seasons 2025 2026 --force` and `download_data` writes each dataset to ONE
  fixed filename — so a `--refresh` would have **overwritten the raw parquets
  with only 2025+2026, deleting 2016–2024 (the training history)**. To refresh
  for an upcoming week, pull the FULL range first:
  `uv run python -m src.download_data --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 --datasets core,context,advanced --force`,
  then `uv run python -m src.features`. (Do NOT run `predict --refresh` while
  the model depends on 2016–2024.)
- **`participation.parquet` refresh fails** on 2026 (`load_participation`:
  "Season must be between 2016 and 2025") — expected; it's not used by
  `features`/`holdout`/`predict`, and it fails before write so the prior file
  stays intact. Ignore that one `[FAIL]`.

## Live 2026 holdout — W5 scored, W7 logged (2026-10-22)
- **Week 5 (logged pre-kickoff 2026-10-08T16:05; games Oct 10–12; scored
  2026-10-22 after a full-range data refresh):**
  - **Offense (442 logged, n=320 scored):** model **Spearman +0.727**, MAE 5.43 —
    beats trailing-3 (+0.698, MAE 5.49) and last-game (+0.644, MAE 6.30) on
    ranking AND has the lowest MAE — 5 straight weeks ahead of both baselines.
    **Trend w1→w5: +0.682 → +0.759 → +0.786 → +0.786 → +0.727** (cooled from the
    w3–w4 plateau but still clearly ahead; single-week noise, no action).
  - **D/ST (n=30):** Spearman **+0.168** (noisy vs w4 +0.378 — expected weekly
    variance); **top-5 gap +3.96 = best of the season so far** (w3 −1.51,
    w4 +2.69, w5 +3.96). Board top-5 HOU/JAX/CIN/NYJ/WAS landed actual ranks
    **2/3/11/13/22**, top-5 mean actual 9.00 vs rest 5.04. HOU (3 pts allowed,
    4 sacks → 16.0) and NYJ (15.0, 8 sacks) carried it; WAS (allowed 27, rank
    22) was the miss. Method unchanged — judged on the gap trend.
  - Score files: `models/holdout_score.json` (+`2026_w05`),
    `models/holdout_score_def.json` (+`2026_w05`).
- **Week 6: SKIPPED** — no board was logged (offense or D/ST) and no W6 board
  exists, so W6 is not in the holdout record. Per user: evaluate the prior
  completed week (W5) instead.
- **Week 7 (logged pre-kickoff 2026-10-22T16:39 — the clean act; games Oct
  22–25):** offense **446 players**, D/ST **28 teams** (2 byes). W7 boards in
  `notebooks/w7_board.html` + `models/w7_board.json` (offense top-10/pos in
  ESPN standard, full 28-team D/ST). **W7 D/ST top-5: HOU (vs NYG, 16.0 imp),
  MIN (17.5), NYJ (17.5), CLE (18.0), TEN (19.5)**; sits: NYG (26.5 imp),
  GB (28.2 imp, vs DET). Score after the games:
  `uv run python -m src.holdout --season 2026 --week 7 --score` and
  `uv run python -m src.holdout_def --season 2026 --week 7 --score`.
- Data refresh for this cycle: full range 2016–2026 re-pulled (core/context/
  advanced, `--force`) + `src.features` rebuild → `features.parquet` 64,104
  rows (2026 now w1–w6 actuals). Only the expected `participation` [FAIL]
  (2026 unsupported by nflreadpy).

## Live 2026 holdout — W7 scored, W8 logged (2026-11-01)
- **Week 7 (logged pre-kickoff 2026-10-22T16:39; games Oct 22–26; scored
  2026-11-01 after a full-range data refresh):**
  - **Offense (446 logged, n=293 scored):** model **Spearman +0.688**, MAE 4.07 —
    beats trailing-3 (+0.651, MAE 3.75) and last-game (+0.566, MAE 4.42) on
    ranking; TR3 took the lowest MAE this week. **7 straight weeks ahead of both
    baselines on Spearman.** Trend w1→w7: +0.682 → +0.759 → +0.786 → +0.786 →
    +0.727 → +0.688 (w6 skipped) — cooled from the w3–w4 plateau but still
    clearly ahead; single-week noise, no action.
  - **D/ST (n=28):** Spearman **+0.132**; **top-5 gap −0.55** (w3 −1.51,
    w4 +2.69, w5 +3.96, w7 −0.55) — a down week: board #1 HOU allowed 27
    (actual rank 17), top-5 actual ranks 3/16/17/18/22; TB (board #18) went
    18.0 (3 TO, 5 sacks → actual rank 1). MIN (board #2 → actual 3) was the
    clean top-5 hit. Method unchanged — judged on the gap trend.
  - Score files: `models/holdout_score.json` (+`2026_w07`),
    `models/holdout_score_def.json` (+`2026_w07`).
- **Week 8 (logged pre-kickoff 2026-11-01T10:29 for the 13 weekend games;
  games Oct 29–Nov 2):** offense **454 players**, D/ST **28 teams** (no byes).
  W8 boards in `notebooks/w8_board.html` + `models/w8_board.json` (offense
  top-10/pos in ESPN standard, full 28-team D/ST). **W8 D/ST top-5: CIN (vs
  TEN, 17.0 imp, 8.4), WAS (vs PHI, 17.0, 8.4), NE (vs MIA, 17.2), PIT (vs
  CLE, 17.5), JAX (vs IND, 18.2)**; sits: ARI (30.8 imp, vs DAL — the
  slate's strongest projected offense), TEN (29.5, vs CIN), BAL (28.8, vs
  BUF). Score after the games:
  `uv run python -m src.holdout --season 2026 --week 8 --score` and
  `uv run python -m src.holdout_def --season 2026 --week 8 --score`.
- **W8 log caveat (NEW EDGE CASE — partial target week in features.parquet):**
  the W8 Thursday game (CAR @ GB, Oct 29) was already played when the log was
  requested. `src.holdout.predict_week` checks features.parquet for
  (season, week) rows FIRST and only falls back to the scheduled panel when
  there are ZERO such rows — after this cycle's refresh the played Thursday
  game's 25 rows existed, so the first `--log` run silently captured only
  those 23 players and skipped the full slate. Repair: (1) truncated the 23
  fresh entries (mine, ~2 min old; verified they were the exact tail),
  (2) backed up `features.parquet` (sha256), filtered out the 2026 w8 rows,
  re-ran `--log` → 454 players from the scheduled panel (features strictly
  weeks ≤7), (3) restored `features.parquet` byte-identical (sha256-verified).
  The CAR/GB entries are post-kickoff in time but leakage-free (the scheduled
  panel never sees target-week outcomes). **Rule for future late-request
  weeks: if any target-week game is already played (rows in features.parquet),
  `holdout --log` will silently skip the scheduled panel — always check the
  logged player count against the full slate (~450) before trusting a log.**
  (A proper fix — e.g. falling back to the scheduled panel when the week's
  played games are a proper subset of the scheduled games — is a code change
  to A/B, not a live-cycle patch.)
- Data refresh for this cycle: full range 2016–2026 re-pulled (core/context/
  advanced, `--force`) + `src.features` rebuild → `features.parquet` 64,460
  rows (2026 now w1–w7 actuals + the played w8 Thursday game). Only the
  expected `participation` [FAIL] (2026 unsupported by nflreadpy).

## Live 2026 holdout — W8–10 backfilled & scored, W11 logged (2026-11-17)
- **Context:** W9 and W10 were missed in the live loop (model used for other
  work over that stretch); W8 had been logged 2026-11-01 but never scored.
  All three backfilled 2026-11-17: full data refresh (`--force`; W9–W10
  actuals + W11 lines landed) + `src.features` rebuild (65,466 rows; 2026 now
  w1–w10 actuals) + replay `--log` with the **frozen V6 model** (features
  strictly prior-week; the log path never sees target-week outcomes), then
  scored. No model change — the standing decision rule was not invoked.
- **Offense (Spearman vs baselines, ESPN Standard):**
  - W8 (289 scored): model **+0.678** (MAE 4.28) vs tr3 +0.644 (3.86) / last +0.576 (4.34)
  - W9 (337): model **+0.718** (MAE 3.37) vs tr3 +0.631 (3.66) / last +0.556 (4.27)
  - W10 (299): model **+0.701** (MAE 3.46) vs tr3 +0.605 (4.08) / last +0.557 (4.66)
  - Model ahead on ranking all three weeks; W9 and W10 per-position: model
    wins all four positions (W8's one exception was QB, last-game's week:
    model +0.326 vs last +0.373).
  - Season trend: w1 +0.682 → w2 +0.759 → w3 +0.786 → w4 +0.786 → w5 +0.727 →
    (w6 skipped) → w7 +0.688 → w8 +0.678 → w9 +0.718 → w10 +0.701.
- **D/ST (top-5 gap vs rest):** w8 **+7.15** (ρ +0.226; top-5 actual ranks
  1/3/4/10/21 — board #1 CIN was actual rank 1), w9 **+6.72** (ρ +0.630;
  ranks 2/5/6/8/9), w10 **+3.91** (ρ +0.501; ranks 1/6/17/18 — board #1 LA
  vs ARI went 19.0 pts, actual rank 1). Three straight positive-gap weeks.
  W10: 2 teams unscored (incomplete games) — 28 logged, 26 scored. Gap
  trend: w3 −1.51 → w4 +2.69 → w5 +3.96 → w7 −0.55 → w8 +7.15 → w9 +6.72 →
  w10 +3.91.
- **Week 11 (logged pre-kickoff 2026-11-17T00:12; games Nov 19–22):**
  offense **470 players**, D/ST **26 teams** (13 games). Scheduled-panel path
  (no W11 rows in features.parquet yet — the fallback fired and the logged
  count was verified against the full slate). Boards:
  `notebooks/w11_board.html` + `models/w11_board.json`. **W11 D/ST top-5: KC
  (vs ARI, 12.5 imp, 11.4), BUF (vs MIA, 14.5), DEN (vs LV, 15.2), HOU (vs
  IND, 16.5), PIT (vs PHI, 17.0)**; sits: TEN (31.5 imp, vs DAL — the
  slate's strongest projected offense), MIA (31.0, vs BUF), NO (30.0, vs
  CHI), WAS (29.5, vs CIN), ARI (29.0, vs KC).
- **Score after the games:** `uv run python -m src.holdout --season 2026 --week 11 --score`
  and `uv run python -m src.holdout_def --season 2026 --week 11 --score`.
- Data refresh: 10/11 datasets updated; only the expected `participation`
  [FAIL] (nflreadpy season range caps at 2025) — the 2026-09-29 file stays.

## Third-party audit #3 — live-path correctness fixes + NaN A/B (2026-11-19)

A third-party LLM audited the model again (pasted as-is; focused on "two silent
bugs in the live prediction path" + a NaN-handling claim + feature upgrades +
external benchmarks). Every testable claim verified against the code; every
model-affecting change A/B'd on the clean 2025 holdout before adoption (standing
rule). This audit's highest-impact items were correctness bugs in the **live
(upcoming-week) path** — features that were silently `0.0` on the board.

### Claim verification (audit claim → verdict)

1. **`snap_pct_tr3/s2d` silently `0.0` on live upcoming weeks** (`holdout.py:238`
   passed raw `load_snap_counts`; `_attach_snap_shares` left-joins on
   (season,week,name) → null at the unplayed week → `fill_null(0.0)`) —
   **TRUE.** Same class of bug as the NGS W4 collapse (Mahomes 272→44). **Fixed.**
2. **`prev_season_*` bridge features silently `0.0` on live upcoming weeks**
   (`holdout.py:167` filtered the panel to the current season only, so
   `_prev_season_bridge` found no prior-season rows) — **TRUE.** **Fixed.**
3. **`.fill_null(0.0)` before HGBR** (dome→0°F, missing NGS→0% separation,
   week-1→0-yard game; HGBR handles NaN natively) — **TRUE as a representation
   choice.** A/B'd below: within noise on ρ → **not adopted** (kept zero-fill).
4. **`a is None: continue` at `holdout.py:420` drops logged players with no
   actual row** (an active-but-zero player scored as a non-game) — **TRUE.**
   **Fixed** (scored as actual=0.0, `n_zeroed` reported).
5. **`compare_fantasypros.py` hardcoded to 2024; QB `STAT_MAP` missing
   `rush_yds`/`rush_tds`** — **TRUE.** **Fixed** (season generalization + QB
   rushing added).
6. **OUR FIND (not in the audit): `snap_pct_*` trailing was computed over a
   SCRAMBLED physical row order.** `load_snap_counts` ends with
   `.sort(...).unique(subset=...)`, and in polars 1.44 `unique()` returns
   hash-ordered rows, so the `.over()` trailing windowed over arbitrary rows
   (not the prior weeks) — the exact "`.over()` follows PHYSICAL ROW ORDER"
   gotcha already noted in this file. Verified (Rodgers rows at physical idx
   379/606/664/1203). **Fixed** (re-sort on (name, season, week) before the
   trailing, same pattern as `_ngs_trailing`). This changes the played-week
   (training) features → A/B'd below.

### A/B (clean 2025 holdout; train 2016–23 / val 2024 / test 2025, shipped 24-config grid)

`notebooks/_ab_snapfix_nan.py` → `models/ab_snapfix_nan_report.json`. Three arms,
same test rows; cross-position all-null NGS cols dropped per-position (asserted
empty in test → a no-op for the zero-fill arms):

| arm | change | pooled MAE | pooled ρ | MAE wins vs V0 | verdict |
| --- | --- | --- | --- | --- | --- |
| V0 | old (scrambled) snap + zero-fill (= shipped) | 6.87 | 0.7347 | — | — |
| V1 | **corrected** snap order + zero-fill | 6.88 | 0.7338 | 6/16 | **applied** (correctness) |
| V2 | corrected snap order + **NaN** | 6.82 | 0.7383 | 12/16 | within noise (ρ) → not adopted |

(Per-stat win counts are run-to-run unstable — HGBR threading variance on
deltas this small flips individual stats across the zero line; the pooled
MAE/ρ are stable. Re-run gives V2 8–12/16 MAE wins, Δρ +0.002–0.004 either way.)

- **snap-order (V0→V1):** within noise on metrics (MAE −0.02, ρ −0.0025) but a
  genuine correctness bug (the trailing windowed over scrambled rows) → **adopted**
  on correctness; the retrain matches V1 almost exactly (QB pass_yds 53.8/ρ+0.617,
  RB rush 16.7/ρ+0.795, WR rec 18.1/ρ+0.716, TE rec 12.8/ρ+0.702; 12/16 beat
  baseline — unchanged).
- **NaN (V0→V2):** best arm (pooled MAE −0.08, ρ +0.002, 8/16 MAE wins) and more
  correct, but Δρ +0.002 is inside the ±0.01 noise bar → **not adopted** by
  default (kept the simpler zero-fill). Available if desired — would require the
  per-position all-null column drop in `src/model.py`.

### Shipped changes (all correctness; A/B-gated where model-affecting)

- `src/holdout.py` — `_build_scheduled_panel`: B1 (snap dummy row at the target
  week, mirroring the NGS W4 fix) + B2 (prior-season rows added so the bridge
  fires on the live path). Verified exact on the W12 panel: `snap_pct_tr3` 514
  non-zero, all 7 `prev_season_*` populated, Rodgers cross-checks exact
  (0.9233, 204.0).
- `src/features.py` — `_attach_snap_shares`: re-sort on (name, season, week)
  before the trailing (scrambled-order fix).
- `src/holdout.py` — `score_week`: logged players with no actual row now score
  as actual=0.0 (`n_zeroed` reported) instead of being silently dropped.
- `src/compare_fantasypros.py` — QB `STAT_MAP` + `rush_yds`/`rush_tds`; season
  generalization (`--season` / auto-detect) instead of hardcoded 2024.
- Model retrained (corrected snap + zero-fill): 2025 test **12/16 beat
  baseline** (unchanged from the shipped V6 model), ranking strong on the
  decision stats. W11's logged projections stand as logged (pre-audit model);
  this governs W12 onward.
- `features.parquet` rebuilt (snap-order fix changes played-week features).

### NOT applied (A/B within noise — do not re-test without a new signal)

- Keep-NaN before HGBR (audit claim 3): best arm but Δρ +0.002 < ±0.01 noise bar;
  more correct. If adopted, drop the per-position all-null NGS cols in training.

### Not yet built (audit's feature upgrades + external benchmarks — proposed follow-ups)

- Own-player injury status + QB-out flag; red-zone / inside-the-10 usage
  (nflverse PBP); position-specific defense allowed; two-stage volume→production
  + per-position feature pruning; refit on train+val + de-dup band fits.
- External benchmarks: Vegas player props, full-slate ESPN + FantasyPros logging,
  FantasyPros "Accuracy Gap" protocol, FFA 11-season MAE/R² reference.

## Third-party audit — claim verification + A/B + fixes (2026-11-17)

A third-party LLM audited the model (claims about current performance, four
improvement proposals, a 5-tier weekly evaluation suite). Every testable
claim verified against the artifacts; every proposal A/B'd on the clean 2025
holdout before adoption.

### Claim verification (third-party claim → verdict)

- **2024 H2H vs FantasyPros** (MAE/RMSE wins at all positions, 10-14 stat
  components, Spearman 0.655 vs 0.601): TRUE — `models/fantasypros_report.json`.
- **2026 live numbers** (R² 30.5–48.5%, WR +13.9% / TE +9.2% signed bias,
  top-5 ranks 9.7/13.4/24.5/16.1, MAE 5.70/3.89/3.44/2.77): EXACT match —
  `models/goodness_report.json`.
- **"Full-roster Spearman is padded by bench rows"**: CONFIRMED. Within-slice
  (starter pools: top-12 QB/TE, top-24 RB/WR) Spearman on the scored 2026
  weeks is +0.2 to +0.4 vs +0.67 to +0.76 full-roster
  (`notebooks/_tier1_lite.py`). The headline ρ overstates decision-tier
  discrimination; the model's edge is "whose #1 pick is better" (top-1
  actuals +1.2 to +10.6 pts over baselines).
- **"No pooled season-long 2026 scoreboard vs FantasyPros yet"**: TRUE —
  only W4/W5 one-off checks exist (`_h2h_w4.py`, `_fp_w5.py`).
- **TD-band degeneracy** (47.6% of WR `receiving_tds`, 16.5% of RB
  `rushing_tds` rows with p90 < mean): VERIFIED — `models/diag_post_mean_fix.json`;
  309 degenerate TD bands exist on the W11 panel alone.
- **`snap_counts.parquet` downloaded but never wired into features**: TRUE
  (grep of `src/features.py` was empty).
- **`predict.py --refresh` clobbers 2016-2024 history**: TRUE —
  `download()` overwrote the whole parquet with a 2-season pull.
- **Partial-week logging bug (W8 Thursday game)**: TRUE — `predict_week`
  only fell back to the scheduled panel when the week had zero rows in
  features.parquet; a played Thursday game left the rest of the slate unlogged.
- **OUR FIND, not in the audit: the prior-season bridge was never in the
  model.** `_prev_season_bridge`'s final `select` kept only
  (player_id, season) and dropped all 7 bridge columns — since the 2026-10-22
  commit that introduced it. Every "V6 with bridge" claim in NOTES
  overstates what actually shipped. Fixed; A/B'd below.

### A/B results

**Audit proposals** (`notebooks/_exp_audit_ab.py` → `models/ab_audit_report.json`;
train 2016-23, test 2025, shipped hyperparameters per model; the arm
comparison is internal to this script's procedure):

| arm | pooled MAE | pooled ρ | early ρ |
| --- | --- | --- | --- |
| V0 baseline | 3.885 | 0.5633 | 0.5569 |
| V1 + snap share | 3.882 | 0.5658 | 0.5594 |
| V2 + vacated opportunity | 3.874 | 0.5644 | 0.5573 |
| **V3 both (ADOPTED)** | **3.868** | **0.5674** | **0.5597** |

V3 best on both pooled metrics and early-season ρ; 14/16 stats non-negative
(worst regression −0.0005).

**Bridge** (bug fix, `notebooks/_exp_bridge_ab.py` + `models/ab_bridge_report2.json`;
shipped procedure: position-filtered splits, full 48-config grid on 2024 val):
A (no bridge) pooled ρ +0.6586 MAE 4.919, 15/16 beats-baseline;
B (bridge) ρ +0.6589 MAE 4.907, 15/16 — **within noise**. Fix kept (the
function now does what it documents) and the columns ship, but they earn
nothing on 2025.

### Shipped changes

- `src/features.py` — +2 snap-share features (`offense_pct` tr3/s2d from
  `snap_counts.parquet`) + 3 vacated-opportunity features (Out/Doubtful
  same-position teammates' trailing targets/carries + count; backward-asof so
  injured players without a week-W stats row still contribute). Both strictly
  prior. `load_snap_counts()` / `load_injury_reports()`; the scheduled-panel
  path passes both in as well (`src/holdout.py`), so upcoming weeks don't get
  null columns.
- `src/features.py` — `_prev_season_bridge` regression fix (select dropped
  all 7 bridge columns).
- `src/holdout.py` — partial-week guard: scheduled panel is now used whenever
  the target week is not FULLY played (every game has a final score), not
  only when features.parquet has zero rows. Regression test:
  `tests/test_partial_week_guard.py`.
- `src/download_data.py` — partial re-pulls now MERGE: only the pulled
  seasons are replaced, historical seasons are kept. `--refresh` no longer
  clobbers 2016-2024.
- `src/predict.py` — degenerate TD bands (p90 < mean) now display
  `P(TD>=1) = 1 − e^{−μ}` instead of a dead "(0.0-1.0)" interval
  (e.g. `0.0 receiving TDs (P(TD>=1) 4%)`).
- Model retrained (snap + vac + bridge columns present): 2025 test
  10/16 beats-baseline (MAE gate), ranking strong on the decision stats
  (QB pass yds ρ +0.610, RB rush yds +0.795, WR rec yds +0.714, TE rec yds
  +0.702). **W11's logged projections were recorded pre-kickoff with the
  pre-audit model and stand as logged; the new model governs W12 onward.**

### Not yet built (audit's 5-tier suite — proposed follow-ups)

- Tiers 1-4 as a permanent weekly battery (starter-pool ρ/NDCG, start/sit
  fringe pairwise win-rate, flex position-share calibration, top-10
  hit/bust rate + empirical band coverage). `notebooks/_tier1_lite.py` is
  the Tier-1 stub.
- Tier 5: auto-fetch the FantasyPros top-10 board on `--log`, cumulative
  2026 live scoreboard on `--score`.

## Is it GOOD? — goodness battery on live OOS weeks (2026-11-01)

The weekly holdout score only tests the model against two naive baselines
(last game, trailing-3) — a low bar, beaten 7 straight weeks. Per user: the
bar has to be "is it good", not "does it beat the baselines". New reproducible
battery `notebooks/_goodness_battery.py` (re-run any week; writes
`models/goodness_report.json`), run on the ONLY true out-of-sample data for
the shipped model (live 2026 scored weeks — the model was trained through
2025, so 2024–25 is in-sample for it). Covers w1–w5 + w7 scored, plus the w8
Thursday game (25 rows; logged post-kickoff for that game, pre-kickoff
features).

**1. Decision value (start/sit) — mean ACTUAL pts of each method's top pick (pooled):**
- QB: model top-1 **24.05** vs last-game 16.43 / tr3 14.22 (field 15.20) → +7.6/+9.8-pt edge.
- RB: **20.11** vs 9.72 / 9.55 (field 6.23) → +10.4/+10.6.
- WR: **12.44** vs 10.14 / 11.21 (field 4.35) → +2.3/+1.2.
- TE: **11.47** vs 6.71 / 7.30 (field 3.33) → +4.8/+4.2.
- Per-game top-1 win rates are coin-flip-ish (3W-2L-1T QB; 4W-2T-0L RB; 3W-3L-1T WR;
  3W-2L-1T TE, ties=0.5) — single-game variance dominates one pick; the edge is in the
  MEAN, which is what compounds over a season of lineup decisions.

**2. Flex value (RB/WR/TE pooled, top-10 by predicted → mean actual):**
model **15.71** vs tr3 13.06 / last-game 11.07 (field ~4.7; top-20: 13.49/12.10/10.06) —
the model's flex board is ~3.3× the field mean vs 2.3–2.8× for the baselines; model wins
5 of 6 weeks (w2 near-tie vs last-game, 11.72 vs 11.84). This is the direct live test of
the mean-fit fix (E[points]=Σ score·E[stat] only holds for the MEAN, not the median):
the flex separation is real.

**3. Noise floor (MAE vs zero-signal predictor = 2025 w15–17 position means):**
model MAE 1.20–1.46× better than the floor (QB 5.70/6.98, RB 3.89/5.70, WR 3.44/4.12,
TE 2.77/3.48); variance explained: QB 0.305, RB 0.485, WR 0.322, TE 0.283 — the model
captures 30–49% of actual point variance in noisy single-game outcomes.

**4. Live calibration (pooled signed bias, % of actual mean):**
model QB −7.3% / RB +10.0% / WR +13.9% / TE +9.2%, vs baselines all −17 to −22%
(naive stats → conservative points, as expected). The model is 2–4× better calibrated
than either baseline live, but NOT zero: it over-projects skill positions ~+10–14%
(the 2024–25 test-split calibration was ±3.3%) — early-season weeks carry most of this;
track whether it persists through w17 before touching anything.

**5. Top-of-board reliability (mean ACTUAL rank of the projected top-5; chance = median (n+1)/2):**
QB 9.7 (chance 17.5; last 16.6, tr3 14.6) · RB 13.4 (44.5; 18.7/17.2) · WR 24.5 (71.5;
31.8/25.2) · TE 16.1 (39.8; 23.7/21.7) — the model's top-5 reliably lands in the top
half of the actual board, ahead of both baselines at every position.

**Verdict: good, with a caveat.** Real start/sit and flex value (top picks out-score the
naive boards by 2.3–10.6 pts depending on position; flex board ~3.3× the field mean),
well above the noise floor, and far better calibrated than the baselines live. Caveat:
only 6 full scored weeks + one partial — per-game win rates are noisy, and the
+10–14% live skill-position over-projection is the thing to track. **Nothing to change**:
the standing rule governs model changes and nothing here calls for one. **NOT tested:**
full 2025 season on the current model (in-sample — not a valid OOS for it);
FantasyPros head-to-head on live 2026 weeks — **now closed (2026-11-19), see the
dedicated "Live 2026 head-to-head vs FantasyPros" section below.**

## Live 2026 head-to-head vs FantasyPros — the "is it good" gap, closed (2026-11-19)

The "Is it good?" battery (2026-11-01) flagged one untested item: "FantasyPros
head-to-head on live 2026 weeks … not attempted." Closed it. The FP public API
now archives 2026 weeks (top-10/position, free tier), so we ran the apples-to-
apples head-to-head on the SCORED live 2026 weeks (w1–w5, w7–w10 — the only
true OOS for the shipped model, which was trained through 2025).

**Method** (mirrors the 2024 `compare_fantasypros`): for each scored week ×
position, compare the FP top-10 board against OUR logged pre-kickoff
predictions (the clean live record in `holdout_log.jsonl`) on the overlapping
players; actuals from `features.parquet`; ESPN Standard (non-PPR) points. Both
sides are pre-week commitments. n = 358 player-weeks (1,372 raw-stat rows).

**Result — we beat the consensus, modest but real:**
- **Pooled:** points MAE **6.05 vs 6.22**, RMSE **7.57 vs 7.61**, Spearman
  **+0.534 vs +0.512** — we win 3 of 4 positions on each.
- **By position (pts MAE / pts Spearman / top-5 mean actual, ours vs FP):**
  - QB  5.98/6.37 · +0.448/+0.369 · 20.24/20.42 (top-5 near-tie, FP edges us)
  - RB  6.66/6.98 · +0.300/+0.297 · 18.73/18.62 (we win all three)
  - WR  6.96/6.71 · +0.116/+0.139 · 14.43/14.13 (we TRAIL MAE+Spearman; win top-5)
  - TE  4.63/4.85 · +0.321/+0.206 · 9.16/8.62 (we win all three)
- **Weekly:** we win MAE in 6 of 9 weeks (lose w5/w8/w9) and Spearman in 6 of 9
  (lose w2/w5/w9); w10 is a clean sweep (MAE 6.50 vs 7.74, Rho +0.811 vs +0.691).
- **Raw components (apples-to-apples):** pooled MAE 12.27 vs 12.49 (we win);
  we win QB/TE, trail RB/WR by a hair.

**Read:** same shape as the 2024 finding — the "beat the consensus" claim holds
on signal/quality (ranking + error), not as a blowout. The one real weakness is
**WR** (we trail FP on every WR metric). The FP free tier caps at top-10/position,
so this is a top-10 head-to-head, not a full-roster one.

**Repro:** `uv run python -m src.fetch_fantasypros --season 2026 --weeks 1 2 3 4 5 7 8 9 10`
then `uv run python notebooks/_fp_2026_scoreboard.py` → `models/fantasypros_2026_report.json`
(+ `notebooks/fp_2026_scoreboard.html` via `notebooks/_gen_fp_2026_html.py`).

## Third-party LLM feedback review — claim audit + A/B (2026-10-22)

An external LLM (not involved with this repo) submitted a review listing "bugs" and
improvements (pasted as-is; implementing every item was explicitly **not** required).
Every claim was verified against the actual code/data, every testable proposal A/B'd
against the clean 2025 holdout, and decisions used the standing rule (ship only if it
beats baseline V0; within ±0.01 Spearman = noise → keep baseline).

### Claim audit (third-party claim → verdict)

1. **"Scoring multiplier bug"** (passing 0.1/yd, 6/TD in the src tables; only the board
   used 0.04/4) — **partially true.** `src/bench`/`calibrate`/`holdout`/`compare_fantasypros`
   carried an internally consistent heavy-QB 1-PPR scheme (0.10/pass-yd, 6/pass-TD, +1 PPR),
   not the ESPN Standard (0.04/4, no PPR) the user's league actually uses — so pooled
   holdout Spearman measured a different ranking space than the board. **Fixed:** single
   source of truth in `src/scoring.py` (ESPN Standard), all four modules import it; W1–W5
   2026 re-scored (log untouched; 1-PPR numbers archived in
   `models/holdout_score_1ppr_archive.json`). Under ESPN Standard the model still beats
   both baselines every week (pooled ρ: W2 +0.718, W3 +0.758, W4 +0.762, W5 +0.674) and
   still beats FantasyPros on MAE+RMSE at every position.
2. **Double-counted passing yards + opponent join "bug"** in `features.py` — **observation
   true, fix already rejected.** `team_weekly_stats` has `passing_yards == receiving_yards`
   exactly, so old `total_yards` double-counted passing, and `*_opp` was the opponent's
   trailing *offensive* yards (mislabeled "allowed"). But these are strictly trailing
   features (shift(1)) — no lookahead — and the earlier A/B for true opponent-defense
   features (yards/sacks/TDs allowed) found no improvement. Re-A/B'd the fully-corrected
   version (V1): pooled ρ +0.0013, within noise → **not applied**.
3. **NGS joined on display name, not `player_id`** — **true but negligible:** the gsis
   join recovers exactly one QB (Cam Ward, "Cameron Ward"); the 883 dropped
   (season, week, name) rows are pre/postseason weeks (0, 18, 19–22), not name collisions;
   PJ Walker is a name-variant risk. 2025-holdout impact ≈ 0 (V1) → **not applied** per the
   A/B rule; one-line correctness fix to bundle into the next retrain if desired.
4. **`predict.py` lacks scheduled panel / QB rushing / RANK_STAT ranking** — **true for the
   CLI** (the holdout logging path already had the fallback). **Fixed:** unplayed weeks
   fall back to `src.holdout._build_scheduled_panel` (no --refresh needed; verified W8,
   445 players); QB sentences include rushing; `--all` boards rank by projected ESPN
   Standard points; --refresh documented as destructive and not recommended.
5. **"Washington/Shough float to the top of W4 for lack of usage features"** — board quote
   was accurate and current. W4 actuals (ESPN Standard): Washington 12.0→1.0, Downs
   11.6→1.6, Lamb 10.4→24.3 — the low-snap efficiency error is real. The proposed remedy
   (usage shares) was A/B'd and **won** (below) → **applied**.
6. **MAE-based grid selection favors over-regularized trees; use RMSE + early stopping** —
   A/B'd (V4): pooled ρ +0.005, within noise → **not applied**.
7. **Poisson loss for count stats** — A/B'd (V5): pooled ρ −0.005 (worse), early-season
   worse → **rejected**.
8. **Recalibrate p10/p90 bands (conformal or Poisson)** — A/B'd without retraining
   (`notebooks/_exp_bands.py`): split-conformal shifts (2024-val signed-residual
   percentiles) lift QB passing-yards coverage 76.1%→79.7% (nominal 80%), all other bands
   neutral (86–92%); Poisson TD bands over-cover 95%+ → rejected. **Applied:**
   `models/band_shifts.json` + `src/predict.py` applies the shifts at the output layer
   (stored band models untouched, same pattern as zero-clipping).
9. **Prior-season bridge as INPUT features (not post-hoc blending)** — A/B'd (V3): pooled
   ρ +0.010 (borderline), early-season +0.022. Combined with usage shares (V6): pooled ρ
   +0.017, early-season +0.036, 12/16 beat baseline → **applied (V6)**.

### A/B arms (2025 clean holdout; train 2016–23 / val 2024 / test 2025, full 48-config grid)

| arm | change | pooled MAE | pooled ρ | beat-base | early ρ (wk1–4) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | retrained baseline | 8.074 | 0.4959 | 9/16 | 0.4981 | — |
| V1 | NGS gsis join + team-yards redefinition + opp allowed yards | 8.032 | 0.4972 | 9/16 | 0.5047 | noise → not applied |
| V2 | usage shares (target/air-yds/WoPR/RaCR tr3+s2d) | 8.034 | 0.5098 | 12/16 | 0.5267 | **applied** |
| V3 | prior-season per-game bridge | 8.054 | 0.5057 | 11/16 | 0.5201 | borderline; in combo |
| V4 | RMSE-based grid selection | 8.083 | 0.5009 | 10/16 | 0.4999 | noise → not applied |
| V5 | Poisson loss for counts | 8.070 | 0.4912 | 11/16 | 0.4936 | worse → rejected |
| **V6** | **V2 + V3** | **8.063** | **0.5127** | **12/16** | **0.5344** | **applied (shipped model)** |

V6 per-target: 15/16 Spearman up; biggest gains on receiving targets (RB rec_yds +0.035,
RB receptions +0.042, TE rec_yds +0.023, TE receptions +0.021); only regression
QB/passing_yards −0.012 (single-season noise; QB position ρ overall 0.410→0.425, best of
all arms).

### Shipped changes (A/B-gated or correctness fixes)

- `src/scoring.py` — ESPN Standard single source of truth; bench/calibrate/holdout/compare import it.
- `src/features.py` — +8 usage-share features + 7 prior-season features (V6); `load_usage()`;
  the scheduled-panel path passes a dummy-row usage table (`src/holdout.py`).
- `src/model.py` retrain (2026-10-22) — shipped `models/models.joblib` now carries the V6
  features (95 model features). **Live-holdout note:** 2026 W1–W7 logs come from the
  pre-V6 model (frozen, scored as-is); W8 onward logs with the V6 model.
- `src/predict.py` — scheduled-panel fallback, QB rushing in the sentence, fantasy-point
  board ranking, conformal band shifts.
- `models/band_shifts.json` — conformal band calibration applied at the output layer.

### NOT applied (tested, rejected or within noise — do not re-test without a new signal)

- V1 team/yardage redefinition (double-count fix + opp allowed yards): within noise on 2025.
- V1 NGS gsis join key: within noise on 2025 (recovers 1 QB); correctness-only.
- V4 RMSE grid selection.
- V5 Poisson count loss.
- Poisson TD uncertainty bands (over-cover 95%+).
- Injury/roster "vacated opportunity" features: NOT A/B'd — `injuries.parquet` is
  report-level (sparse `report_status` strings), not player-game designations; building a
  clean vacated-usage feature would need a data pipeline the A/B can't cheaply validate,
  and the V6 usage shares already capture much of the reallocation signal in-sample.

Repro: `uv run python notebooks/_exp_feedback_ab.py` (V0–V5, ~30 min) + the V6 combo arm
appended to `models/ab_feedback_report.json`; band experiment:
`uv run python notebooks/_exp_bands.py`.

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
