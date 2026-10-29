# Data Transformations and Imputation Log

Every step from the raw Bloomberg pull (`data/is2/bbg_data/`, frozen 2026-10-21) to the final table
`data/processed/features_1m_v3.parquet`. Use it as the reference for the report's data section, and update it
in the same change as the code. The evidence for each quirk is in [DATA_AUDIT.md](DATA_AUDIT.md) (F1-F11).

Rebuild: `uv run hftf data build-features`, then `uv run hftf data audit --strict`. The build is
deterministic. `features_1m_v3.manifest.json` records what was dropped and filled on each run.

## Principles

1. **Nothing is synthesised.** No session, bar, tick or quote is invented. A missing session is dropped
   whole. No model-based imputation is used anywhere: no interpolation, regression, KNN or cross-stock
   borrowing.
2. **Only last-observation-carried-forward (LOCF), and only inside a session.** A carried value never
   crosses the overnight boundary, and every carried value is flagged.
3. **Causal.** Every value in a row is observable at `decision_time = time + 1 min`. Rolling windows reset
   at each open, the volume profile uses earlier sessions only, and daily features are lagged one session.
4. **Real extremes stay.** Nothing is clipped or winsorised in the dataset. Scaling and clipping belong to
   the environment's normaliser, fitted on the train split only.
5. **Excluded is not deleted.** Rows that should not seed episodes stay in the table with `split = excluded`.
   They still serve as rolling-window history.

## Summary

| # | Technique | Kind | Where | What it affects |
|---|---|---|---|---|
| T1 | Bloomberg N/A sentinel → NaN | cleaning | `loaders.clean_sentinels` | index weights, EPS (daily/static) |
| T2 | All-NaN weekend rows dropped | cleaning | `loaders.load_daily` | daily index files |
| T3 | Zero-price pre-open quotes → NaN | cleaning | `loaders.load_bars` | BID/ASK bars 09:00-09:15 and the auction |
| T4 | Duplicate bars de-duplicated (last wins) | cleaning | `loaders.load_bars` | overlapping chunk files |
| T5 | Tick trades filtered by condition code | cleaning | `ticks.clean_trades` | `T`, `ACB`, `AOB/AUO/AUC` prints dropped |
| T6 | Quotes collapsed per timestamp, then LOCF | cleaning | `ticks.clean_quotes` | the ~3% crossed states → 0 |
| T7 | Lee-Ready trade signing | derivation | `ticks.clean_trades` | `buy_vol`, `sell_vol`, `signed_vol`, `trade_imb` |
| T8 | Tick size inferred from the price grid | derivation | `derive.infer_tick_size` | `tick_size`, `spread_ticks` |
| T9 | Bar-level L1 imbalance proxy | derivation | `derive.bar_imbalance_proxy` | `imb_bar_proxy` on every day |
| T10 | Uniform 09:15-15:15 grid | transformation | `calendar.session_grid`, `GRID_END` | 15:15-15:30 removed on every day |
| T11 | Incomplete sessions dropped whole | exclusion | `features.build_security` | 2026-09-10 (all stocks) |
| T12 | No-trade minute: volume 0, OHLC = last close | LOCF imputation | `features.build_security` | `fill_trade` |
| T13 | No-quote minute: last bid/ask carried | LOCF imputation | `features.build_security` | `fill_quote` |
| T14 | Context index minute carried before differencing | LOCF imputation | `features.context_features` | `ctx_*` |
| T15 | Partial-window warm-up estimators | estimation | `features._rv`, `_diff`, `_roll` | first 30 min (`dq_warmup`) |
| T16 | Causal volume profile (previous 20 sessions) | derivation | `features.build_security` | `prof_*`, `rel_volume` |
| T17 | Daily features lagged one session | transformation | `features.daily_lagged` | `d_*` |
| T18 | 1-day implied-vol gap carried | LOCF imputation | `features.daily_lagged` | `d_ivol_30d`, 1 day |
| T19 | Regime-input transforms (EMA, log) | transformation | `features.build_security` | `imb_bar_proxy_ema5`, `log_*_30m` |
| T20 | Tick features left NaN on non-tick days | deliberately not imputed | `features.build_security` | `l1_imb_*`, `ofi*`, trade flow |
| T21 | Split labels and exclusions | labelling | `features.assign_split` | `split`, `dq_burnin`, `dq_partial_day` |
| T22 | Data-quality flags | flagging | `features.build_security` | `dq_offbook_frac`, `dq_*`, `fill_*` |
| T23 | 1-second L1 grid of the tick sessions | LOCF within session, flagged by age | `l1.second_grid` | `l1_1s_v1.parquet` (TickSim) |

## Cleaning the raw pull

**T1. N/A sentinel.** Bloomberg returns `-2.424536e-14` for "not available" in some numeric fields. Values
within 1e-16 of it become NaN. It affects every index-membership weight, so those weights are unusable, plus
some EPS fields. None of these fields feed the feature table.

**T2. Weekend rows.** Daily index files contain weekend and holiday rows where every price is NaN. Rows with
all of OHLC NaN are dropped.

**T3. Zero-price quotes.** During the pre-open call auction (09:00-09:15), and in the closing auction from
2026-08-03, BID/ASK bars report price 0.0 because no book is visible. Prices ≤ 0 become NaN, so they can
never form a spread or a mid.

**T4. De-duplication.** Bars can appear in more than one chunk file. Files are read in sorted order and the
last row per timestamp wins.

**T5. Trade-print filter.** Bloomberg's NSE ticks are ~1 s conflated snapshots. The prints kept are untagged
and `BC` ones (the last print, and the volume batched since the previous snapshot) inside 09:15-15:30 with
a price > 0. Dropped: `T` (cumulative indicative volume), `ACB` (day total) and `AOB/AUO/AUC` (auction
markers). Check: clean tick volume equals 1-min bar volume within 0.04% on every full tick day (R06).
Without this filter, tick volume was overstated 2-76×.

**T6. Quote collapse.** BID and ASK arrive as separate rows sharing one timestamp. The last BID and the last
ASK are kept per timestamp, and only then forward-filled to form the L1 state. States with a non-positive
price or size, or with bid ≥ ask, are dropped. Check: 0 crossed quotes (R07). A row-by-row forward fill gave
~3% crossed states.

## Derived quantities (not in the pull)

**T7. Trade sign.** Each trade is compared with the quote prevailing *strictly before* it (`merge_asof`,
exact matches excluded): above the mid = buy, below = sell. At the mid, the tick rule (sign of the last
non-zero price change) decides. If neither works, the sign is 0 (unsigned).

**T8. Tick size.** `TICK_SIZE` was not returned. The tick is the largest candidate (5, 1, 0.5, 0.1, 0.05,
0.01) on whose grid ≥ 99.5% of trade prices lie. Check: it matches the NSE price-band rule for all stocks (R09).

**T9. Imbalance proxy.** A BID (ASK) bar's `volume / numEvents` is the mean displayed size at the touch in
that minute, and `imb_bar_proxy = (b̄ − ā)/(b̄ + ā)`. It replaces tick L1 imbalance on the 78% of days
without ticks, and it is used on all splits for consistency. Check: Pearson 0.77-0.86 against tick L1
imbalance (R08). Inside a session it is carried forward across minutes without a quote bar.

## Session grid and exclusions

**T10. 09:15-15:15 grid.** From 2026-08-03 the data shows a closing call auction from 15:15 to ~15:30: at
most 2 bars, zero-price quotes, indicative `T` prints, and an `AUC` uncross. Every session, before and
after that date, is cut to 09:15-15:15 (360 one-minute rows), so all splits share one market structure.
This removes 725-727 TRADE bars per stock. They are 4.1-4.8% of continuous-session volume and come mostly
from the pre-August days, when 15:15-15:30 still traded continuously. That volume now shows up in
`dq_offbook_frac`. The auction is not modelled.

**T11. Whole-session drop.** A session is kept only if it has TRADE, BID and ASK bars and the context index
bars. Only 2026-09-10 fails: ADE has TRADE and ASK but no BID, and nothing else exists. It is dropped for
all three stocks. Rejected alternative: synthesising a minute path for an unobserved day would be
fabrication. The env samples whole sessions, so losing one costs a little data and introduces no bias. The
lagged daily features of Sep 11 still use Bloomberg's real Sep 10 daily bar. The readiness gate records
Sep 10 as an acknowledged gap (`acknowledged_missing_sessions`).

## Imputation (all LOCF within the session, all flagged)

**T12. No-trade minute** (`fill_trade`). No TRADE bar means nothing traded: `volume`, `n_events` and
`value` = 0, and OHLC = the last close of the same session. 565 rows in total. 564 are on the excluded
2026-10-21 afternoon, after the data stops, and **1 is in a usable session**.

**T13. No-quote minute** (`fill_quote`). The book is assumed unchanged, so the last bid and ask of the same
session are carried. 568 rows: 564 on the excluded Oct 21 and **4 in usable sessions** (Jul 28, Aug 17,
Aug 26). `mid` and `spread_bps` exist only where bid > 0 and ask > bid. Otherwise `px` falls back to the
last trade close.

**T14. Context index minute.** NIFTY, BANKNIFTY and India VIX bars are placed on the same grid and carried
forward inside the session *before* any return or change is taken, so a missing minute gives a 0 return
rather than a NaN. 182-183 minutes per index are missing, all on the excluded Oct 21 afternoon.

**T18. Daily implied vol.** If `30DAY_IMPVOL_100.0%MNY_DF` is missing for one day, the previous day's value
is carried (limit 1 day) before the one-session lag. It affects ADSEZ on 2026-06-08 (an excluded burn-in day).

## Estimation over shorter windows (warm-up)

**T15. Partial windows** (`dq_warmup` = minutes 0-29 of each session). A w-minute window at minute m < w
holds only m observations of the current session. v2 left those minutes NaN, which removed the first 10-60
minutes of every day from the observation. v3 uses what the session has observed so far:
- realised volatility: `rv_w = sqrt(Σ r² · w / n)` over the n ≥ 3 available returns. Scaling by w/n makes
  the partial sum an unbiased estimate of w-minute variance under locally constant variance;
- spread variance and mean, Parkinson volatility: the sample statistic over the n ≥ 3 available minutes
  (no rescaling, because these are per-minute statistics);
- returns `ret_w`, `ctx_*_ret_30m` and `ctx_vix_chg_30m`: the change since the session's first value while m < w;
- tick-only rolling sums (`ofi_5m` and so on) keep v2's rule (n ≥ 2, unscaled sums of flow).

Episode rows inside the warm-up: 7,176 of 98,256 (minutes 4-29). They carry `dq_warmup` so the effect can
be ablated.

## Feature transformations

**T16. Volume profile.** `prof_share` / `prof_avg_vol` at minute m is the mean of the share / volume at m
over the **previous** 20 sessions (at least 3 required), and `prof_cum_share` is its running sum.
`rel_volume = volume / prof_avg_vol`. No current-day data is used.

**T17. Daily features.** Returns, 20-day vol (annualised √252), 20-day ADV, range, Amihud illiquidity,
implied vol and volume come from the daily file and are shifted by one session, so each is known before the
open. Same-day daily volume is used only for `dq_offbook_frac` (a flag, never an input).

**T19. Regime inputs** (`configs/env.yaml` `regime_inputs`):
- `imb_bar_proxy_ema5`: an EMA (span 5, `adjust=False`) of `imb_bar_proxy` within the session. The raw
  proxy is noisy (std 0.44 → 0.31);
- `log_spread_var_30m = log(spread_var_30m + 1e-8)` and `log_rv_30m = log(rv_30m + 1e-8)`. Both are
  heavy-tailed, and the log makes them usable by Gaussian membership functions. On episode rows the minima
  are 0.026 bps² and 0.0003, so ε never dominates.

The three are near-orthogonal (Spearman |ρ| ≤ 0.29).

## Deliberately not imputed

**T20. Tick-only features** (`l1_imb_*`, `microprice_dev_bps`, `tick_spread_*`, `ofi*`, `l1_depth_mean`,
`n_quote_updates`, `*_sz_last`, `n_trades`, `tick_volume`, `buy/sell/signed_vol`, `trade_imb`) exist on
the 20 tick days only (val/test) and are **NaN elsewhere, never zero-filled**. Inside a tick day, minute
counts with no event are 0, and the last state is carried. None of them is an env observation. They support
analysis and the R06-R08 checks.

Also not done: no outlier clipping (spreads up to 56 bps and 1-min moves of about 1.2% are real prints on
heavy volume), no smoothing of prices, no cross-day forward fill, and no use of futures or FX (the NZ1 ASK
Jun 3-5 and FX Aug 26 gaps are therefore irrelevant).

## 1-second L1 grid (TickSim input, M5)

**T23. `data/processed/l1_1s_v1.parquet`** (`uv run hftf data build-l1`, `hftf_data.l1`; a manifest is
written next to it). One row per second of 09:15-15:15 IST for every tick session kept in
`features_1m_v3` (63 sessions: val 24, test 36, the excluded partial Oct 21: 3), 1,360,800 rows.
- **Quotes:** the T6-collapsed L1 state after the last update stamped in that second. A second without an
  update **carries the previous state inside the session** (LOCF, as T13), and `quote_age_s` counts the
  carried seconds (17.6% of rows; p99 3-4 s, max 16 s on val/test). Seconds before the session's first
  valid quote (at most 2 s after 09:15) stay **NaN, never back-filled**; the simulator reads rows from 09:20.
- **Trades:** the T5-filtered prints with the T7 sign, summed per second (`volume`, `buy_vol`, `sell_vol`)
  with `px_min` / `px_max`. A second without a print has volume 0 (observed: nothing traded) and NaN prices.
- **Clock:** bar `m` = seconds `60m+1 … 60m+60` (DATA_AUDIT F12). Per-minute tick volume equals bar volume
  on every tick minute, and the bar's closing quote equals row `60(m+1)`.
- Nothing is synthesised: there are no grids for non-tick days, and the train split has none.

## Labels and flags

**T21. Splits** (`configs/env.yaml`): train 2026-06-03 → 08-31, val 09-01 → 09-30, test 10-01 → 10-19.
These rows are `excluded`:
- `dq_burnin`: the first 3 sessions (Jun 3-5), which have no volume profile yet;
- `dq_partial_day`: active minutes < 80% of the median (2026-10-21, data to ~12:05).

| split | sessions per stock | rows (3 stocks) |
|---|---|---|
| train | 60 | 64,800 |
| val | 20 | 21,600 |
| test | 12 | 12,960 |
| excluded | 4 | 4,320 |

**T22. Flags** (filter or analysis only, never model inputs): `fill_trade`, `fill_quote`, `dq_warmup`,
`dq_burnin`, `dq_partial_day`, `dq_offbook_frac` (the share of daily volume outside the grid's bars; it uses
same-day data, so it is **not causal**). The eval-only label `regime_eval_prevday` (previous-day RV
tercile) is not a model input either.

## Guarantees checked on every build

`hftf data audit --strict` (thresholds in `configs/data_readiness.yaml`; see [DATA_READINESS.md](DATA_READINESS.md)):
- **R06-R09:** T5, T6, T8 and T9 still hold on the real data;
- **R11:** coverage of the eFNN inputs;
- **R12:** 0 NaN in every observed column (`episode_columns`) on all train/val/test rows from the 09:20
  decision. The env never has to impute at run time.

Unit tests (`packages/hftf-data/tests/`) pin T5-T9, T10, T15 and T21 on synthetic data.

## For the next stage (M1 simulator)

- Read `features_1m_v3.parquet` and sample episodes from `split ∈ {train, val, test}` rows only, with
  `t0 ∈ [09:20, 15:15 − H]`. The row stamped `time` is observable at `decision_time`.
- Fit observation scalers on train rows only. `train_stats` in the manifest gives mean, std, p01 and p99 per
  episode column, e.g. for clipping at p01/p99 before standardising.
- Use realised `volume` only inside the fill model, never in the observation. Expected volume for `Q` and
  the schedule is `prof_avg_vol` / `prof_share`.
