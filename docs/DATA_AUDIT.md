# Data Audit: Bloomberg pull (`data/is2/bbg_data`)

Audited 2026-10-21, finalised 2026-10-22 (F10, F11). Reproduce with `uv run hftf data build-features`
(writes `data/processed/features_1m_v3.parquet`), then `uv run hftf data audit` (writes `docs/data/data_audit.json`).

## Verdict

**The data is usable for the project. Its scope is a pilot.** It has every input the method needs:
L1 order-book imbalance, spread and spread variance, realized volatility, the volume profile, and
market context. For a credible paper or thesis, though, it is thin: **3 stocks × 96 sessions of 1-minute
bars and 20 sessions of conflated L1 ticks.** That is enough to build and debug the full pipeline and
to show regime-dependent behaviour. It is not enough for strong out-of-sample claims. The
collector notebook confirms this was the `LIMIT = 3` smoke-test pass. **The pull is now frozen** (no further
pulls in this project's scope). `features_1m_v3` is the final dataset (F11), and the readiness gate is READY (12/12).

## What was delivered

| Dataset | Coverage | Granularity | Status |
|---|---|---|---|
| 1-min TRADE bars | ADE, ADSEZ, APHS, NIFTY, BANKNIFTY, INDIA VIX, NZ1/NZ2 (Nifty fut), AF1 (BankNifty fut) | 2026-06-03 → 2026-10-21 12:05 IST, 96 sessions | Good |
| 1-min BID/ASK bars | 3 equities, 3 futures, USDINR, EURINR | same | Good; DXY empty (no entitlement) |
| Ticks (TRADE/BID/ASK + sizes) | ADE, ADSEZ, APHS | 2026-09-21 → 2026-10-21, 20 full + 1 partial session | Good after cleaning (see below) |
| Daily history | 6 equities, 3 indices, 3 futures, FX, rates, commodities, 3 options | 2011-10-21 → 2026-10-21 (15y) | Good |
| Static/reference | 3 per class, Nifty membership, futures + options chains | snapshot | `TICK_SIZE`, `TRADING_LOT_SIZE` missing |
| Events | dividends, earnings dates | history | Good; EPS uses N/A sentinel |
| `features/features_1m.parquet` | collector-built features | 1-min | **Defective, superseded by v2** |

Equities are ADANI ENTERPRISES (ADE, ~₹3000, ₹385 cr ADV), ADANI PORTS (ADSEZ, ~₹1770, ₹316 cr ADV) and
APOLLO HOSPITALS (APHS, ~₹8760, ₹267 cr ADV), the first three Nifty-50 members alphabetically.
`ACC/ACEM/ARBP` have daily data only.

## Findings and fixes

### F1. Tick TRADE sizes overstated volume 2-76× (critical, fixed)
Bloomberg's NSE tick history is a **~1-second conflated snapshot stream**, not the raw exchange feed:
about 40k events per stock-day, with 55% of rows sharing a timestamp. The condition codes mean:

| Code | Meaning (empirical) | Treatment |
|---|---|---|
| none | last print in the snapshot | keep |
| `BC` | volume batched since the previous snapshot | keep |
| `T` | cumulative indicative volume (pre-open and pre-close) | drop |
| `ACB` | day-total summary print at ~15:29 | drop |
| `AOB`/`AUO`/`AUC` | auction markers | drop |

After filtering, continuous-session tick volume equals 1-min bar volume at a ratio of **0.9996-1.0005 on
every full day** (Oct 21 is 1.02-1.04 only because ticks were pulled a few minutes after the bars).
The collector's `tick_volume`, `buy_vol`, `sell_vol`, `signed_vol`, `trade_imb` and `n_trades` were
all contaminated. → `hftf_data.ticks.clean_trades`.

### F2. ~3% crossed quotes in the collector's L1 state (fixed)
BID and ASK arrive as separate rows sharing one timestamp. Forward-filling row by row pairs a new bid
with a stale ask. Collapsing to the last BID/ASK **per timestamp** before the fill leaves 0 crossed
states. → `hftf_data.ticks.clean_quotes`.

### F3. L1 imbalance existed for only 22% of minutes (fixed by derivation)
Ticks cover 20 of 96 sessions. A **bar-level proxy** fills the rest: each BID/ASK bar's `volume / numEvents`
is the mean displayed size at the touch, and `(b̄ − ā)/(b̄ + ā)` correlates with the tick-derived
L1 imbalance at **Pearson 0.77 / 0.84 / 0.86** (ADE / ADSEZ / APHS, ~7.4k minutes each). The feature
table carries `l1_imb` and `l1_imb_source ∈ {tick, bar_proxy}`, so the agent can be trained on the
proxy and evaluated on tick days. OFI and microprice are tick-only and have no proxy.

### F4. Tick size and lot size missing from static data (fixed by derivation)
Tick size is inferred from the trade-price grid. It matches NSE's price-band tick rule
(₹1000-5000 → ₹0.10, ₹5000-10000 → ₹0.50) for all 3 stocks. → `hftf_data.derive`.
The cash-equity lot size is 1 share. The futures lot sizes are in `static_futures` (NIFTY 65, BANKNIFTY 30).

### F5. Pre-open quotes have price 0.0 (handled)
BID/ASK bars and ticks during 09:00-09:15 report price 0 (auction, no visible book). They are masked to NaN.
The environment trades only in the continuous session (09:15-15:30 IST).

### F6. Bloomberg N/A sentinel `-2.424536e-14` (handled)
It appears in index-membership weights (all of them, so the weights are unusable) and EPS fields.
`loaders.clean_sentinels` converts it to NaN.

### F7. Calendar gaps
- **2026-09-10 was a trading session, but its bars are missing** for 8 of 9 TRADE instruments and for every
  BID/ASK series except ADE ASK (ADE has TRADE and ASK but no BID). This is a collection gap. With the pull
  frozen, the session is **dropped whole** from the feature table for every stock (F11), not imputed.
- Exchange holidays, legitimately empty: Jun 26, Sep 14, Oct 2, Oct 20.
- FX also lacks Aug 26 (an FX-market holiday).
- Oct 21 is a partial session (data ends ~12:05 IST). `dq_partial_day` flags it, and it is `split = excluded`.
- NZ1 ASK lacks Jun 3-5. No feature uses futures quotes, so no action is needed.
- Daily NIFTY/BANKNIFTY rows include weekends with all-NaN prices. The loaders drop them.

### F10. Closing call auction from 2026-08-03 (handled: the grid ends at 15:15)
Through 2026-07-31 every stock trades continuously until 15:30. From 2026-08-03 (all 53 later sessions),
the 15:15-15:30 window has at most 2 one-minute bars. Its ticks are indicative `T` prints with zero-price
quotes, and an `AUC` (auction uncross) print lands at ~15:29:40. That is a closing call auction. It was
measured from the data, not taken from an NSE circular, so check a circular before quoting it.
v2 put those 15 minutes on the grid as zero-volume, stale-price "trading" minutes, which an agent could have
learned to trade into. → `calendar.GRID_END`: the grid is 09:15-15:15 (360 min) on **every** day, so train
(days before and after the change), val and test share one market structure. The pre-August days lose their
real 15:15-15:30 trading. `dq_offbook_frac` shows it (median 10% before the change, 3% after). The auction
itself (executing a residual at the close) is not modelled in v1.

### F11. Final dataset gap policy (`features_1m_v3`)
Every value is observed, carried forward inside its own session, or estimated over a shorter window during
warm-up. Every carried or estimated value is flagged. Nothing is synthesised.
- **A session is kept only if its TRADE, BID, ASK and context bars all exist.** This drops 2026-09-10 for all
  three stocks.
- **Within-session fills:** a minute with no TRADE bar gets volume 0 and OHLC = last close (`fill_trade`).
  A minute with no BID/ASK bar carries the last quote (`fill_quote`). Almost all of these are on the excluded
  Oct 21 afternoon; only 1 and 4 rows are in usable sessions. Context
  index bars are carried forward the same way before differencing.
- **Warm-up:** rolling statistics need at least 3 same-session observations. Realised variance over n < w
  returns is scaled by w/n, and inside the first w minutes `ret_w` is the return since the open
  (`dq_warmup` marks mso < 30). v2 left 10-60 minutes NaN after every open.
- **Exclusions** (`split = excluded`): the profile burn-in days (Jun 3-5, `dq_burnin`) and the partial Oct 21.
- **Not clipped:** spreads up to 56 bps and 1-min moves of about 1.2% are real prints on heavy volume.
  Winsorising is the env normaliser's job, fitted on train.
- **Added inputs:** `imb_bar_proxy_ema5`, `log_spread_var_30m`, `log_rv_30m` (`env.yaml` `regime_inputs`).

The result is 92 usable sessions per stock (train 60, val 20, test 12), 98,256 episode rows, and 0 NaN in
the R12 columns. `data/processed/features_1m_v3.manifest.json` records what was dropped, the fill shares
and the train-only statistics.

### F8. Off-book volume days (flagged)
On 2026-08-31, 09-18 and 09-29, continuous-session bar volume is only 42-65% of Bloomberg daily volume
(block, bulk or closing-session prints). `dq_offbook_frac` records this. These days are kept, but their
participation statistics are biased.

### F9. Smaller issues
- Daily `NUM_TRADES` is only 62% populated, and `PE_RATIO`/`PX_TO_BOOK` are sparse. None of these are needed.
- `VOLUME_AVG_20D`, `HIGH_52WEEK` and similar are not valid historical fields. They are recomputed from daily data.
- The options snapshot logged 9.7k field errors. Options are out of scope for v1.
- The probe (`capabilities.json`) reports `bar_history_probe: false` everywhere, yet bars reach back ~140
  days. The probe was wrong, and the data is fine.

### F12. Bars are right-closed on the tick clock (measured 2026-10-29, for TickSim)
Tick stamps carry sub-second precision. Put on a 1-second grid (second `s` = floor of seconds after
09:15 IST), the 1-minute bar labelled `m` holds **exactly** the ticks of seconds `60m+1 … 60m+60`. Volume
matches on 100% of tick minutes (left-closed binning `60m … 60m+59`: 51%), and the bar's closing bid/ask
equal the tick L1 state at second `60(m+1)` on 100% of minutes (0 mismatches; at second `60m+59`: 28%).
So the state the agent had seen when deciding at the end of bar `m` is grid row `60(m+1)`, and the tick
grid and the bars share one clock. → `hftf_data.l1`, `hftf_sim.tick`.

## Market structure facts measured from this data

| | ADE | ADSEZ | APHS |
|---|---|---|---|
| median price (₹) | 2999.6 | 1766.7 | 8759.3 |
| tick size (₹) | 0.10 | 0.10 | 0.50 |
| median spread | 2.0 bps / 6 ticks | 1.8 bps / 3 ticks | 1.8 bps / 3 ticks |
| median ADV (shares / ₹ cr) | 1.29M / 385 | 1.77M / 316 | 0.31M / 267 |
| median daily RV (bps) | 145 | 116 | 97 |
| median L1 depth, bid+ask (shares) | 123 | 209 | 74 |

- **U-shaped intraday volume:** on the 09:15-15:15 grid, the first 30 min carry ≈15% of the day and the
  last hour (14:15-15:15) ≈22%. Both figures hold before and after the auction change. Spreads are widest in
  the first 5 minutes (3.5 bps). Before the change, 15:15-15:30 was wide too (3.8 bps), but those minutes are
  now off the grid.
- **L1 depth is tiny next to any realistic parent order:** 1% of ADV is roughly 100× L1 depth.
  Price impact has to be modelled, not read off the book.
- **The three eFNN regime inputs are nearly orthogonal:** Spearman |ρ| ≤ 0.29 between imbalance (5-min EMA),
  30-min spread variance and 30-min RV. Each adds information to the fuzzy partition.

## What cannot be recovered from this pull
- **Book depth beyond L1** (levels 2-5). The fill simulator therefore uses a parametric impact model,
  calibrated per stock, instead of replaying a full book. Getting depth needs a forward recording via
  `//blp/mktdepthdata` (`bbg_collector.py depth`, in git history), if entitled.
- **Our own order's interaction with the queue.** Any historical backtest is counterfactual. Impact
  and limit-order fill probability are model assumptions, and they must be stated as such.
- **Intraday history before 2026-06-03.** Bloomberg serves about 140 days of bars and about 1 month of ticks
  on a rolling basis. Anything not pulled now is lost.

## Status and what remains

The pass/fail gate is in [DATA_READINESS.md](DATA_READINESS.md). Status: **READY (12/12)**. Every transformation is logged in
[DATA_TRANSFORMS.md](DATA_TRANSFORMS.md).

The data work is done. If a later project extends the data, the gap-driven pull notebook is in git history
(`de800d8`, `Bloomberg_training_pull.ipynb`). Pull only the names and sessions needed, and merge the delta
**into** `data/is2/`. Nothing in the pipeline is specific to these 3 stocks.
