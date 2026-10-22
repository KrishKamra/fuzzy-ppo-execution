# Data Readiness Checklist

"Data ready" has a pass/fail definition. Check it with:

```bash
uv run hftf data build-features            # R11 needs the feature table to be current
uv run hftf data audit --tier research     # or --tier pilot; add --strict to exit 1 on any failure
```

The audit prints a PASS/FAIL table and stores it under `readiness` in `docs/data/data_audit.json`.
Thresholds live in `configs/data_readiness.yaml`. Change them there, not in code.

## Tiers

| Tier | Purpose | Gate for |
|---|---|---|
| **pilot** | build and debug the full pipeline | M1-M3 (simulator, PPO-MLP, eFNN) |
| **research** | reportable out-of-sample results | M4 (experiments, ablations, write-up) |

## Checks

| ID | Requirement | pilot | research | Why it matters |
|---|---|---|---|---|
| R01 | Intraday equities (bars collected) | ≥ 3 | **≥ 50** | Generalisation across stocks; 3 names overfit |
| R02 | 1-min sessions per equity × {TRADE, BID, ASK} | ≥ 60 | ≥ 90 | Enough sessions for train/val/test by date |
| R03 | Trading sessions missing inside the bar window | ≤ 1 | **0** | Gaps break the volume profile and walk-forward splits |
| R04 | Context bars: NIFTY, BANKNIFTY, India VIX | all | all | Market-context features |
| R05 | Equities with ≥ N full tick days | 3 with ≥ 15 | **10 with ≥ 40** | Validates the imbalance proxy; TickSim; high-fidelity test |
| R06 | Clean tick volume vs bar volume, full days | ≤ 0.1% error | same | Proves tick cleaning is correct |
| R07 | Crossed quotes after cleaning | 0 | 0 | Book state is valid |
| R08 | Imbalance proxy vs tick L1 imbalance (Pearson) | ≥ 0.70 | same | Proxy is a valid eFNN input on non-tick days |
| R09 | Inferred tick size = NSE price-band rule | all | all | Spread-in-ticks and fill prices |
| R10 | Daily history per equity | ≥ 250 sessions | same | ADV, daily vol, Amihud features |
| R11 | `features_1m_v2` covers all equities; eFNN inputs non-null (full days) | ≥ 95% | same | The table the env actually reads |

A "full tick day" is a day whose ticks reach 15:20 IST. Partial days (e.g. pulled mid-session) don't count.

## Current status (audit of 2026-10-21)

| Tier | Result | Failing |
|---|---|---|
| pilot | **READY** | none |
| research | **NOT READY** | R01 (3/50 equities), R03 (2026-09-10 missing), R05 (0/10 equities with ≥ 40 tick days; best 20) |

## Procedure to reach the research tier

The training-data pull is the root notebook `Bloomberg_training_pull.ipynb`. It
supersedes the `LIMIT`-based flow in `data/is2/bloomberg_data_collection.ipynb`, and it is gap-driven:
each run fetches only the sessions and tick days that are missing or partial on disk.

**1. Preview (optional, works on the Mac too):** set `MODE = "plan"` and run all. It prints the missing
sessions, the request count and a coverage preview of R01-R05 and R10. It contacts nothing and writes nothing.

**2. Pull (on the Bloomberg PC):** set `MODE = "pull"` and **Restart & Run All**. In one pass it:
- builds the universe from all current Nifty-50 members (NSE `IS` tickers; the old `IN` seeds pulled the
  composite book), plus names that already have bars and a 5-name Nifty Next 50 buffer for R01. **Verify
  `BUFFER_INDEX`** (`NIFTYJR Index`) in the Terminal; it is skipped if it does not resolve;
- probes a resolved member, checks that bar times are UTC, and measures the real bar and tick history depth;
- re-requests any missing or partial session inside the window. That covers 2026-09-10 (R03) and the
  2026-10-21 partial day with no manual file deletion. Sep 10 must be fetched while it is inside the ~140-day
  window, i.e. **before late January 2027**;
- pulls ticks for a fixed tick universe (`reference/tick_universe.json`, 12 names: the 3 existing names
  plus the most liquid by turnover), so tick days accumulate on the same names;
- stops cleanly on a Bloomberg data-limit message. The next run resumes from the gaps.

**3. Accumulate ticks (R05).** Bloomberg serves about 1 month of ticks, so 40 full days needs **at least 2 pulls
about 4 weeks apart**. Re-run the notebook **weekly**, on the same Bloomberg PC folder.

**4. Merge, don't replace.** Each run writes `bbg_data_delta_<run>.zip` with only the files it changed.
Unzip it **into** `data/is2/` (`unzip -o bbg_data_delta_<run>.zip -d data/is2/`). New bar chunks are named
`<first>_<last>__p<run>.parquet`, so they never overwrite older chunks, and the loaders keep the newest rows.
Replacing the folder would discard data that has rolled out of Bloomberg's window.

**5. Rebuild and gate:**
```bash
uv run hftf data build-features
uv run hftf data audit --tier research --strict
```
Re-read `docs/DATA_AUDIT.md` if new quirks show up, e.g. R06 failing on a new stock means a condition
code we haven't seen yet.

## Optional (not gated)
- `bbg.record_depth(limit=10, minutes=375)` during market hours, if entitled: L2 depth for future impact
  calibration.
- `bbg.snapshot_options()` daily: an IV history. Options are out of scope for v1.
