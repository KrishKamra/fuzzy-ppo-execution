# Data Readiness Gate

"Data ready" has a pass/fail definition. Check it with:

```bash
uv run hftf data build-features     # R11/R12 read the feature table, so build it first
uv run hftf data audit --strict     # exits 1 if any check fails
```

The audit prints a PASS/FAIL table and writes it under `readiness` in `docs/data/data_audit.json`
(git-ignored). Thresholds live in `configs/data_readiness.yaml`. Change them there, not in code. Re-run
the gate after **any** change to `hftf_data`, before training.

**Scope (2026-10-22).** This project runs on a frozen Bloomberg pull: 3 stocks, no further pulls.
The earlier `pilot` and `research` (50-stock) tiers were removed. Research could not be reached by
processing, and pilot is implied by this gate. The thresholds below are what the existing pull supports
honestly, with one acknowledged gap (2026-09-10) that is dropped whole, never imputed.

## Checks

| ID | Requirement | Threshold | Why it matters |
|---|---|---|---|
| R01 | Intraday equities (bars collected) | ≥ 3 | The stock universe |
| R02 | 1-min sessions per equity × {TRADE, BID, ASK} | ≥ 90 | Enough sessions for train/val/test by date |
| R03 | Trading sessions missing inside the bar window, excluding acknowledged gaps | 0 (2026-09-10 acknowledged) | An unexplained gap breaks the volume profile and the splits |
| R04 | Context bars: NIFTY, BANKNIFTY, India VIX | all | Market-context features |
| R05 | Equities with ≥ 20 full tick days | ≥ 3 | Validates the imbalance proxy; tick-day test split |
| R06 | Clean tick volume vs bar volume, full days | ≤ 0.1% error | Proves tick cleaning is correct (DATA_TRANSFORMS T5) |
| R07 | Crossed quotes after cleaning | 0 | Book state is valid (T6) |
| R08 | Imbalance proxy vs tick L1 imbalance (Pearson) | ≥ 0.70 | The proxy is a valid eFNN input (T9) |
| R09 | Inferred tick size = NSE price-band rule | all | Spread-in-ticks and fill prices (T8) |
| R10 | Daily history per equity | ≥ 250 sessions | ADV, daily vol, Amihud features |
| R11 | Feature table covers all equities; eFNN inputs non-null (full days) | ≥ 95% | The table the env reads |
| R12 | Episode rows (train/val/test, from the 09:20 decision) have no NaN in any observed column | 0 NaN | The env never imputes at run time |

A "full tick day" is a day whose ticks reach 15:20 IST. R12's columns are `episode_columns` in the config.
R01-R10 describe the frozen raw pull and can only change if the raw data changes. R11-R12 guard the
feature code.

## Current status (audit of 2026-10-22)

**READY: 12/12.** 3 stocks; 92 usable sessions each (train 60 / val 20 / test 12); 98,256 episode rows with
0 NaN. How each gap in the pull is handled is in [DATA_TRANSFORMS.md](DATA_TRANSFORMS.md).

## Extending the data later (out of scope now)

The retired gap-driven pull notebook (`Bloomberg_training_pull.ipynb`) and the original collector
(`data/is2/bbg_collector.py`, `bloomberg_data_collection.ipynb`) are in git history (`de800d8`). Pull only
what is needed, merge the delta **into** `data/is2/`, then rebuild and re-run this gate.
