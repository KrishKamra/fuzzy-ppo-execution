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

On the Bloomberg Terminal PC, in `data/is2/bloomberg_data_collection.ipynb`:

**1. Fix tickers in the collector source.** `bbg_collector.py` is regenerated from notebook cell 4
(`%%writefile`), so edit **that cell**, not the `.py` file.
- `STARTER_EQUITIES`: change `"RIL IN Equity"` to `"RELIANCE IN Equity"` (verify in the Terminal). Check
  `"TTMT IN Equity"` (Tata Motors was demerged; find the current ticker).
- `SEED_CLASSES`: verify or remove `NSENEXT50 Index`, `MIBOR Index`, `RBIRREPO Index`.
- In cell 8, run the probe on a resolving ticker: `bbg.probe("RELIANCE IN Equity")`. The current
  `RIL IN Equity` probe resolves nothing, which is why `capabilities.json` reported no history.

**2. Settings (cell 6):** `LIMIT = None`. Keep `bbg.TICK_UNIVERSE_N = 10` (≥ 10 needed for R05). Run
`bbg.estimate(None)` (cell 11) and confirm it fits the terminal's daily/monthly data limits. If not,
spread the run over several days; finished chunks are skipped on re-run.

**3. Remove the Sep 10 gap (R03).** The collector skips chunks that already exist, so delete them first:
```
bbg_data/intraday_bars/1m/*/*/20260906_20260910.*
```
Then run cell 17 (`collect_bars`). This must happen while Sep 10 is still inside Bloomberg's ~140-day
intraday window, i.e. **before late January 2027**.

**4. Run the full pipeline:** cells 10 → 13 → 15 → 17 → 19 → 21, or `bbg.run_all(None, skip=('probe',))`.
Then run cell 25 to bundle.

**5. Accumulate ticks (R05).** Bloomberg serves about 1 month of ticks, so 40 full days needs **at least 2 pulls
about 4 weeks apart**. Weekly is safer. Run cell 19 (`collect_ticks(None)`) each time.

**6. Merge, don't replace.** Unzip each new bundle **into** the existing `data/is2/bbg_data/`, letting new
files add to old ones. Replacing the folder would discard ticks and bars that have rolled out of
Bloomberg's window. Chunk and day files are uniquely named, so merging is safe.

**7. Rebuild and gate:**
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
