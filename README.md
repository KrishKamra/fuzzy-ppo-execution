# hftf: Regime-Aware Optimal Execution with Fuzzy PPO

A reinforcement-learning execution agent for NSE equities. **Proximal Policy Optimization (PPO)** is
combined with an **evolving Fuzzy Neural Network (eFNN)**. Order-book imbalance, spread variance and
realized volatility feed a fuzzy layer that **creates, merges and prunes rules** as market regimes shift.
The agent learns how to slice a parent order (a pace relative to a VWAP schedule) to minimise
implementation shortfall: spread, market impact, timing risk and NSE transaction costs. The actor-critic
losses, including the Bellman value loss, also train the fuzzy membership functions.

> **Status:** M0 (workspace, data audit, final feature table `features_1m_v3`, readiness gate READY 12/12)
> M1 (batched execution simulator `hftf-sim`; immediate/TWAP/VWAP/POV/Almgren-Chriss baselines in
> `hftf-eval`) and M2 (PPO-MLP, which beats VWAP by about 0.2 bps; [docs/RESULTS.md](docs/RESULTS.md)) are complete.
> M3 (static ANFIS and evolving fuzzy layer, with KL-guarded add/merge/prune) is complete too. Next is M4: experiments
> and the report. Val gains are not yet reliable (see RESULTS).
> See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#9-milestones).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) ≥ 0.5. Python 3.12 is fetched by uv, and torch installs as
a CPU build.

```bash
uv sync                              # create .venv and install all workspace packages + dev tools
uv run pytest                        # tests
uv run hftf data build-features      # final 1-min feature table -> data/processed/features_1m_v3.parquet (+ manifest)
uv run hftf data audit --strict      # readiness gate R01-R12 (raw pull + feature table)
uv run hftf eval baselines --split val   # TWAP/VWAP/POV/AC/immediate in the simulator (paired episodes)
uv run hftf train ppo --run ppo_mlp_s0   # PPO-MLP, ~2 min on CPU -> runs/ppo_mlp_s0/{best,last}.pt
uv run hftf eval policy runs/ppo_mlp_s0/best.pt --split val   # vs baselines, paired difference to VWAP
uv run hftf train ppo --run efnn_s0 --policy fuzzy   # eFNN (add --no-evolve for static ANFIS)
uv run hftf eval rules runs/efnn_s0/best.pt   # IF imbalance/spread-var/RV THEN pace, with firing shares
```

The raw data is expected at `data/is2/bbg_data/`. Override that with `HFTF_RAW_DIR`. It is licensed
Bloomberg data and is git-ignored. **The dataset is frozen:** no further Bloomberg pulls are planned, and gaps
are handled in `hftf_data.features` (logged in [docs/DATA_TRANSFORMS.md](docs/DATA_TRANSFORMS.md)). Build the features before
the audit, because check R12 reads the table.

## Repository layout (uv workspace monorepo)

| Path | Package | Role |
|---|---|---|
| `packages/hftf-data` | `hftf_data` | loaders, tick cleaning, derived quantities, causal features, audit **(built)** |
| `packages/hftf-sim` | `hftf_sim` | batched execution environment, impact and NSE fee models |
| `packages/hftf-fuzzy` | `hftf_fuzzy` | evolving fuzzy layer (Gaussian MFs, TSK heads, add/merge/prune) |
| `packages/hftf-rl` | `hftf_rl` | PPO trainer, MLP and fuzzy actor-critics, evolution callback |
| `packages/hftf-eval` | `hftf_eval` | TWAP/VWAP/POV/Almgren-Chriss baselines, metrics, paired statistics |
| `apps/cli` | `hftf_cli` | the `hftf` command |
| `configs/` | | YAML for env, fees, eFNN, PPO |
| `docs/` | | architecture, data audit, prompts |

## Documentation

- **[project_context.md](project_context.md)**: domain knowledge, data dictionary, invariants. Start here.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: the binding design and decision log.
- **[docs/DATA_AUDIT.md](docs/DATA_AUDIT.md)**: what the Bloomberg pull contains, its defects, and how each was fixed.
- **[docs/DATA_READINESS.md](docs/DATA_READINESS.md)**: the pass/fail definition of "data ready".
- **[docs/DATA_TRANSFORMS.md](docs/DATA_TRANSFORMS.md)**: every cleaning, imputation and transformation step (T1-T22).
- [docs/prompts/project_context_prompt.md](docs/prompts/project_context_prompt.md): regenerates `project_context.md`.

## Data at a glance

There are 3 Nifty-50 stocks (Adani Enterprises, Adani Ports, Apollo Hospitals), with 96 sessions of
1-minute TRADE/BID/ASK bars (Jun-Oct 2026), 20 sessions of ~1 s conflated L1 ticks, index and futures
context, and 15 years of daily history. The audit found and fixed inflated tick volumes (2-76×),
transiently crossed quotes, and missing tick/lot sizes. It also rebuilt L1 imbalance for the days without
ticks using a bar-level proxy (ρ ≈ 0.77-0.86 against ticks). From 2026-08-03 NSE runs a closing call
auction (15:15-15:30) in this data, so every session is cut to 09:15-15:15. The final table has 92 usable
sessions per stock (train 60, val 20, test 12). 2026-09-10 is missing from the pull and is dropped whole,
not imputed.

## License

Code: [MIT](LICENSE). Market data is © Bloomberg L.P. It is not covered by the license and must not be
redistributed.
