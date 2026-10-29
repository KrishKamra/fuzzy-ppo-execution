# CLAUDE.md

Read `project_context.md` first (domain, data dictionary, invariants). The binding design is in
`docs/ARCHITECTURE.md`; data quirks are in `docs/DATA_AUDIT.md`.

## Commands
- `uv sync`: install the workspace (CPU torch from the PyTorch CPU index)
- `uv run pytest`: tests | `uv run ruff check . && uv run ruff format --check .`: lint
- `uv run hftf data build-features`: → `data/processed/features_1m_v3.parquet` (+ `.manifest.json`), the final dataset
- `uv run hftf data audit [--strict]`: readiness gate R01-R12 over the raw pull and the feature table (`docs/DATA_READINESS.md`)
- `uv run hftf eval baselines [--split val] [--randomise-impact]`: paired baseline backtest in `hftf-sim`
- `uv run hftf train ppo --run NAME [--seed N]` → `runs/NAME/`; `uv run hftf eval policy runs/NAME/best.pt --split val`
- `--policy fuzzy [--no-evolve] [--antecedent-grad value_only] [--gate hard]` trains eFNN / ANFIS / ablations; `--hold-out SEC` leaves a stock out; `uv run hftf eval rules CKPT` gives the rule report and rule-regime AMI
- `uv run hftf eval compare mlp='runs/m4_mlp_s*/best.pt' efnn='runs/m4_efnn_s*/best.pt' --vs mlp [--by-regime] [--by security,horizon] [--security SEC]`: seed ensembles, session-cluster CIs with Holm-adjusted p and date-cluster CIs; `hftf eval robustness ... --param eta` sweeps impact
- M5: `uv run hftf data build-l1` → `data/processed/l1_1s_v1.parquet` (1-second L1 grid of the tick sessions); `--sim tick` on `eval baselines|policy|compare|robustness` replays fills against it (TickSim, val/test tick sessions only; `--sim bar --tick-sessions` for paired BarSim); `train ppo --limit` adds the passive-fraction action (dead zone `limit.dead_zone`; the passive head starts at the always-passive rule, `configs/ppo.yaml init_passive`); `uv run hftf eval calibrate-limit` fits BarSim's passive-fill caps on val ticks (never test)
- M6: `uv run hftf serve` runs the demo API on 127.0.0.1:8000 (`configs/demo.yaml`; OpenAPI at /docs; `apps/api`); `uv run hftf eval restat ROWS --groups mlp,anfis,efnn --vs mlp` recomputes report tables from saved rows; frontend `web/` per `docs/DEMO_PLAN.md` and `docs/UI_DESIGN.md`
- Training reward is the expected-drift Almgren-Chriss cost (`env.yaml reward.price_drift: expected`); evaluation always charges realised IS
- Results go in `docs/RESULTS.md` (fresh paired episodes, all seeds; never the selection set)
- Every imputation/transformation is logged in `docs/DATA_TRANSFORMS.md`: update it with any change to `hftf_data`

## Rules
- Causality: model inputs must be observable at `decision_time`; scalers fit on the train split only.
- The eFNN imbalance input is `imb_bar_proxy` on all splits (not `l1_imb`).
- `data/` is licensed Bloomberg data: never commit it or print large chunks of it. The pull is frozen: fix gaps in code, flagged, and never synthesise sessions.
- Code lives in the right workspace package (see ARCHITECTURE §8). Notebooks hold no reusable logic.
- When a design decision changes, update `docs/ARCHITECTURE.md` (decision log) in the same change.
