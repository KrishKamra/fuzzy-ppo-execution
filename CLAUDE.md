# CLAUDE.md

Read `project_context.md` first (domain, data dictionary, invariants). The binding design is in
`docs/ARCHITECTURE.md`; data quirks are in `docs/DATA_AUDIT.md`.

## Commands
- `uv sync`: install the workspace (CPU torch from the PyTorch CPU index)
- `uv run pytest`: tests | `uv run ruff check . && uv run ruff format --check .`: lint
- `uv run hftf data build-features`: → `data/processed/features_1m_v3.parquet` (+ `.manifest.json`), the final dataset
- `uv run hftf data audit [--strict]`: readiness gate R01-R12 over the raw pull and the feature table (`docs/DATA_READINESS.md`)
- Every imputation/transformation is logged in `docs/DATA_TRANSFORMS.md`: update it with any change to `hftf_data`

## Rules
- Causality: model inputs must be observable at `decision_time`; scalers fit on the train split only.
- The eFNN imbalance input is `imb_bar_proxy` on all splits (not `l1_imb`).
- `data/` is licensed Bloomberg data: never commit it or print large chunks of it. The pull is frozen: fix gaps in code, flagged, and never synthesise sessions.
- Code lives in the right workspace package (see ARCHITECTURE §8). Notebooks hold no reusable logic.
- When a design decision changes, update `docs/ARCHITECTURE.md` (decision log) in the same change.
