# CLAUDE.md

Read `project_context.md` first (domain, data dictionary, invariants). The binding design is in
`docs/ARCHITECTURE.md`; data quirks are in `docs/DATA_AUDIT.md`.

## Commands
- `uv sync`: install the workspace (CPU torch from the PyTorch CPU index)
- `uv run pytest`: tests | `uv run ruff check . && uv run ruff format --check .`: lint
- `uv run hftf data audit [--tier pilot|research] [--strict]`: validate the raw pull and run the readiness checks (`docs/DATA_READINESS.md`)
- `uv run hftf data build-features`: → `data/processed/features_1m_v2.parquet`

## Rules
- Causality: model inputs must be observable at `decision_time`; scalers fit on the train split only.
- The eFNN imbalance input is `imb_bar_proxy` on all splits (not `l1_imb`).
- `data/` is licensed Bloomberg data: never commit it or print large chunks of it.
- Code lives in the right workspace package (see ARCHITECTURE §8). Notebooks hold no reusable logic.
- When a design decision changes, update `docs/ARCHITECTURE.md` (decision log) in the same change.
