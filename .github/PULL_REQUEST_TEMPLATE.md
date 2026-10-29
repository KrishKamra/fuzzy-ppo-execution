## What and why

<!-- One or two sentences: what changes, and the reason. Link the issue (Closes #…). -->

## How it was checked

- [ ] `uv run ruff check . && uv run ruff format --check .` and `uv run pytest`
- [ ] `cd web && npx tsc --noEmit && npm test && npm run build` (if `web/` changed)
- [ ] `npm run gen:api` in `web/` and committed (if the API changed)
- [ ] Results added to `docs/RESULTS.md` with the command that reproduces them (if this is an experiment)
- [ ] `docs/ARCHITECTURE.md` decision log updated (if a design decision changed)
- [ ] `docs/DATA_TRANSFORMS.md` updated (if `hftf_data` changed)

## Data and evaluation hygiene

- [ ] No licensed data, checkpoints or screenshots showing prices are committed
- [ ] Nothing was selected or tuned on the test split
