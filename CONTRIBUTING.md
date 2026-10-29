# Contributing

Thanks for helping. This is a research codebase, so a change is finished only when its tests, its documentation
and (for experiments) its reported results are finished too.

## Set up

```bash
uv sync                                  # Python 3.12, every workspace package, dev tools (CPU torch)
cd web && npm install                    # the web app (Node.js 22)
```

The licensed Bloomberg data and the trained checkpoints are not in the repository. Everything below works
without them: the tests use synthetic data, and the two tests that need the real feature table skip themselves.

## Before you open a pull request

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
cd web && npx tsc --noEmit && npm test && npm run build
```

CI runs the same checks, plus an **API contract** job. If you change `apps/api`, run `npm run gen:api` in
`web/` and commit the regenerated `web/src/api/`.

## Ground rules

These keep the results valid. Most of them are also in [CLAUDE.md](CLAUDE.md) and [project_context.md](project_context.md).

- **Never commit licensed data.** That covers `data/`, checkpoints in `runs/`, and screenshots or figures that
  show prices, ticks or bars. Aggregate statistics (costs in bps) are fine.
- **Causality.** Model inputs must be observable at `decision_time`. Scalers are fitted on the train split only.
- **Never select or tune on the test split.** Model selection uses validation; the test split has been used
  for the final evaluation and one disclosed replication.
- **Paired evaluation.** Compare policies on identical episodes, and report session-cluster CIs and
  Holm-adjusted tests (`hftf eval compare`).
- **Report every experiment** in [docs/RESULTS.md](docs/RESULTS.md) with the command that reproduces it,
  including the ones that did not work.
- **Record design decisions** in the decision log of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and every
  data transformation in [docs/DATA_TRANSFORMS.md](docs/DATA_TRANSFORMS.md).
- **Code lives in the right package** ([ARCHITECTURE §8](docs/ARCHITECTURE.md)). Notebooks hold no reusable logic.

## Style

- Python: ruff (format and lint), type hints, docstrings that state units and assumptions.
- TypeScript: strict mode; types come from the API schema, not hand-written copies.
- Commits: one topic per commit, with a subject such as `(feat): …`, `(fix): …` or `(docs): …` and a body
  that explains why. Credit co-authors with `Co-Authored-By:` lines.
- Figures: regenerate with `uv run hftf report figures` instead of editing images.

## Reporting problems

Use the issue forms (bug, feature, experiment). Report security problems and data leaks privately, as
described in [SECURITY.md](SECURITY.md). Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).
