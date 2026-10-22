# data/

Git-ignored. Bloomberg data is licensed and must not be committed or redistributed.

| Path | What | Written by |
|---|---|---|
| `is2/bbg_data/` | raw Bloomberg pull (read-only, frozen 2026-10-21) | Bloomberg collector, `LIMIT = 3` pass (tooling in git history at `de800d8`) |
| `processed/features_1m_v3.parquet` | **final** causal 1-min feature table, with `split` labels | `uv run hftf data build-features` |
| `processed/features_1m_v3.manifest.json` | sessions per split, dropped and excluded days, fill shares, train-only stats | same |

Set `HFTF_RAW_DIR` to read a raw pull from somewhere else. Read `docs/DATA_AUDIT.md` before using any raw file.
`is2/bbg_data/features/features_1m.parquet` is defective (see DATA_AUDIT F1-F3). Do not use it.
