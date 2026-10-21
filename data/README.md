# data/

Git-ignored. Bloomberg data is licensed and must not be committed or redistributed.

| Path | What | Written by |
|---|---|---|
| `is2/bbg_data/` | raw Bloomberg pull (read-only) | `is2/bbg_collector.py` / `is2/bloomberg_data_collection.ipynb` on a Terminal PC |
| `processed/features_1m_v2.parquet` | corrected causal 1-min feature table | `uv run hftf data build-features` |

Set `HFTF_RAW_DIR` to read a raw pull from somewhere else. Read `docs/DATA_AUDIT.md` before using any raw file.
`is2/bbg_data/features/features_1m.parquet` is defective (see DATA_AUDIT F1-F3). Do not use it.
