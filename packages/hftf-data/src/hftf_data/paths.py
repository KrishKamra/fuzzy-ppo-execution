"""Filesystem layout. Override the raw root with HFTF_RAW_DIR if the Bloomberg pull lives elsewhere."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class DataPaths:
    raw: Path = Path(os.environ.get("HFTF_RAW_DIR", REPO_ROOT / "data" / "is2" / "bbg_data"))
    processed: Path = REPO_ROOT / "data" / "processed"
    reports: Path = REPO_ROOT / "docs" / "data"

    @property
    def bars(self) -> Path:
        return self.raw / "intraday_bars" / "1m"

    @property
    def ticks(self) -> Path:
        return self.raw / "ticks"

    @property
    def daily(self) -> Path:
        return self.raw / "daily"

    @property
    def reference(self) -> Path:
        return self.raw / "reference"


def safe_name(security: str) -> str:
    """'ADE IS Equity' -> 'ADE_IS_Equity' (the collector's on-disk folder naming)."""
    return "_".join(security.replace("/", " ").split())
