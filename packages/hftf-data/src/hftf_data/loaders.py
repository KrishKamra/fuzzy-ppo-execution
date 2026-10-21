"""Readers for the Bloomberg pull (data/is2/bbg_data). They only read and normalise.

Derivations live in derive.py, ticks.py and features.py.

Known raw-data quirks handled here:
  * Bloomberg returns -2.424536e-14 as an N/A sentinel in some numeric fields (index weights, EPS).
  * Daily index files contain weekend/holiday rows with every price NaN.
  * Intraday chunks are split per 5 calendar days; '.empty' marker files mean "requested, nothing returned".
  * BID/ASK bars report price 0.0 during the pre-open auction (no visible book).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hftf_data.calendar import IST
from hftf_data.paths import DataPaths, safe_name

BBG_NA_SENTINEL = -2.424536e-14
PRICE_COLS = ("open", "high", "low", "close")


def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    out = df.copy()
    out[num] = out[num].mask(np.isclose(out[num], BBG_NA_SENTINEL, rtol=0, atol=1e-16))
    return out


def list_securities(kind: str, event: str = "TRADE", paths: DataPaths | None = None) -> list[str]:
    """kind: 'bars' or 'ticks'. Returns on-disk folder names that hold at least one parquet file."""
    p = paths or DataPaths()
    root = p.bars / event if kind == "bars" else p.ticks
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.glob("*.parquet")))


def load_bars(security: str, event: str = "TRADE", paths: DataPaths | None = None) -> pd.DataFrame:
    """All 1-minute bars of one event type for one security, deduplicated and time-sorted (UTC)."""
    p = paths or DataPaths()
    files = sorted((p.bars / event / safe_name(security)).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time", keep="last").sort_values("time", ignore_index=True)
    if event in ("BID", "ASK"):
        df[list(PRICE_COLS)] = df[list(PRICE_COLS)].where(df[list(PRICE_COLS)] > 0)
    df["date"] = df["time"].dt.tz_convert(IST).dt.date
    return df


def tick_days(security: str, paths: DataPaths | None = None) -> list[str]:
    p = paths or DataPaths()
    return sorted(f.stem for f in (p.ticks / safe_name(security)).glob("*.parquet"))


def load_ticks(security: str, day: str, paths: DataPaths | None = None) -> pd.DataFrame:
    """Raw ticks for one security-day (day as YYYYMMDD). Columns: time,type,value,size,conditionCodes,..."""
    p = paths or DataPaths()
    df = pd.read_parquet(p.ticks / safe_name(security) / f"{day}.parquet")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time", kind="mergesort", ignore_index=True)


def load_daily(asset_class: str, paths: DataPaths | None = None) -> pd.DataFrame:
    """Daily history for one asset class with sentinel cleanup and all-NaN price rows dropped."""
    p = paths or DataPaths()
    df = clean_sentinels(pd.read_parquet(p.daily / f"daily_{asset_class}.parquet"))
    price_cols = [c for c in ("PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST") if c in df.columns]
    df = df.dropna(subset=price_cols, how="all")
    return df.sort_values(["security", "date"], ignore_index=True)


def load_static(asset_class: str, paths: DataPaths | None = None) -> pd.DataFrame:
    p = paths or DataPaths()
    return clean_sentinels(pd.read_parquet(p.reference / f"static_{asset_class}.parquet"))
