"""Data audit: coverage, integrity and reconciliation checks over the raw Bloomberg pull.

Produces a machine-readable dict (written to docs/data/data_audit.json by the CLI). The narrative
interpretation lives in docs/DATA_AUDIT.md.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from hftf_data.calendar import continuous_mask
from hftf_data.derive import bar_imbalance_proxy, infer_tick_size, nse_tick_size
from hftf_data.loaders import list_securities, load_bars, load_daily, load_static, load_ticks, tick_days
from hftf_data.paths import DataPaths
from hftf_data.ticks import clean_quotes, clean_trades


def _trading_days(paths: DataPaths) -> set:
    idx = load_daily("index", paths)
    return set(idx.loc[idx["security"] == "NIFTY Index", "date"].dt.date)


def bar_coverage(paths: DataPaths) -> list[dict]:
    sessions = _trading_days(paths)
    rows = []
    for event in ("TRADE", "BID", "ASK"):
        root = paths.bars / event
        for d in sorted(root.iterdir()) if root.exists() else []:
            n_empty = len(list(d.glob("*.empty")))
            b = load_bars(d.name, event, paths)
            if b.empty:
                rows.append({"security": d.name, "event": event, "rows": 0, "empty_chunks": n_empty})
                continue
            days = set(b["date"])
            window = {s for s in sessions if min(days) <= s <= max(days)}
            rows.append(
                {
                    "security": d.name,
                    "event": event,
                    "rows": len(b),
                    "days": len(days),
                    "first": str(min(days)),
                    "last": str(max(days)),
                    "missing_sessions": sorted(str(x) for x in window - days),
                    "zero_price_rows": int(b["close"].isna().sum()) if event != "TRADE" else 0,
                    "empty_chunks": n_empty,
                }
            )
    return rows


def tick_reconciliation(paths: DataPaths) -> list[dict]:
    """Per security-day: cleaned continuous tick volume vs continuous-session TRADE bar volume."""
    rows = []
    for sec in list_securities("ticks", paths=paths):
        bars = load_bars(sec, "TRADE", paths)
        bars = bars[continuous_mask(bars["time"])]
        bar_vol = bars.groupby("date")["volume"].sum()
        for day in tick_days(sec, paths):
            t = load_ticks(sec, day, paths)
            q = clean_quotes(t)
            tr = clean_trades(t, q)
            d = pd.Timestamp(day).date()
            raw_tr = t[t["type"] == "TRADE"]
            rows.append(
                {
                    "security": sec,
                    "day": str(d),
                    "last_tick_ist": t["time"].max().tz_convert("Asia/Kolkata").strftime("%H:%M:%S"),
                    "raw_rows": len(t),
                    "raw_trade_volume": int(raw_tr["size"].sum()),
                    "clean_trade_volume": int(tr["size"].sum()),
                    "bar_volume": float(bar_vol.get(d, np.nan)),
                    "clean_over_bar": round(float(tr["size"].sum() / bar_vol.get(d, np.nan)), 4),
                    "quote_states": len(q),
                    "crossed_after_clean": int((q["bid"] >= q["ask"]).sum()),
                    "unsigned_trade_frac": round(float((tr["sign"] == 0).mean()), 4) if len(tr) else None,
                    "condition_codes": dict(Counter(raw_tr["conditionCodes"].fillna("<none>"))),
                }
            )
    return rows


def imbalance_proxy_check(paths: DataPaths) -> list[dict]:
    from hftf_data.ticks import minute_microstructure

    out = []
    for sec in list_securities("ticks", paths=paths):
        proxy = bar_imbalance_proxy(load_bars(sec, "BID", paths), load_bars(sec, "ASK", paths))
        tick = pd.concat(
            [minute_microstructure(load_ticks(sec, d, paths))["l1_imb_mean"] for d in tick_days(sec, paths)]
        )
        j = pd.concat([proxy, tick], axis=1, join="inner").dropna()
        out.append(
            {
                "security": sec,
                "n_minutes": len(j),
                "pearson": round(float(j.corr().iloc[0, 1]), 4),
                "spearman": round(float(j.corr("spearman").iloc[0, 1]), 4),
            }
        )
    return out


def tick_size_check(paths: DataPaths) -> list[dict]:
    out = []
    for sec in load_static("equity", paths)["security"]:
        b = load_bars(sec, "TRADE", paths)
        if b.empty:
            continue
        px = b["close"]
        out.append(
            {
                "security": sec,
                "inferred_tick": infer_tick_size(px),
                "nse_rule_tick_at_median_px": nse_tick_size(float(px.median())),
                "median_px": round(float(px.median()), 2),
            }
        )
    return out


def daily_field_coverage(paths: DataPaths) -> dict:
    out = {}
    for cls in ("equity", "index", "futures", "fx", "rates", "commodity", "options"):
        if not (paths.daily / f"daily_{cls}.parquet").exists():  # options are no longer pulled (v1 scope)
            continue
        df = load_daily(cls, paths)
        out[cls] = {
            "securities": sorted(df["security"].unique()),
            "first": str(df["date"].min().date()),
            "last": str(df["date"].max().date()),
            "rows": len(df),
            "sessions_per_security": df.groupby("security")["date"].nunique().to_dict(),
            "non_null": df.drop(columns=["security", "date", "class"], errors="ignore")
            .notna()
            .mean()
            .round(3)
            .to_dict(),
        }
    return out


def run_audit(paths: DataPaths | None = None) -> dict:
    paths = paths or DataPaths()
    static_eq = load_static("equity", paths)
    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_root": str(paths.raw),
        "intraday_equities": sorted(static_eq["security"]),
        "static_missing_fields": [f for f in ("TICK_SIZE", "TRADING_LOT_SIZE") if f not in static_eq.columns],
        "bars": bar_coverage(paths),
        "ticks": tick_reconciliation(paths),
        "imbalance_proxy": imbalance_proxy_check(paths),
        "tick_size": tick_size_check(paths),
        "daily": daily_field_coverage(paths),
        "tick_window": {
            sec: [tick_days(sec, paths)[0], tick_days(sec, paths)[-1]]
            for sec in list_securities("ticks", paths=paths)
        },
    }
