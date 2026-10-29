"""1-second L1 grid for the tick-replay simulator (TickSim, ARCHITECTURE §4.8).

Each tick session becomes 21,600 rows (09:15-15:15 IST, one per second). Row `s` covers the second
[s, s+1) after 09:15 and holds:
  * the L1 state after every quote update stamped in that second (bid, ask, bid_sz, ask_sz), carried
    forward inside the session when the second has no update (`quote_age_s` counts the carried seconds);
  * the continuous-session prints stamped in that second (volume, Lee-Ready buy_vol / sell_vol,
    px_min / px_max). A second without prints has volume 0 and NaN prices: nothing traded.

Clock alignment (DATA_AUDIT F12): Bloomberg's 1-minute bar `m` holds exactly the ticks of grid seconds
60m+1 … 60m+60 (volume and closing quotes agree on 100% of tick minutes). So the L1 state the agent had
seen when deciding at the end of bar m is row 60(m+1), and that row's mid is the bar's `mid`.

Quotes and trades come from `ticks.clean_quotes` / `ticks.clean_trades` (DATA_AUDIT F1, F2). Seconds before
the session's first valid quote stay NaN (never back-filled); the simulator only reads rows from 09:20.
Only sessions kept in the final feature table are built, and each carries that table's `split`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hftf_data.calendar import IST, SESSION_MINUTES
from hftf_data.loaders import load_ticks, tick_days
from hftf_data.paths import DataPaths
from hftf_data.ticks import clean_quotes, clean_trades

L1_FILE = "l1_1s_v1.parquet"
SESSION_SECONDS = SESSION_MINUTES * 60
QUOTE_COLS = ("bid", "ask", "bid_sz", "ask_sz")
TRADE_COLS = ("volume", "buy_vol", "sell_vol", "px_min", "px_max")


def _second(time: pd.Series, day: pd.Timestamp) -> pd.Series:
    start = pd.Timestamp(f"{day.date()} 09:15", tz=IST).tz_convert("UTC")
    return ((time - start) // pd.Timedelta(seconds=1)).astype(np.int64)


def second_grid(ticks: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The 1-second L1 grid of one security-day of raw ticks (index: second of the session)."""
    grid = pd.RangeIndex(SESSION_SECONDS, name="sec")
    quotes = clean_quotes(ticks)
    q = quotes.assign(sec=_second(quotes["time"], day))
    q = q[(q["sec"] >= 0) & (q["sec"] < SESSION_SECONDS)]
    state = q.groupby("sec")[list(QUOTE_COLS)].last().reindex(grid)
    updated = pd.Series(np.arange(SESSION_SECONDS), index=grid).where(state["bid"].notna())
    out = state.ffill()
    out["quote_age_s"] = (grid.to_numpy() - updated.ffill()).to_numpy()  # NaN before the first quote

    tr = clean_trades(ticks, quotes)
    tr = tr.assign(sec=_second(tr["time"], day))
    tr = tr[(tr["sec"] >= 0) & (tr["sec"] < SESSION_SECONDS)]
    tr = tr.assign(
        buy=np.where(tr["sign"] > 0, tr["size"], 0.0), sell=np.where(tr["sign"] < 0, tr["size"], 0.0)
    )
    agg = tr.groupby("sec").agg(
        volume=("size", "sum"),
        buy_vol=("buy", "sum"),
        sell_vol=("sell", "sum"),
        px_min=("price", "min"),
        px_max=("price", "max"),
    )
    agg = agg.reindex(grid)
    for c in ("volume", "buy_vol", "sell_vol"):
        agg[c] = agg[c].fillna(0.0)  # no print in that second: nothing traded (observed, not imputed)
    return out.join(agg).reset_index()


def build_l1_table(sessions: pd.DataFrame, paths: DataPaths | None = None) -> pd.DataFrame:
    """1-second grids of every tick day among `sessions` (columns security, date, split: the sessions of
    the final feature table)."""
    paths = paths or DataPaths()
    frames = []
    for sec, g in sessions.groupby("security", sort=True):
        have = set(tick_days(sec, paths))
        for date, split in zip(g["date"], g["split"], strict=True):
            day = pd.Timestamp(date)
            if day.strftime("%Y%m%d") not in have:
                continue
            f = second_grid(load_ticks(sec, day.strftime("%Y%m%d"), paths), day)
            frames.append(f.assign(security=sec, date=day, split=split))
    out = pd.concat(frames, ignore_index=True)
    lead = ["security", "date", "split", "sec"]
    return out[lead + [c for c in out.columns if c not in lead]]


def l1_manifest(df: pd.DataFrame) -> dict:
    """What the grid holds: sessions per split, seconds before the first quote, carried-quote ages."""
    day = df.drop_duplicates(["security", "date"])
    first = df[df["bid"].notna()].groupby(["security", "date"])["sec"].min()
    age = df.loc[df["split"] != "excluded", "quote_age_s"]  # the excluded partial day ends ~12:05
    return {
        "file": L1_FILE,
        "rows": len(df),
        "grid": f"09:15-15:15 IST, {SESSION_SECONDS} s/session",
        "sessions_by_split": {
            s: g.groupby("security")["date"].nunique().to_dict() for s, g in day.groupby("split")
        },
        "first_quote_sec": {"max": int(first.max()), "median": float(first.median())},
        "quote_age_s": {
            "share_carried": round(float((age > 0).mean()), 4),
            "p99": float(age.quantile(0.99)),
            "max": float(age.max()),
        },
        "trade_volume": float(df["volume"].sum()),
    }
