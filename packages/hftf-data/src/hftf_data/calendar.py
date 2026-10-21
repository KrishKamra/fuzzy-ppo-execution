"""NSE session calendar and time helpers.

All raw timestamps are UTC. NSE cash market (IST, UTC+05:30):
  pre-open order entry 09:00-09:08, order matching 09:08-09:12, buffer to 09:15
  continuous trading    09:15-15:30
  closing session       15:40-16:00 (trades at closing price)
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

IST = "Asia/Kolkata"
PREOPEN_START = dt.time(9, 0)
CONT_START = dt.time(9, 15)
CONT_END = dt.time(15, 30)  # exclusive
SESSION_MINUTES = 375
OPEN_MINUTE_OF_DAY = 9 * 60 + 15

# Weekdays inside the bar window with no NSE session (confirmed against daily index data).
KNOWN_HOLIDAYS_2026 = {
    dt.date(2026, 6, 26),  # Muharram
    dt.date(2026, 9, 14),  # Ganesh Chaturthi
    dt.date(2026, 10, 2),  # Gandhi Jayanti
    dt.date(2026, 10, 20),  # Dussehra
}


def to_ist(ts: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    if isinstance(ts, pd.Series):
        return pd.to_datetime(ts, utc=True).dt.tz_convert(IST)
    return pd.DatetimeIndex(ts).tz_convert(IST)


def continuous_mask(ts_utc: pd.Series) -> pd.Series:
    """True for timestamps inside the continuous session [09:15, 15:30) IST."""
    t = to_ist(ts_utc).dt.time
    return (t >= CONT_START) & (t < CONT_END)


def session_grid(days: list[str] | list[dt.date]) -> pd.DatetimeIndex:
    """1-minute bar-start grid (UTC) covering the continuous session of each day."""
    parts = [
        pd.date_range(
            pd.Timestamp(f"{d} 09:15", tz=IST).tz_convert("UTC"), periods=SESSION_MINUTES, freq="min"
        )
        for d in days
    ]
    if not parts:
        return pd.DatetimeIndex([], tz="UTC")
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]
