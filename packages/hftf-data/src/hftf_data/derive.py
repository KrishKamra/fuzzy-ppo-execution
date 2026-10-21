"""Quantities missing from the Bloomberg pull that can be reconstructed from what was delivered."""

from __future__ import annotations

import numpy as np
import pandas as pd

# NSE cash-equity tick size by price band (price-band regime introduced 2025; NSE reviews the band
# per stock monthly from the previous month's closing price). Verified against our trade prints:
# ADE/ADSEZ (~1700-3100) trade on a 0.10 grid, APHS (~7600-9000) on a 0.50 grid.
NSE_TICK_BANDS: tuple[tuple[float, float], ...] = (
    (250.0, 0.01),
    (1_000.0, 0.05),
    (5_000.0, 0.10),
    (10_000.0, 0.50),
    (20_000.0, 1.00),
    (np.inf, 5.00),
)
CANDIDATE_TICKS = (5.0, 1.0, 0.5, 0.1, 0.05, 0.01)


def nse_tick_size(price: float) -> float:
    for upper, tick in NSE_TICK_BANDS:
        if price < upper:
            return tick
    raise ValueError(price)


def infer_tick_size(prices: pd.Series | np.ndarray, coverage: float = 0.995) -> float:
    """Largest candidate tick such that >= `coverage` of observed prices lie on its grid."""
    paise = np.round(np.asarray(prices, dtype=float) * 100).astype(np.int64)
    paise = paise[paise > 0]
    if paise.size == 0:
        return float("nan")
    for tick in CANDIDATE_TICKS:
        step = round(tick * 100)
        if np.mean(paise % step == 0) >= coverage:
            return tick
    return 0.01


def bar_imbalance_proxy(bid_bars: pd.DataFrame, ask_bars: pd.DataFrame) -> pd.Series:
    """Minute-level L1 size imbalance reconstructed from BID/ASK bars alone.

    A BID (ASK) bar's `volume` is the sum of displayed sizes over its `numEvents` quote updates, so
    volume/numEvents is the mean displayed size at the touch. (b - a)/(b + a) on those means tracks the
    tick-derived L1 imbalance with Pearson 0.77-0.85 on the 20 overlapping days, which makes it the
    fallback for the bar-only days (Jun 3 - Sep 18).
    """
    b = bid_bars.set_index("time")
    a = ask_bars.set_index("time")
    bsz = b["volume"] / b["numEvents"].replace(0, np.nan)
    asz = a["volume"] / a["numEvents"].replace(0, np.nan)
    bsz, asz = bsz.align(asz, join="outer")
    return ((bsz - asz) / (bsz + asz)).rename("imb_bar_proxy")
