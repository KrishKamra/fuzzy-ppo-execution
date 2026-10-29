"""Synthetic sessions for tests: no licensed data needed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hftf_data.calendar import SESSION_MINUTES
from hftf_sim.config import EnvConfig, ImpactConfig
from hftf_sim.episodes import Episodes
from hftf_sim.market import MARKET_COLUMNS, REGIME_COLUMNS, MarketData


def synthetic_frame(n_sessions: int = 2, seed: int = 0, flat: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_sessions * SESSION_MINUTES
    df = pd.DataFrame(
        {
            "security": "TEST IS Equity",
            "date": np.repeat(pd.date_range("2026-06-08", periods=n_sessions), SESSION_MINUTES),
            "mso": np.tile(np.arange(SESSION_MINUTES), n_sessions),
            "split": "train",
        }
    )
    for c in (*REGIME_COLUMNS, *MARKET_COLUMNS):
        df[c] = rng.normal(size=n)
    walk = 0.0 if flat else np.cumsum(rng.normal(0, 5e-4, size=n))
    df["mid"] = 1000.0 * np.exp(walk)
    df["spread_bps"] = 2.0
    df["volume"] = 1000.0 if flat else rng.uniform(200, 2000, size=n)
    df["value"] = df["volume"] * df["mid"]
    df["rv_30m"] = 0.003
    half = df["mid"] * df["spread_bps"] / 2e4
    df["bid"], df["ask"] = df["mid"] - half, df["mid"] + half
    df["h"], df["l"] = df["ask"], df["bid"]  # trades at the touch, never through it
    df["prof_avg_vol"] = 1000.0
    return df


STATS = {c: (0.0, 1.0) for c in (*REGIME_COLUMNS, *MARKET_COLUMNS)}


def synthetic_market(**kw) -> MarketData:
    return MarketData.from_frame(synthetic_frame(**kw), stats=STATS)


def fixed_episodes(n: int, horizon: int = 30, qty: float = 3000.0, side: int = 1, row0: int = 4) -> Episodes:
    full = np.full
    return Episodes(full(n, 0), full(n, row0), full(n, horizon), full(n, side), full(n, qty), np.arange(n))


def linear_cfg(gamma: float = 0.0, lam: float = 0.0) -> EnvConfig:
    """Linear temporary impact; with `gamma > 0` a permanent (non-decaying) impact, as in Almgren-Chriss."""
    cfg = EnvConfig.load()
    cfg.impact = ImpactConfig(beta=1.0, eta=0.5, gamma=gamma, tau_res_min=1e12)
    cfg.reward.risk_aversion = lam
    return cfg


def synthetic_ticks(frame: pd.DataFrame, depth: float = 1e9) -> pd.DataFrame:
    """A 1-second L1 grid consistent with the bars of `synthetic_frame` (DATA_AUDIT F12): second `s` of a
    session carries the closing quote of bar ⌈s/60⌉ − 1, so row 60(m+1) is bar m's close. No prints: tests
    write the trade columns they need."""
    rows = []
    for (sec, date), g in frame.groupby(["security", "date"], sort=False):
        bar = np.maximum((np.arange(SESSION_MINUTES * 60) - 1) // 60, 0)
        mid = g["mid"].to_numpy()[bar]
        half = mid * g["spread_bps"].to_numpy()[bar] / 2e4
        rows.append(
            pd.DataFrame(
                {
                    "security": sec, "date": date, "sec": np.arange(len(bar)),
                    "bid": mid - half, "ask": mid + half, "bid_sz": depth, "ask_sz": depth,
                    "buy_vol": 0.0, "sell_vol": 0.0, "px_min": np.nan, "px_max": np.nan,
                }
            )
        )  # fmt: skip
    return pd.concat(rows, ignore_index=True)
