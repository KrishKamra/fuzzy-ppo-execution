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
