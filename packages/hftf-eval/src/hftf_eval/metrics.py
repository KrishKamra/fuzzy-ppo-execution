"""Execution metrics over backtest rows (costs in bps of arrival notional; higher is worse)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cvar(cost: np.ndarray, level: float = 0.95) -> float:
    """Mean of the worst (1 − level) share of costs."""
    cost = np.sort(np.asarray(cost, dtype=np.float64))
    tail = cost[int(np.floor(level * len(cost))) :]
    return float(tail.mean()) if len(tail) else float("nan")


def summarize(df: pd.DataFrame, by: str = "policy") -> pd.DataFrame:
    """IS mean/median/std, CVaR95 of IS, fees, the risk-adjusted objective, VWAP slippage, child count."""
    g = df.groupby(by, sort=False)
    return pd.DataFrame(
        {
            "episodes": g.size(),
            "is_mean": g["is_bps"].mean(),
            "is_median": g["is_bps"].median(),
            "is_std": g["is_bps"].std(),
            "is_cvar95": g["is_bps"].apply(cvar),
            "fees": g["fees_bps"].mean(),
            "objective": g["cost_bps"].mean(),
            "vwap_slip": g["vwap_slip_bps"].mean(),
            "children": g["n_children"].mean(),
        }
    )
