"""TickSim (ARCHITECTURE §4.8): the BarSim env with fills replayed against the 1-second L1 grid.

The agent, its observation and its decision clock are BarSim's (one decision per minute from the 1-minute
features); only the fill changes. The child for fill bar `j` is sent at the bar boundary and split into
`sub_slices` orders at offsets `latency_s + ⌊60·i/n⌋` seconds, i.e. grid rows `60j + offset` (DATA_AUDIT
F12: row 60j is the state the agent saw). Each order of `q_i` shares fills against that row's touch:

    p_i = m_i · (1 + side · (D_i + s_i/2 + (x_i/q_i) · η·σ·(x_i/V_i)^β)),    x_i = (q_i − depth_i)⁺

with `depth_i` the displayed size on the side we take (ask for a buy) and `V_i = V/n` the bar volume per
slice. Shares within the touch pay the half spread only; the excess walks the book with BarSim's
square-root term (no depth beyond L1 is known). Each order adds `γ·σ·(q_i/V_i)^β / n` to the transient
displacement `D`, which decays continuously (OW, resilience τ_res), so with one slice and no depth the
fill is BarSim's at the tick mid and spread. Historical prices and depth do not react to our orders
except through `D` (counterfactual replay, as in BarSim).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hftf_data.calendar import SESSION_MINUTES
from hftf_data.paths import REPO_ROOT
from hftf_sim.config import EnvConfig
from hftf_sim.env import BatchExecutionEnv
from hftf_sim.fees import FeeModel
from hftf_sim.limit import queue_passive_fill
from hftf_sim.market import MarketData

SECONDS = SESSION_MINUTES * 60
TRADE_COLS = ("buy_vol", "sell_vol", "px_min", "px_max")


@dataclass
class TickData:
    """1-second L1 arrays of the tick sessions, shape (n_tick_sessions, 21600). `row[s]` maps a MarketData
    session to its tick row, or −1 if the session has no ticks."""

    row: np.ndarray
    mid: np.ndarray
    spread: np.ndarray  # fraction of mid
    bid_sz: np.ndarray
    ask_sz: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    buy_vol: np.ndarray  # prints in each second (limit-order queue model)
    sell_vol: np.ndarray
    px_min: np.ndarray  # NaN in seconds without prints
    px_max: np.ndarray

    @classmethod
    def load(cls, market: MarketData, path: str | Path | None = None) -> TickData:
        path = REPO_ROOT / EnvConfig().tick.path if path is None else Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        cols = ["security", "date", "sec", "bid", "ask", "bid_sz", "ask_sz", *TRADE_COLS]
        return cls.from_frame(market, pd.read_parquet(path, columns=cols))

    @classmethod
    def from_frame(cls, market: MarketData, df: pd.DataFrame) -> TickData:
        keys = market.sessions[["security", "date"]].assign(date=lambda d: pd.to_datetime(d["date"]))
        df = df.assign(date=pd.to_datetime(df["date"]))
        df = df.merge(keys.reset_index(names="session"), on=["security", "date"])
        df = df.sort_values(["session", "sec"])
        sessions = df["session"].unique()
        if (df.groupby("session").size() != SECONDS).any():
            raise ValueError("every tick session needs the full 1-second grid")
        row = np.full(len(market.sessions), -1, dtype=np.int64)
        row[sessions] = np.arange(len(sessions))

        def grid(col: str) -> np.ndarray:
            return df[col].to_numpy(dtype=np.float64).reshape(len(sessions), SECONDS)

        bid, ask = grid("bid"), grid("ask")
        mid = (bid + ask) / 2
        trades = [grid(c) for c in TRADE_COLS]
        return cls(row, mid, (ask - bid) / mid, grid("bid_sz"), grid("ask_sz"), bid, ask, *trades)

    def passive_fill(
        self, post: np.ndarray, side: np.ndarray, tr: np.ndarray, g0: np.ndarray, g_end: np.ndarray
    ) -> np.ndarray:
        """Queue-model fills of `post` shares joining the near touch of tick row `tr` at second `g0` and
        resting through second `g_end` (limit.queue_passive_fill)."""
        price = np.where(side > 0, self.bid[tr, g0], self.ask[tr, g0])
        queue = np.where(side > 0, self.bid_sz[tr, g0], self.ask_sz[tr, g0])
        width = int((g_end - g0).max())
        g = g0[:, None] + np.arange(1, width + 1)[None, :]
        inside = (g <= g_end[:, None]) & (g < SECONDS)
        g = np.minimum(g, SECONDS - 1)
        r = tr[:, None]

        def window(a: np.ndarray, fill: float) -> np.ndarray:
            return np.where(inside, a[r, g], fill)

        return queue_passive_fill(
            post, side, price, queue, window(self.px_min, np.nan), window(self.px_max, np.nan),
            window(self.buy_vol, 0.0), window(self.sell_vol, 0.0),
        )  # fmt: skip

    def session_ids(self, market: MarketData, split: str | None = None) -> np.ndarray:
        """MarketData session ids that have ticks (optionally within one split)."""
        ids = np.flatnonzero(self.row >= 0)
        return ids if split is None else np.intersect1d(ids, market.session_ids(split))


class TickExecutionEnv(BatchExecutionEnv):
    """BarSim with depth-aware fills at the replayed touch; every episode must be on a tick session."""

    def __init__(self, market: MarketData, cfg: EnvConfig, ticks: TickData, **kwargs) -> None:
        super().__init__(market, cfg, **kwargs)
        self.ticks = ticks
        tc = cfg.tick
        self.n_sub = tc.sub_slices
        self.offsets = tc.latency_s + (60 * np.arange(self.n_sub)) // self.n_sub  # seconds into the bar
        self.depth_fills = tc.depth_fills

    def _tick_row(self, session: np.ndarray) -> np.ndarray:
        r = self.ticks.row[session]
        if (r < 0).any():
            raise ValueError("TickSim episodes must be on tick sessions (TickData.session_ids)")
        return r

    def _arrival_mid(self, session: np.ndarray, row0: np.ndarray) -> np.ndarray:
        return self.ticks.mid[self._tick_row(session), 60 * (row0 + 1)]

    def _mark_mid(self, session: np.ndarray, j: np.ndarray) -> np.ndarray:
        return self.ticks.mid[self._tick_row(session), 60 * j + self.offsets[-1]]

    def _passive_fill(self, post: np.ndarray, live: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TickSim: the order joins the back of the displayed queue when it arrives (row 60j + L) and rests
        to the end of the bar (row 60j + 60)."""
        ep, tk = self.ep, self.ticks
        tr, j = self._tick_row(ep.session), self._fill_row()
        g0 = 60 * j + self.offsets[0]
        price = np.where(ep.side > 0, tk.bid[tr, g0], tk.ask[tr, g0])
        filled = tk.passive_fill(post, ep.side, tr, g0, 60 * j + 60)
        return np.where(live, filled, 0.0), price

    def _fill(self, q: np.ndarray, remaining: np.ndarray, disp: np.ndarray, live: np.ndarray):
        """Child `q` as `sub_slices` whole-share orders against the touch. `disp` is the displacement at the
        first order's time; the returned kick is referred back to that time, so the caller's per-minute
        decay ages every order's kick by its true elapsed time."""
        ep, mk, tk, imp = self.ep, self.market, self.ticks, self.impact
        s, j = ep.session, self._fill_row()
        tr, n = self._tick_row(s), self.n_sub
        sigma, volume, prof = mk.sigma[s, j], mk.volume[s, j] / n, mk.prof[s, j] / n
        base = np.floor(q / n)
        notional0 = ep.qty * self.m0
        left, m_prev, d = remaining.copy(), self.m_prev.copy(), disp.copy()
        is_c, mtm, fees, cash, kick = (np.zeros(self.n) for _ in range(5))
        for i, off in enumerate(self.offsets):
            if i:  # decay between this order and the previous one
                d = d * np.exp(-(off - self.offsets[i - 1]) / (60.0 * imp.tau))
            qi = base + (i < q - base * n)  # spread the remainder over the first orders
            g = 60 * j + off
            m, spread = tk.mid[tr, g], tk.spread[tr, g]
            excess = qi
            if self.depth_fills:  # the side we take: the ask for a buy, the bid for a sell
                excess = np.maximum(qi - np.where(ep.side > 0, tk.ask_sz[tr, g], tk.bid_sz[tr, g]), 0.0)
            walk = imp.temporary(sigma, imp.scaled_size(excess, volume, prof))
            temp = np.divide(excess, qi, out=np.zeros(self.n), where=qi > 0) * walk
            price = m * (1.0 + ep.side * (d + spread / 2 + temp))
            is_c += ep.side * (price - self.m0) * qi
            mtm += ep.side * (qi * (price - m) + self.drift * left * (m - m_prev))
            fees += self.fees(qi * price, ep.side, qi > 0)
            cash += qi * price
            k = imp.transient_kick(sigma, imp.scaled_size(qi, volume, prof)) / n
            d = d + k
            kick += k * np.exp((off - self.offsets[0]) / (60.0 * imp.tau))
            left, m_prev = left - qi, m
        avg = np.divide(cash, q, out=m_prev.copy(), where=q > 0)
        costs = {
            "is_bps": is_c / notional0 * 1e4,
            "mtm_bps": mtm / notional0 * 1e4,
            "fees_bps": fees / notional0 * 1e4,
            "risk_bps": self.risk_aversion * (sigma * 1e4) ** 2 * (left / ep.qty) ** 2,
        }
        costs = {k: np.where(live, v, 0.0) for k, v in costs.items()}
        return avg, np.where(live, kick, 0.0), costs


def make_env(
    market: MarketData,
    cfg: EnvConfig,
    ticks: TickData | None = None,
    fees: FeeModel | None = None,
    **kwargs,
) -> BatchExecutionEnv:
    """BarSim, or TickSim when `ticks` is given."""
    if ticks is None:
        return BatchExecutionEnv(market, cfg, fees=fees, **kwargs)
    return TickExecutionEnv(market, cfg, ticks, fees=fees, **kwargs)
