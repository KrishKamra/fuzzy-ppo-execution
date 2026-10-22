"""Parent orders: (session, t0, H, side, Q) sampled per ARCHITECTURE §4.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hftf_data.calendar import SESSION_MINUTES
from hftf_sim.config import EpisodeConfig
from hftf_sim.market import MarketData


def _minute(hhmm: str) -> int:
    """Minutes since 09:15 for an IST clock time."""
    h, m = map(int, hhmm.split(":"))
    return (h - 9) * 60 + m - 15


@dataclass
class Episodes:
    """A batch of parent orders. `row0` is the row observed at t0 (decision at minute row0 + 1); the
    children fill in rows row0 + 1 … row0 + H, and `mid[row0]` is the arrival price."""

    session: np.ndarray
    row0: np.ndarray
    horizon: np.ndarray
    side: np.ndarray  # +1 buy, -1 sell
    qty: np.ndarray  # parent size Q in shares (integer valued)
    seed: np.ndarray  # per-episode seed (impact draws), so policies see identical episodes

    def __len__(self) -> int:
        return len(self.session)

    def take(self, idx: np.ndarray) -> Episodes:
        return Episodes(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))


class EpisodeSampler:
    """Uniform over the split's sessions, horizons, `t0 ∈ [earliest_start, latest_end − H]` and sides;
    `Q = ρ · Σ expected volume over the fill bars`, `ρ ~ LogUniform`."""

    def __init__(self, market: MarketData, cfg: EpisodeConfig, split: str | None = "train") -> None:
        self.market = market
        self.cfg = cfg
        self.sessions = market.session_ids(split)
        if len(self.sessions) == 0:
            raise ValueError(f"no sessions in split {split!r}")
        self.horizons = np.asarray(cfg.horizons_min)
        self.first_decision = _minute(cfg.earliest_start)
        self.last_end = _minute(cfg.latest_end)
        if self.first_decision < 1 or self.last_end > SESSION_MINUTES:
            raise ValueError("episode window outside the 09:15-15:15 grid")
        self.sides = np.array([1 if s == "buy" else -1 for s in cfg.sides])

    def sample(self, n: int, rng: np.random.Generator) -> Episodes:
        session = rng.choice(self.sessions, size=n)
        horizon = rng.choice(self.horizons, size=n)
        t0 = rng.integers(self.first_decision, self.last_end - horizon + 1)  # decision minute
        row0 = t0 - 1
        lo, hi = np.log(self.cfg.participation_loguniform)
        rho = np.exp(rng.uniform(lo, hi, size=n))
        cp = self.market.cum_prof
        expected = cp[session, row0 + horizon + 1] - cp[session, row0 + 1]
        qty = np.maximum(np.rint(rho * expected), 1.0)
        side = rng.choice(self.sides, size=n)
        seed = rng.integers(0, 2**31 - 1, size=n)
        return Episodes(session, row0, horizon, side, qty, seed)
