"""Parametric impact (ARCHITECTURE §4.4): square-root temporary + Obizhaeva-Wang transient.

All quantities are fractions of the historical mid, so the executed price is
`p = m · (1 + side · (D + s/2 + η·σ·(q/V)^β))`, where `D` is the decayed transient displacement
from our earlier children: `D_t = Σ_{u<t} γ·σ_u·(q_u/V_u)^β · exp(−(t−u)/τ_res)`.
"""

from __future__ import annotations

import numpy as np

from hftf_sim.config import ImpactConfig


class ImpactModel:
    """Per-episode impact parameters (one value per env) and the transient state update."""

    def __init__(self, cfg: ImpactConfig, n: int) -> None:
        self.cfg = cfg
        self.beta = cfg.beta
        self.eta = np.full(n, cfg.eta)
        self.gamma = np.full(n, cfg.gamma)
        self.tau = np.full(n, cfg.tau_res_min)

    def draw(self, idx: np.ndarray, seeds: np.ndarray, randomise: bool) -> None:
        """Set the parameters of envs `idx` for new episodes: the defaults, or (domain randomisation) uniform
        draws from each episode's own seed, so every policy sees the same impact for the same episode."""
        params = (("eta", self.eta), ("gamma", self.gamma), ("tau_res_min", self.tau))
        for name, attr in params:
            attr[idx] = getattr(self.cfg, name)
        if not randomise:
            return
        for i, seed in zip(idx, seeds, strict=True):
            u = np.random.default_rng(seed).uniform(size=len(params))
            for (name, attr), ui in zip(params, u, strict=True):
                if name in self.cfg.randomise:
                    lo, hi = self.cfg.randomise[name]
                    attr[i] = lo + (hi - lo) * ui

    def scaled_size(self, q: np.ndarray, volume: np.ndarray, expected: np.ndarray) -> np.ndarray:
        """`(q/V_eff)^β` with `V_eff = max(V, floor · expected volume, 1)`."""
        v_eff = np.maximum(np.maximum(volume, self.cfg.volume_floor_frac * expected), 1.0)
        return (q / v_eff) ** self.beta

    def temporary(self, sigma: np.ndarray, size: np.ndarray) -> np.ndarray:
        return self.eta * sigma * size

    def transient_kick(self, sigma: np.ndarray, size: np.ndarray) -> np.ndarray:
        return self.gamma * sigma * size

    def decay(self) -> np.ndarray:
        """Per-step (1 min) decay factor of the transient displacement."""
        return np.exp(-1.0 / self.tau)
