"""Rule-based execution baselines (ARCHITECTURE §7). Each maps the env state to child sizes in shares
using only what is observable at the decision (remaining inventory, the causal profile, past bars)."""

from __future__ import annotations

import numpy as np

from hftf_sim.env import BatchExecutionEnv


def remaining_steps(env: BatchExecutionEnv) -> np.ndarray:
    return np.maximum(env.ep.horizon - env.k, 1)


class Immediate:
    """Everything at t0 as one marketable order."""

    name = "immediate"

    def __call__(self, env: BatchExecutionEnv) -> np.ndarray:
        return env.remaining.copy()


class TWAP:
    """Equal slices of the remaining inventory over the remaining steps."""

    name = "twap"

    def __call__(self, env: BatchExecutionEnv) -> np.ndarray:
        return env.remaining / remaining_steps(env)


class VWAP:
    """Adaptive VWAP on the expected profile: the pace action a = 1 (participation cap applies)."""

    name = "vwap"

    def __call__(self, env: BatchExecutionEnv) -> np.ndarray:
        return env.pace_to_shares(np.ones(env.n))


class PassiveVWAP:
    """Adaptive VWAP that posts a fixed fraction `f` of every child passively at the near touch (limit
    orders, ARCHITECTURE §4.9): the naive reference for a learned passive fraction."""

    action_space = "pace"

    def __init__(self, fraction: float = 0.5) -> None:
        self.fraction = fraction
        self.name = f"vwap_passive{round(fraction * 100)}"

    def __call__(self, env: BatchExecutionEnv) -> np.ndarray:
        return np.stack([np.ones(env.n), np.full(env.n, self.fraction)], axis=1)


class POV:
    """A fixed fraction of the last completed bar's volume (the next bar's volume is not yet known)."""

    def __init__(self, rate: float = 0.10) -> None:
        self.rate = rate
        self.name = f"pov{round(rate * 100)}"

    def __call__(self, env: BatchExecutionEnv) -> np.ndarray:
        row = env.ep.row0 + np.minimum(env.k, env.ep.horizon - 1)
        return np.minimum(self.rate * env.market.volume[env.ep.session, row], env.remaining)


def ac_trajectory(kappa: np.ndarray, horizon: np.ndarray) -> np.ndarray:
    """Remaining-inventory fractions x_j, j = 0..max(H), of the discrete Almgren-Chriss solution
    `x_j = sinh(κ(H − j)) / sinh(κH)` (TWAP, `x_j = 1 − j/H`, in the limit κ → 0). Padded with 0 after H."""
    kappa = np.asarray(kappa, dtype=np.float64)[:, None]
    h = np.asarray(horizon)[:, None]
    j = np.arange(int(h.max()) + 1)[None, :]
    left = np.clip(h - j, 0, None)
    small = kappa < 1e-8
    k = np.where(small, 1.0, kappa)
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.where(small, left / h, np.sinh(k * left) / np.sinh(k * h))
    return np.nan_to_num(x, nan=0.0)


def ac_kappa(eta_lin: np.ndarray, lam_step: np.ndarray) -> np.ndarray:
    """κ from `cosh κ = 1 + λ̃ / (2η̃)` (unit step), with η̃ the linear temporary-impact cost and λ̃ the
    per-step risk penalty, both in bps for a trade or holding of the whole order."""
    return np.arccosh(1.0 + lam_step / (2.0 * np.maximum(eta_lin, 1e-12)))


def ac_expected_cost(
    x: np.ndarray, half_spread: float, eta_lin: float, perm_lin: float, lam_step: float
) -> float:
    """Expected cost (bps of arrival notional) of a remaining-inventory path `x` (x_0 = 1, x_H = 0) under
    linear impact on a flat market: half spread + `η̃·Σn²` + permanent `g·Σ_{u<k} n_u·n_k` + `λ̃·Σ x_{k+1}²`.
    Equals Almgren-Chriss `E[cost] = ½gX² + εX + (η̃ − ½g)Σn²` plus the risk term."""
    x = np.asarray(x, dtype=np.float64)
    n = -np.diff(x)
    earlier = np.concatenate([[0.0], np.cumsum(n)[:-1]])
    return float(
        half_spread + eta_lin * (n**2).sum() + perm_lin * (n * earlier).sum() + lam_step * (x[1:] ** 2).sum()
    )


class AlmgrenChriss:
    """Static AC schedule from the config's impact and risk aversion, linearised at the TWAP rate:
    `η̃ = 1e4·η·σ̂·(Q/(H·V̂))^β·H`, `λ̃ = λ·(σ̂·1e4)²`, with `σ̂` observed at t0 and `V̂` the expected minute
    volume. The child at step k brings the inventory back onto `Q·x_{k+1}`."""

    name = "almgren_chriss"

    def __init__(self) -> None:
        self._target: np.ndarray | None = None
        self._episodes = None

    def _plan(self, env: BatchExecutionEnv) -> np.ndarray:
        ep, mk, imp = env.ep, env.market, env.cfg.impact
        sigma = mk.sigma[ep.session, ep.row0]
        cp = mk.cum_prof
        v_hat = (cp[ep.session, ep.row0 + ep.horizon + 1] - cp[ep.session, ep.row0 + 1]) / ep.horizon
        eta_lin = 1e4 * imp.eta * sigma * (ep.qty / (ep.horizon * np.maximum(v_hat, 1.0))) ** imp.beta
        eta_lin = eta_lin * ep.horizon
        lam_step = env.cfg.reward.risk_aversion * (sigma * 1e4) ** 2
        return ac_trajectory(ac_kappa(eta_lin, lam_step), ep.horizon)

    def __call__(self, env: BatchExecutionEnv) -> np.ndarray:
        if self._episodes is not env.ep or (env.k == 0).all():
            self._target, self._episodes = self._plan(env), env.ep
        nxt = np.minimum(env.k + 1, self._target.shape[1] - 1)
        target_left = env.ep.qty * self._target[np.arange(env.n), nxt]
        return np.clip(env.remaining - target_left, 0.0, None)


def default_baselines() -> list:
    return [Immediate(), TWAP(), VWAP(), POV(0.10), AlmgrenChriss()]
