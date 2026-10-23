"""Running observation normalisation and return-based reward scaling (fit during training only)."""

from __future__ import annotations

import numpy as np

CLIP = 10.0


class RunningMeanStd:
    """Parallel-update running mean and variance (Chan et al.)."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        mean, var, n = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta, total = mean - self.mean, self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + var * n + delta**2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -CLIP, CLIP)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    @classmethod
    def from_state(cls, state: dict) -> RunningMeanStd:
        rms = cls()
        rms.mean, rms.var, rms.count = np.asarray(state["mean"]), np.asarray(state["var"]), state["count"]
        return rms


class RewardScaler:
    """Divides rewards by the running std of the discounted return-to-date (reset at episode ends)."""

    def __init__(self, n_envs: int, gamma: float) -> None:
        self.ret = np.zeros(n_envs)
        self.gamma = gamma
        self.rms = RunningMeanStd()

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + reward
        self.rms.update(self.ret)
        self.ret[done] = 0.0
        return reward / np.sqrt(self.rms.var + 1e-8)
