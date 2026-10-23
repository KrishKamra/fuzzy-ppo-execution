"""Rollout storage and GAE. Episodes end by forced execution, a true terminal, so there is no bootstrap
across a `done` (the autoreset observation after it belongs to the next episode)."""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, steps: int, n_envs: int, obs_dim: int) -> None:
        self.obs = np.zeros((steps, n_envs, obs_dim), dtype=np.float32)
        self.u = np.zeros((steps, n_envs), dtype=np.float32)
        self.logp = np.zeros((steps, n_envs), dtype=np.float32)
        self.value = np.zeros((steps, n_envs), dtype=np.float32)
        self.reward = np.zeros((steps, n_envs), dtype=np.float32)
        self.done = np.zeros((steps, n_envs), dtype=bool)
        self.steps = steps
        self.t = 0

    def add(self, obs, u, logp, value, reward, done) -> None:
        t = self.t
        self.obs[t], self.u[t], self.logp[t], self.value[t] = obs, u, logp, value
        self.reward[t], self.done[t] = reward, done
        self.t += 1

    def gae(self, last_value: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        """Advantages and returns (value targets)."""
        adv = np.zeros_like(self.reward)
        running = np.zeros_like(last_value, dtype=np.float32)
        next_value = last_value
        for t in reversed(range(self.steps)):
            live = 1.0 - self.done[t]
            delta = self.reward[t] + gamma * next_value * live - self.value[t]
            running = delta + gamma * lam * live * running
            adv[t] = running
            next_value = self.value[t]
        return adv, adv + self.value

    def reset(self) -> None:
        self.t = 0
