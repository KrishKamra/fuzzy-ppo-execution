"""Actor-critics with a Beta pace head (ARCHITECTURE §4.3, §6).

The policy samples `u ~ Beta(α, β)` on (0, 1) and the env receives the pace `a = a_max · u`.
`(α, β) = 1 + softplus(·)` keeps the density unimodal. Heads are initialised so the starting policy is
a noisy adaptive VWAP (mean pace `init_pace`).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta

U_EPS = 1e-6


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _layer(n_in: int, n_out: int, gain: float) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class MLPActorCritic(nn.Module):
    """Shared MLP trunk `h = MLP(obs)` with a Beta actor head and a value head (the PPO-MLP baseline)."""

    def __init__(
        self,
        obs_dim: int,
        hidden: list[int],
        a_max: float,
        activation: str = "tanh",
        init_pace: float = 1.0,
        init_concentration: float = 8.0,
    ) -> None:
        super().__init__()
        self.arch = {
            "obs_dim": obs_dim,
            "hidden": list(hidden),
            "a_max": a_max,
            "activation": activation,
            "init_pace": init_pace,
            "init_concentration": init_concentration,
        }
        self.a_max = a_max
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        layers, n_in = [], obs_dim
        for n in hidden:
            layers += [_layer(n_in, n, math.sqrt(2)), act()]
            n_in = n
        self.trunk = nn.Sequential(*layers)
        self.actor = _layer(n_in, 2, 0.01)
        self.critic = _layer(n_in, 1, 1.0)
        m = init_pace / a_max
        if min(m, 1 - m) * init_concentration <= 1.0:
            raise ValueError("init_pace/init_concentration give a Beta with α or β ≤ 1")
        with torch.no_grad():
            self.actor.bias.copy_(
                torch.tensor(
                    [
                        _inv_softplus(m * init_concentration - 1),
                        _inv_softplus((1 - m) * init_concentration - 1),
                    ]
                )
            )

    def forward(self, obs: torch.Tensor) -> tuple[Beta, torch.Tensor]:
        h = self.trunk(obs)
        ab = 1.0 + nn.functional.softplus(self.actor(h))
        return Beta(ab[..., 0], ab[..., 1]), self.critic(h).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.trunk(obs)).squeeze(-1)

    @torch.no_grad()
    def sample(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns `u` (clamped inside (0, 1)), log π(u) and V for a batch of normalised observations."""
        dist, v = self(torch.as_tensor(obs, dtype=torch.float32))
        u = dist.sample().clamp(U_EPS, 1 - U_EPS)
        return u.numpy(), dist.log_prob(u).numpy(), v.numpy()

    @torch.no_grad()
    def mean_pace(self, obs: np.ndarray) -> np.ndarray:
        dist, _ = self(torch.as_tensor(obs, dtype=torch.float32))
        return dist.mean.numpy() * self.a_max


POLICIES = {"mlp": MLPActorCritic}
