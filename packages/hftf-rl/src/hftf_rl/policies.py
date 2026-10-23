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

from hftf_fuzzy.layer import FuzzyTSK

U_EPS = 1e-6


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _layer(n_in: int, n_out: int, gain: float) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _trunk(obs_dim: int, hidden: list[int], activation: str) -> tuple[nn.Sequential, int]:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers, n_in = [], obs_dim
    for n in hidden:
        layers += [_layer(n_in, n, math.sqrt(2)), act()]
        n_in = n
    return nn.Sequential(*layers), n_in


def _beta_bias(a_max: float, init_pace: float, init_concentration: float) -> torch.Tensor:
    m = init_pace / a_max
    if min(m, 1 - m) * init_concentration <= 1.0:
        raise ValueError("init_pace/init_concentration give a Beta with α or β ≤ 1")
    return torch.tensor(
        [_inv_softplus(m * init_concentration - 1), _inv_softplus((1 - m) * init_concentration - 1)]
    )


class _PaceActorCritic(nn.Module):
    """Shared sampling helpers; subclasses implement `forward(obs) -> (Beta, V)`."""

    kind = ""
    a_max = 4.0

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)[1]

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


class MLPActorCritic(_PaceActorCritic):
    """Shared MLP trunk `h = MLP(obs)` with a Beta actor head and a value head (the PPO-MLP baseline)."""

    kind = "mlp"

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
        self.trunk, n_in = _trunk(obs_dim, hidden, activation)
        self.actor = _layer(n_in, 2, 0.01)
        self.critic = _layer(n_in, 1, 1.0)
        with torch.no_grad():
            self.actor.bias.copy_(_beta_bias(a_max, init_pace, init_concentration))

    def forward(self, obs: torch.Tensor) -> tuple[Beta, torch.Tensor]:
        h = self.trunk(obs)
        ab = 1.0 + nn.functional.softplus(self.actor(h))
        return Beta(ab[..., 0], ab[..., 1]), self.critic(h).squeeze(-1)


class FuzzyActorCritic(_PaceActorCritic):
    """MLP trunk `h = MLP(obs)` with fuzzy-gated TSK heads over the regime input `x = obs[:, :regime_dim]`
    (ARCHITECTURE §5.1): `(α, β) = 1 + softplus(Σ_r φ_r (A_r h + b_r))`, `V = Σ_r φ_r (u_r·h + d_r)`.

    `antecedent_grad`: `all` (actor and critic losses shape the MFs), `value_only` (only the critic's Bellman
    loss does: φ is detached in the actor head) or `none` (φ detached everywhere; MFs move only by evolution).
    Every rule starts with the same consequents, so the initial policy is the MLP's noisy VWAP whatever the
    antecedents are, and re-placing them (k-means) is function-preserving.
    """

    kind = "fuzzy"

    def __init__(
        self,
        obs_dim: int,
        hidden: list[int],
        a_max: float,
        activation: str = "tanh",
        init_pace: float = 1.0,
        init_concentration: float = 8.0,
        n_rules: int = 4,
        regime_dim: int = 3,
        antecedent_grad: str = "all",
        width: float = 1.0,
    ) -> None:
        super().__init__()
        if antecedent_grad not in ("all", "value_only", "none"):
            raise ValueError(f"antecedent_grad must be all | value_only | none, got {antecedent_grad!r}")
        self._arch = {
            "obs_dim": obs_dim,
            "hidden": list(hidden),
            "a_max": a_max,
            "activation": activation,
            "init_pace": init_pace,
            "init_concentration": init_concentration,
            "regime_dim": regime_dim,
            "antecedent_grad": antecedent_grad,
            "width": width,
        }
        self.a_max, self.regime_dim, self.antecedent_grad = a_max, regime_dim, antecedent_grad
        self.trunk, n_in = _trunk(obs_dim, hidden, activation)
        self.fuzzy = FuzzyTSK(regime_dim, n_rules, {"actor": 2, "critic": 1}, n_in, width)
        actor, critic = _layer(n_in, 2, 0.01), _layer(n_in, 1, 1.0)
        with torch.no_grad():
            self.fuzzy.actor_W.copy_(actor.weight.expand(n_rules, -1, -1))
            self.fuzzy.actor_b.copy_(_beta_bias(a_max, init_pace, init_concentration).expand(n_rules, -1))
            self.fuzzy.critic_W.copy_(critic.weight.expand(n_rules, -1, -1))

    @property
    def arch(self) -> dict:
        return {**self._arch, "n_rules": self.fuzzy.n_rules}

    def regime(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., : self.regime_dim]

    def forward(self, obs: torch.Tensor) -> tuple[Beta, torch.Tensor]:
        h = self.trunk(obs)
        phi = self.fuzzy.phi(self.regime(obs))
        phi_actor = phi if self.antecedent_grad == "all" else phi.detach()
        phi_critic = phi.detach() if self.antecedent_grad == "none" else phi
        ab = 1.0 + nn.functional.softplus(self.fuzzy.head("actor", h, phi_actor))
        return Beta(ab[..., 0], ab[..., 1]), self.fuzzy.head("critic", h, phi_critic).squeeze(-1)

    @torch.no_grad()
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)


POLICIES = {"mlp": MLPActorCritic, "fuzzy": FuzzyActorCritic}
