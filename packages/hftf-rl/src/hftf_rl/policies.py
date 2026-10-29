"""Actor-critics with a Beta pace head (ARCHITECTURE §4.3, §6).

The policy samples `u ~ Beta(α, β)` on (0, 1) and the env receives the pace `a = a_max · u`.
`(α, β) = 1 + softplus(·)` keeps the density unimodal. Heads are initialised so the starting policy is
a noisy adaptive VWAP (mean pace `init_pace`).

With `limit=True` (ARCHITECTURE §4.9) the actor has a second, independent Beta for the passive fraction
`f ∈ (0, 1)` of each child: `u = (u_pace, u_passive)` and the env receives `(a_max · u_pace, u_passive)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Distribution, Independent

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


def _actor_bias(
    a_max: float,
    init_pace: float,
    init_concentration: float,
    limit: bool,
    init_passive: float,
    passive_concentration: float | None,
):
    """Initial actor bias: the pace Beta, and (limit orders) the passive-fraction Beta, whose concentration
    defaults to the pace's. A mean near 1 needs a larger one (both Beta parameters must exceed 1)."""
    bias = _beta_bias(a_max, init_pace, init_concentration)
    if not limit:
        return bias
    conc = init_concentration if passive_concentration is None else passive_concentration
    return torch.cat([bias, _beta_bias(1.0, init_passive, conc)])


class _PaceActorCritic(nn.Module):
    """Shared sampling helpers; subclasses implement `forward(obs) -> (distribution, V)`."""

    kind = ""
    a_max = 4.0
    limit = False

    def _dist(self, ab: torch.Tensor) -> Distribution:
        """Beta over the pace, or (limit orders) independent Betas over pace and passive fraction."""
        if not self.limit:
            return Beta(ab[..., 0], ab[..., 1])
        return Independent(Beta(ab[..., 0::2], ab[..., 1::2]), 1)

    def to_action(self, u: np.ndarray) -> np.ndarray:
        """Unit-interval sample(s) -> env action: the pace, or (pace, passive fraction)."""
        return u * (np.array([self.a_max, 1.0]) if self.limit else self.a_max)

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
        """Deterministic action (Beta means): the pace, or (pace, passive fraction) with limit orders."""
        dist, _ = self(torch.as_tensor(obs, dtype=torch.float32))
        return self.to_action(dist.mean.numpy())


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
        limit: bool = False,
        init_passive: float = 0.2,
        passive_concentration: float | None = None,
    ) -> None:
        super().__init__()
        self.arch = {
            "obs_dim": obs_dim,
            "hidden": list(hidden),
            "a_max": a_max,
            "activation": activation,
            "init_pace": init_pace,
            "init_concentration": init_concentration,
            "limit": limit,
            "init_passive": init_passive,
            "passive_concentration": passive_concentration,
        }
        self.a_max, self.limit = a_max, limit
        self.trunk, n_in = _trunk(obs_dim, hidden, activation)
        self.actor = _layer(n_in, 4 if limit else 2, 0.01)
        self.critic = _layer(n_in, 1, 1.0)
        with torch.no_grad():
            self.actor.bias.copy_(
                _actor_bias(a_max, init_pace, init_concentration, limit, init_passive, passive_concentration)
            )

    def forward(self, obs: torch.Tensor) -> tuple[Distribution, torch.Tensor]:
        h = self.trunk(obs)
        ab = 1.0 + nn.functional.softplus(self.actor(h))
        return self._dist(ab), self.critic(h).squeeze(-1)


class FuzzyActorCritic(_PaceActorCritic):
    """MLP trunk `h = MLP(obs)` with fuzzy-gated TSK heads over the regime input `x = obs[:, :regime_dim]`
    (ARCHITECTURE §5.1): `(α, β) = 1 + softplus(Σ_r φ_r (A_r h + b_r))`, `V = Σ_r φ_r (u_r·h + d_r)`.

    `antecedent_grad`: `all` (actor and critic losses shape the MFs), `value_only` (only the critic's Bellman
    loss does: φ is detached in the actor head) or `none` (φ detached everywhere; MFs move only by evolution).
    Every rule starts with the same consequents, so the initial policy is the MLP's noisy VWAP whatever the
    antecedents are, and re-placing them (k-means) is function-preserving.

    `gate = hard` is the ablation without fuzziness: φ is the one-hot of the most strongly firing rule (a
    k-means-style partition), so no gradient reaches the antecedents; use it with evolution off.
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
        gate: str = "soft",
        limit: bool = False,
        init_passive: float = 0.2,
        passive_concentration: float | None = None,
    ) -> None:
        super().__init__()
        if antecedent_grad not in ("all", "value_only", "none"):
            raise ValueError(f"antecedent_grad must be all | value_only | none, got {antecedent_grad!r}")
        if gate not in ("soft", "hard"):
            raise ValueError(f"gate must be soft | hard, got {gate!r}")
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
            "gate": gate,
            "limit": limit,
            "init_passive": init_passive,
            "passive_concentration": passive_concentration,
        }
        self.a_max, self.regime_dim, self.antecedent_grad = a_max, regime_dim, antecedent_grad
        self.gate, self.limit = gate, limit
        self.trunk, n_in = _trunk(obs_dim, hidden, activation)
        n_actor = 4 if limit else 2
        self.fuzzy = FuzzyTSK(regime_dim, n_rules, {"actor": n_actor, "critic": 1}, n_in, width)
        actor, critic = _layer(n_in, n_actor, 0.01), _layer(n_in, 1, 1.0)
        bias = _actor_bias(a_max, init_pace, init_concentration, limit, init_passive, passive_concentration)
        with torch.no_grad():
            self.fuzzy.actor_W.copy_(actor.weight.expand(n_rules, -1, -1))
            self.fuzzy.actor_b.copy_(bias.expand(n_rules, -1))
            self.fuzzy.critic_W.copy_(critic.weight.expand(n_rules, -1, -1))

    @property
    def arch(self) -> dict:
        return {**self._arch, "n_rules": self.fuzzy.n_rules}

    def regime(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., : self.regime_dim]

    def forward(self, obs: torch.Tensor) -> tuple[Distribution, torch.Tensor]:
        h = self.trunk(obs)
        phi = self.fuzzy.phi(self.regime(obs))
        if self.gate == "hard":
            phi = nn.functional.one_hot(phi.argmax(-1), phi.shape[-1]).to(phi.dtype)
        phi_actor = phi if self.antecedent_grad == "all" else phi.detach()
        phi_critic = phi.detach() if self.antecedent_grad == "none" else phi
        ab = 1.0 + nn.functional.softplus(self.fuzzy.head("actor", h, phi_actor))
        return self._dist(ab), self.fuzzy.head("critic", h, phi_critic).squeeze(-1)

    @torch.no_grad()
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)


POLICIES = {"mlp": MLPActorCritic, "fuzzy": FuzzyActorCritic}
