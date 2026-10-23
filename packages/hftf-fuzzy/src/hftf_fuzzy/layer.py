"""First-order TSK fuzzy layer with Gaussian antecedents (ARCHITECTURE §5.1).

    τ_r(x) = exp(−½ Σ_j ((x_j − c_rj)/σ_rj)²),   σ = softplus(ρ)
    φ_r(x) = m_r τ_r / Σ_k m_k τ_k              (a softmax of log m + log τ, so it never underflows)
    head(h) = Σ_r φ_r(x) · (W_r h + b_r)

`m_r` is a rule's support mass (a buffer, not trained): 1 for a new rule, and the sum of the two masses for a
merged one, so merging two identical rules leaves φ, and hence the output, unchanged.

Every tensor whose first axis is the rule axis is a "rule parameter". Structure edits (evolution.py) replace
them all at once and record which old rule each new row came from (`last_rows`, -1 for a new row), so an
optimizer's per-row state can be carried over (`remap_optimizer_state`).
"""

from __future__ import annotations

import math

import torch
from torch import nn

MIN_WIDTH = 1e-3


def inv_softplus(y: torch.Tensor | float) -> torch.Tensor:
    y = torch.as_tensor(y, dtype=torch.float32)
    return y + torch.log(-torch.expm1(-y))


class FuzzyTSK(nn.Module):
    """Rules over a `d`-dimensional regime input, with one linear consequent per rule for each named head."""

    def __init__(self, d: int, n_rules: int, heads: dict[str, int], hidden: int, width: float = 1.0) -> None:
        super().__init__()
        self.d, self.hidden, self.head_dims = d, hidden, dict(heads)
        self.centres = nn.Parameter(torch.zeros(n_rules, d))
        self.rho = nn.Parameter(torch.full((n_rules, d), float(inv_softplus(width))))
        for name, out in heads.items():
            self.register_parameter(f"{name}_W", nn.Parameter(torch.zeros(n_rules, out, hidden)))
            self.register_parameter(f"{name}_b", nn.Parameter(torch.zeros(n_rules, out)))
        self.register_buffer("log_mass", torch.zeros(n_rules))
        self.last_rows: list[int] = list(range(n_rules))

    # ------------------------------------------------------------- inference
    @property
    def n_rules(self) -> int:
        return self.centres.shape[0]

    def widths(self) -> torch.Tensor:
        return nn.functional.softplus(self.rho) + MIN_WIDTH

    def log_firing(self, x: torch.Tensor) -> torch.Tensor:
        z = (x[..., None, :] - self.centres) / self.widths()
        return -0.5 * z.pow(2).sum(-1)

    def firing(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalised firing strengths τ ∈ (0, 1], shape (..., R)."""
        return self.log_firing(x).exp()

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised firing φ (a soft regime assignment), shape (..., R)."""
        return torch.softmax(self.log_firing(x) + self.log_mass, dim=-1)

    def consequents(self, name: str, h: torch.Tensor) -> torch.Tensor:
        """Every rule's linear output `W_r h + b_r`, shape (..., R, out)."""
        W, b = getattr(self, f"{name}_W"), getattr(self, f"{name}_b")
        return torch.einsum("rok,...k->...ro", W, h) + b

    def head(self, name: str, h: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return (phi[..., None] * self.consequents(name, h)).sum(-2)

    # -------------------------------------------------------- structure edits
    def rule_parameter_names(self) -> list[str]:
        return ["centres", "rho", *(f"{n}_{p}" for n in self.head_dims for p in ("W", "b"))]

    def antecedent_parameters(self) -> list[nn.Parameter]:
        return [self.centres, self.rho]

    def consequent_parameters(self) -> list[nn.Parameter]:
        return [getattr(self, n) for n in self.rule_parameter_names()[2:]]

    def rule_tensor_names(self) -> list[str]:
        return [*self.rule_parameter_names(), "log_mass"]

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {n: getattr(self, n).detach().clone() for n in self.rule_tensor_names()}

    def set_rules(self, tensors: dict[str, torch.Tensor], rows: list[int]) -> None:
        """Replace every rule parameter (same Parameter objects, new data) and record the row provenance."""
        for name in self.rule_parameter_names():
            p = getattr(self, name)
            p.data = tensors[name].detach().clone().to(p.dtype)
            p.grad = None
        self.log_mass = tensors["log_mass"].detach().clone()
        self.last_rows = list(rows)

    def restore(self, snap: dict[str, torch.Tensor]) -> None:
        self.set_rules(snap, list(range(snap["centres"].shape[0])))

    def add_rule(
        self, centre: torch.Tensor, width: torch.Tensor, consequents: dict[str, torch.Tensor]
    ) -> None:
        """Append a rule; `consequents` maps "<head>_W" / "<head>_b" to its (out, hidden) / (out,) values."""
        cur = self.snapshot()
        new = {
            "centres": torch.cat([cur["centres"], centre[None]]),
            "rho": torch.cat([cur["rho"], inv_softplus((width - MIN_WIDTH).clamp_min(1e-4))[None]]),
        }
        for name in self.rule_parameter_names()[2:]:
            new[name] = torch.cat([cur[name], consequents[name][None]])
        new["log_mass"] = torch.cat([cur["log_mass"], torch.zeros(1)])
        self.set_rules(new, [*range(self.n_rules), -1])

    def remove_rules(self, drop: list[int]) -> None:
        keep = [r for r in range(self.n_rules) if r not in set(drop)]
        self.set_rules({n: t[keep] for n, t in self.snapshot().items()}, keep)

    def merge_rules(self, i: int, j: int, wi: float, wj: float) -> None:
        """Support-weighted merge of rules i and j: moment-matched Gaussian, weighted consequents."""
        cur = self.snapshot()
        a, b = wi / (wi + wj), wj / (wi + wj)
        sig = self.widths().detach()
        c = a * cur["centres"][i] + b * cur["centres"][j]
        var = a * (sig[i] ** 2 + cur["centres"][i] ** 2) + b * (sig[j] ** 2 + cur["centres"][j] ** 2) - c**2
        merged = {
            "centres": c,
            "rho": inv_softplus((var.clamp_min(1e-8).sqrt() - MIN_WIDTH).clamp_min(1e-4)),
        }
        for name in self.rule_parameter_names()[2:]:
            merged[name] = a * cur[name][i] + b * cur[name][j]
        merged["log_mass"] = torch.logaddexp(cur["log_mass"][i], cur["log_mass"][j])
        keep = [r for r in range(self.n_rules) if r not in (i, j)]
        new = {n: torch.cat([cur[n][keep], merged[n][None]]) for n in cur}
        self.set_rules(new, [*keep, -1])


def remap_optimizer_state(opt: torch.optim.Optimizer, layer: FuzzyTSK) -> None:
    """After a structure edit, keep Adam moments for surviving rules and start new rows from zero."""
    rows = layer.last_rows
    for name in layer.rule_parameter_names():
        p = getattr(layer, name)
        state = opt.state.get(p)
        if not state:
            continue
        for key, val in list(state.items()):
            if torch.is_tensor(val) and val.dim() > 0 and val.shape[1:] == p.shape[1:]:
                new = torch.zeros_like(p.data)
                for r_new, r_old in enumerate(rows):
                    if 0 <= r_old < val.shape[0]:
                        new[r_new] = val[r_old]
                state[key] = new


@torch.no_grad()
def bhattacharyya(c1: torch.Tensor, s1: torch.Tensor, c2: torch.Tensor, s2: torch.Tensor) -> float:
    """Bhattacharyya coefficient of two diagonal Gaussians (1 = identical, 0 = disjoint)."""
    v1, v2 = s1**2, s2**2
    v = (v1 + v2) / 2
    db = 0.125 * ((c1 - c2) ** 2 / v).sum() + 0.5 * (
        torch.log(v).sum() - 0.5 * (torch.log(v1).sum() + torch.log(v2).sum())
    )
    return math.exp(-float(db))
