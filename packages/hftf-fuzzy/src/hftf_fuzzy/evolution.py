"""Structure evolution between PPO iterations (ARCHITECTURE §5.3): add on novelty, merge on overlap,
prune on low utility, under rule-count guards, per-iteration limits and a cooldown.

`Evolver.step` edits the layer in place and returns the events. The caller (hftf_rl) can measure the
policy change and undo an edit with `layer.restore(snap)` and `evolver.restore(state)`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from hftf_fuzzy.layer import FuzzyTSK, bhattacharyya


@dataclass
class EvolutionConfig:
    enabled: bool = True
    min_rules: int = 2
    max_rules: int = 16
    tau_add: float = 0.15
    n_min_uncovered: int = 64
    tau_merge: float = 0.8
    tau_prune: float = 0.02
    utility_ema: float = 0.9
    min_age_iters: int = 10
    max_adds_per_iter: int = 1
    max_prunes_per_iter: int = 1
    cooldown_iters: int = 3
    min_width: float = 0.1  # floor for a new rule's width (standardised units)


@dataclass
class Event:
    kind: str  # add | merge | prune
    rules: list[int]  # rule indices before the edit (add: the new index)
    n_rules: int  # after the edit
    detail: dict = field(default_factory=dict)


class Evolver:
    def __init__(self, cfg: EvolutionConfig, n_rules: int) -> None:
        self.cfg = cfg
        self.utility = np.full(n_rules, 1.0 / n_rules)
        self.age = np.zeros(n_rules, dtype=int)
        self.cooldown = 0

    def state(self) -> dict:
        return copy.deepcopy({"utility": self.utility, "age": self.age, "cooldown": self.cooldown})

    def restore(self, state: dict) -> None:
        self.utility, self.age, self.cooldown = state["utility"], state["age"], state["cooldown"]

    def _remap(self, rows: list[int], new_utility: float = 0.0, new_age: int = 0) -> None:
        self.utility = np.array([self.utility[r] if r >= 0 else new_utility for r in rows])
        self.age = np.array([self.age[r] if r >= 0 else new_age for r in rows], dtype=int)

    @torch.no_grad()
    def step(self, layer: FuzzyTSK, x: torch.Tensor) -> list[Event]:
        """One evolution pass over buffer samples `x` (N, d) of the regime input."""
        cfg = self.cfg
        phi = layer.phi(x)
        self.utility = cfg.utility_ema * self.utility + (1 - cfg.utility_ema) * phi.mean(0).numpy()
        self.age += 1
        if not cfg.enabled:
            return []
        if self.cooldown > 0:
            self.cooldown -= 1
            return []
        events: list[Event] = []

        # 1. add: samples no rule covers well
        tau_max = layer.firing(x).max(-1).values
        uncovered = tau_max < cfg.tau_add
        if layer.n_rules < cfg.max_rules and int(uncovered.sum()) >= cfg.n_min_uncovered:
            xs = x[uncovered]
            centre, width = xs.mean(0), xs.std(0).clamp_min(cfg.min_width)
            # Function-preserving: the new consequent copies the existing mixture's average consequent at the
            # uncovered samples, so the outputs there barely move when the new rule takes over.
            w = phi[uncovered]  # (n, R)
            cons = {}
            for name in layer.rule_parameter_names()[2:]:
                cons[name] = torch.einsum("nr,r...->...", w, getattr(layer, name).detach()) / w.shape[0]
            layer.add_rule(centre, width, cons)
            self._remap(layer.last_rows, new_utility=float(w.shape[0]) / x.shape[0])
            events.append(
                Event("add", [layer.n_rules - 1], layer.n_rules, {"uncovered": int(uncovered.sum())})
            )

        # 2. merge: the most overlapping pair of established rules
        if not events and layer.n_rules > cfg.min_rules:
            c, s = layer.centres.detach(), layer.widths().detach()
            best, pair = cfg.tau_merge, None
            for i in range(layer.n_rules):
                for j in range(i + 1, layer.n_rules):
                    if min(self.age[i], self.age[j]) < cfg.min_age_iters:
                        continue
                    bc = bhattacharyya(c[i], s[i], c[j], s[j])
                    if bc > best:
                        best, pair = bc, (i, j)
            if pair is not None:
                i, j = pair
                wi, wj = max(self.utility[i], 1e-6), max(self.utility[j], 1e-6)
                age = int(max(self.age[i], self.age[j]))
                u = float(self.utility[i] + self.utility[j])
                layer.merge_rules(i, j, wi, wj)
                self._remap(layer.last_rows, new_utility=u, new_age=age)
                events.append(Event("merge", [i, j], layer.n_rules, {"bhattacharyya": round(best, 4)}))

        # 3. prune: low long-run utility, old enough
        if layer.n_rules > cfg.min_rules:
            low = np.flatnonzero((self.utility < cfg.tau_prune) & (self.age >= cfg.min_age_iters))
            for r in sorted(low, key=lambda r: self.utility[r])[: cfg.max_prunes_per_iter]:
                if layer.n_rules <= cfg.min_rules:
                    break
                u = float(self.utility[r])
                layer.remove_rules([int(r)])
                self._remap(layer.last_rows)
                events.append(Event("prune", [int(r)], layer.n_rules, {"utility": round(u, 5)}))
                break

        if events:
            self.cooldown = cfg.cooldown_iters
        return events


def kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain k-means++ / Lloyd. Returns centres (k, d) and per-cluster std (k, d)."""
    g = torch.Generator().manual_seed(seed)
    centres = x[torch.randint(len(x), (1,), generator=g)]
    while len(centres) < k:
        d2 = torch.cdist(x, centres).min(1).values ** 2
        centres = torch.cat([centres, x[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
    for _ in range(iters):
        label = torch.cdist(x, centres).argmin(1)
        centres = torch.stack([x[label == c].mean(0) if (label == c).any() else centres[c] for c in range(k)])
    label = torch.cdist(x, centres).argmin(1)
    std = torch.stack([x[label == c].std(0) if (label == c).sum() > 1 else x.std(0) for c in range(k)])
    return centres, std
