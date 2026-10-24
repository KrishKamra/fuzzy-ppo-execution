"""Validated PPO config (`configs/ppo.yaml`, ARCHITECTURE §6)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from hftf_data.paths import REPO_ROOT

PPO_YAML = REPO_ROOT / "configs" / "ppo.yaml"
EFNN_YAML = REPO_ROOT / "configs" / "efnn.yaml"


class TrunkConfig(BaseModel):
    hidden: list[int] = [64, 64]
    activation: Literal["tanh", "relu"] = "tanh"


class PPOConfig(BaseModel):
    policy: Literal["mlp", "fuzzy"] = "mlp"
    n_envs: int = 256
    rollout_steps: int = 120
    total_timesteps: int = 6_000_000
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatches: int = 8
    lr: float = 1.0e-3
    lr_anneal: bool = False  # linear decay to 0 over total_timesteps
    entropy_coef: float = 1.0e-3
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    trunk: TrunkConfig = TrunkConfig()
    init_pace: float = 1.0  # initial mean action: a = 1 is adaptive VWAP
    init_concentration: float = 8.0  # alpha + beta of the initial Beta
    randomise_impact: bool = True
    eval_every: int = 3  # iterations between deterministic train/val evaluations
    eval_episodes: int = 2000
    eval_seed: int = 12345
    torch_threads: int = 4
    seed: int = 0

    @classmethod
    def load(cls, path: Path = PPO_YAML) -> PPOConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class EvolutionSettings(BaseModel):
    enabled: bool = True
    tau_add: float = 0.15
    n_min_uncovered: int = 64
    tau_merge: float = 0.8
    tau_prune: float = 0.02
    utility_ema: float = 0.9
    min_age_iters: int = 10
    max_adds_per_iter: int = 1
    max_prunes_per_iter: int = 1
    cooldown_iters: int = 3
    min_width: float = 0.1


class EFNNConfig(BaseModel):
    """`configs/efnn.yaml` (ARCHITECTURE §5)."""

    init_rules: int = 4
    min_rules: int = 2
    max_rules: int = 16
    antecedent_grad: Literal["all", "value_only", "none"] = "all"
    gate: Literal["soft", "hard"] = "soft"  # hard: one-hot φ (k-means partition ablation), evolution off
    antecedent_lr_scale: float = 0.3
    init_width_scale: float = 1.0  # k-means cluster std x this = initial MF width
    edit_kl_max: float = 0.02  # reject a structure edit whose policy KL on the buffer exceeds this
    edit_kl_samples: int = 8192
    evolution: EvolutionSettings = EvolutionSettings()

    @classmethod
    def load(cls, path: Path = EFNN_YAML) -> EFNNConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def evolution_config(self):
        from hftf_fuzzy.evolution import EvolutionConfig

        return EvolutionConfig(
            min_rules=self.min_rules, max_rules=self.max_rules, **self.evolution.model_dump()
        )
