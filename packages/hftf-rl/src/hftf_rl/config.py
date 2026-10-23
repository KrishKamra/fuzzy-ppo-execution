"""Validated PPO config (`configs/ppo.yaml`, ARCHITECTURE §6)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from hftf_data.paths import REPO_ROOT

PPO_YAML = REPO_ROOT / "configs" / "ppo.yaml"


class TrunkConfig(BaseModel):
    hidden: list[int] = [64, 64]
    activation: Literal["tanh", "relu"] = "tanh"


class PPOConfig(BaseModel):
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
