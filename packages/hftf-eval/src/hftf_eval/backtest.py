"""Paired backtests: every policy runs on the same episodes (and the same per-episode impact draws)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from hftf_sim.config import EnvConfig
from hftf_sim.env import BatchExecutionEnv
from hftf_sim.episodes import Episodes, EpisodeSampler
from hftf_sim.fees import FeeModel
from hftf_sim.market import MarketData

Policy = Callable[[BatchExecutionEnv], np.ndarray]  # env state -> child sizes in shares


def sample_episodes(market: MarketData, cfg: EnvConfig, split: str, n: int, seed: int) -> Episodes:
    return EpisodeSampler(market, cfg.episode, split).sample(n, np.random.default_rng(seed))


def run_policy(
    policy: Policy,
    market: MarketData,
    cfg: EnvConfig,
    episodes: Episodes,
    fees: FeeModel | None = None,
    randomise_impact: bool = False,
    batch: int = 4096,
) -> pd.DataFrame:
    """One row per episode: the env summary (bps of arrival notional) plus the episode description."""
    frames = []
    for lo in range(0, len(episodes), batch):
        ep = episodes.take(np.arange(lo, min(lo + batch, len(episodes))))
        env = BatchExecutionEnv(
            market, cfg, fees=fees, n_envs=len(ep), autoreset=False, randomise_impact=randomise_impact
        )
        env.reset(ep)
        while not env.done.all():
            env.step_shares(policy(env))
        sess = market.sessions.iloc[ep.session].reset_index(drop=True)
        frames.append(
            pd.DataFrame(
                {
                    "security": sess["security"],
                    "date": sess["date"],
                    "split": sess["split"],
                    "horizon": ep.horizon,
                    "side": ep.side,
                    "qty": ep.qty,
                    **env.summary(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "episode", np.arange(len(out)))
    out.insert(0, "policy", getattr(policy, "name", type(policy).__name__))
    return out
