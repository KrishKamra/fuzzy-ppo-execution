"""Rule report for a trained fuzzy policy (ARCHITECTURE §5.4): linguistic antecedents from train terciles,
the firing share of each rule and its firing-weighted mean pace on a set of episodes."""

from __future__ import annotations

import numpy as np
import torch

from hftf_fuzzy.interpret import firing_summary, rule_table, terciles
from hftf_rl.ppo import PacePolicy
from hftf_sim.config import EnvConfig
from hftf_sim.env import REGIME_SLICE, BatchExecutionEnv
from hftf_sim.episodes import Episodes
from hftf_sim.market import SIGNED_COLUMNS, MarketData

INPUTS = ["imbalance", "spread_var", "rv"]
TERMS = [("AGAINST", "NEUTRAL", "WITH"), ("LOW", "MED", "HIGH"), ("LOW", "MED", "HIGH")]


def train_regime_terciles(market: MarketData) -> np.ndarray:
    """Terciles of the env-standardised regime inputs on train rows; signed inputs mirrored (both sides)."""
    train = market.session_ids("train")
    x = market.obs[train][:, :, REGIME_SLICE].reshape(-1, REGIME_SLICE.stop)
    signed = np.array([c in SIGNED_COLUMNS for c in market.obs_columns[REGIME_SLICE]])
    x = np.concatenate([x, np.where(signed, -x, x)])
    return terciles(x)


def collect(policy: PacePolicy, market: MarketData, cfg: EnvConfig, episodes: Episodes):
    """Normalised observations, deterministic actions (pace, or pace and passive fraction) and session ids at
    every live decision."""
    env = BatchExecutionEnv(market, cfg, n_envs=len(episodes), autoreset=False, randomise_impact=False)
    env.reset(episodes)
    obs, pace, session = [], [], []
    while not env.done.all():
        live = ~env.done
        o = policy.obs_rms.normalize(env.observe()).astype(np.float32)
        a = policy.model.mean_pace(o)
        obs.append(o[live])
        pace.append(a[live])
        session.append(env.ep.session[live])
        env.step(a)
    return np.concatenate(obs), np.concatenate(pace), np.concatenate(session)


def rule_report(policy: PacePolicy, market: MarketData, cfg: EnvConfig, episodes: Episodes) -> list[dict]:
    layer = policy.model.fuzzy
    obs, pace, _ = collect(policy, market, cfg, episodes)
    x = torch.as_tensor(obs[:, REGIME_SLICE])
    action = pace.reshape(len(pace), -1)  # (pace,) or (pace, passive fraction) with limit orders
    share, rule_pace = firing_summary(layer, x, action[:, 0])
    rows = rule_table(layer, train_regime_terciles(market), INPUTS, TERMS)
    for r in rows:
        r["share"], r["pace"] = float(share[r["rule"]]), float(rule_pace[r["rule"]])
    if action.shape[1] == 2:
        _, rule_passive = firing_summary(layer, x, action[:, 1])
        for r in rows:
            r["passive"] = float(rule_passive[r["rule"]])
    return sorted(rows, key=lambda r: -r["share"])


def dominant_rules(policy: PacePolicy, market: MarketData, cfg: EnvConfig, episodes: Episodes):
    """The most strongly firing rule and the session id at every live decision (for rule-regime AMI)."""
    obs, _, session = collect(policy, market, cfg, episodes)
    with torch.no_grad():
        phi = policy.model.fuzzy.phi(torch.as_tensor(obs[:, REGIME_SLICE]))
    return phi.argmax(-1).numpy(), session
