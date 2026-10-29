"""hftf_sim: execution environment (ARCHITECTURE §4).

Modules: config (validated env/fee YAML), market (session arrays from features_1m_v3), episodes
(EpisodeSampler), impact (square-root + transient ImpactModel), fees (NSE FeeModel),
env (BatchExecutionEnv + gymnasium ExecutionEnv), tick (TickSim: fills against the 1-second L1 grid).
"""

from hftf_sim.config import EnvConfig, FeeConfig
from hftf_sim.env import BatchExecutionEnv, ExecutionEnv, obs_names
from hftf_sim.episodes import Episodes, EpisodeSampler
from hftf_sim.fees import FeeModel
from hftf_sim.impact import ImpactModel
from hftf_sim.market import MarketData
from hftf_sim.tick import TickData, TickExecutionEnv, make_env

__all__ = [
    "BatchExecutionEnv",
    "EnvConfig",
    "EpisodeSampler",
    "Episodes",
    "ExecutionEnv",
    "FeeConfig",
    "FeeModel",
    "ImpactModel",
    "MarketData",
    "TickData",
    "TickExecutionEnv",
    "make_env",
    "obs_names",
]
