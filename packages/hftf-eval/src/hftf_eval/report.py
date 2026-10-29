"""M4 experiment tables: many policies on the same fresh episodes, paired statistics against a reference,
breakdowns by the eval-only regime label, and impact-robustness sweeps (ARCHITECTURE §7)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hftf_data.paths import REPO_ROOT
from hftf_eval.backtest import Policy, run_policy, sample_episodes
from hftf_eval.baselines import VWAP
from hftf_eval.stats import breakdown, paired_stats
from hftf_sim.config import EnvConfig
from hftf_sim.market import MarketData
from hftf_sim.tick import TickData

LABEL = "regime_eval_prevday"  # previous-day RV tercile: evaluation only, never a model input


def eval_labels(features_path: str | Path) -> pd.DataFrame:
    path = Path(features_path)
    path = path if path.is_absolute() else REPO_ROOT / path
    df = pd.read_parquet(path, columns=["security", "date", LABEL])
    return df.drop_duplicates(["security", "date"]).reset_index(drop=True)


def backtest(
    policies: Sequence[Policy],
    market: MarketData,
    cfg: EnvConfig,
    split: str,
    n_episodes: int,
    seed: int,
    labels: pd.DataFrame | None = None,
    ticks: TickData | None = None,
    sessions=None,
) -> pd.DataFrame:
    """Long backtest rows for VWAP and every policy on one shared set of episodes (default impact). With
    `ticks` they run in TickSim; `sessions` restricts the episodes (e.g. to the tick sessions)."""
    ep = sample_episodes(market, cfg, split, n_episodes, seed, sessions)
    rows = [run_policy(p, market, cfg, ep, ticks=ticks) for p in (VWAP(), *policies)]
    df = pd.concat(rows, ignore_index=True)
    if labels is not None:
        df = df.merge(labels, on=["security", "date"], how="left")
    return df


def stats_table(df: pd.DataFrame, policies: Sequence[str], reference: str = "vwap", **kwargs) -> pd.DataFrame:
    """One row per policy: `policy − reference` objective with the session-cluster CI and Wilcoxon p."""
    return pd.DataFrame(
        [{"policy": p, "reference": reference, **paired_stats(df, p, reference, **kwargs)} for p in policies]
    )


def regime_table(
    df: pd.DataFrame, policies: Sequence[str], reference: str = "vwap", by: str = LABEL
) -> pd.DataFrame:
    parts = [breakdown(df, p, reference, by).assign(policy=p, reference=reference) for p in policies]
    return pd.concat(parts, ignore_index=True)


def with_impact(cfg: EnvConfig, **impact) -> EnvConfig:
    """A copy of `cfg` with impact parameters overridden (eta, gamma, beta, tau_res_min)."""
    return cfg.model_copy(update={"impact": cfg.impact.model_copy(update=impact)})


def impact_sweep(
    policies: Sequence[Policy],
    market: MarketData,
    cfg: EnvConfig,
    split: str,
    n_episodes: int,
    seed: int,
    param: str,
    values: Sequence[float],
    vs: str | None = None,
    ticks: TickData | None = None,
    sessions=None,
) -> pd.DataFrame:
    """`policy − VWAP` (and `policy − vs`, if given) for each impact setting; episodes are shared across
    settings and policies."""
    rows = []
    names = [p.name for p in policies]
    for v in values:
        df = backtest(
            policies, market, with_impact(cfg, **{param: v}), split, n_episodes, seed,
            ticks=ticks, sessions=sessions,
        )  # fmt: skip
        vwap_cost = df.loc[df["policy"] == "vwap", "cost_bps"].mean()
        tables = [stats_table(df, names)]
        if vs is not None:
            tables.append(stats_table(df, [n for n in names if n != vs], reference=vs))
        rows += [t.assign(**{param: v, "vwap_cost": vwap_cost}) for t in tables]
    return pd.concat(rows, ignore_index=True)
