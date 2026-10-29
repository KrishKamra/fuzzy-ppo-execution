"""Per-minute trace of one parent order under several policies (the demo's data, ARCHITECTURE §M6).

Every policy runs on the same episode with the same (default) impact, in BarSim or TickSim, so traces are
paired like the backtests. Each policy's trace is columnar (one list per field, one entry per decision)
and its last cumulative cost equals the env summary (test). A policy may implement `explain(env)` returning
`{"phi": (n, R) firing}` for its current decision; trained fuzzy policies do (hftf_rl.ppo.PacePolicy), so
this module needs no torch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hftf_data.calendar import SESSION_MINUTES
from hftf_eval.backtest import Policy
from hftf_sim.config import EnvConfig
from hftf_sim.env import REGIME_SLICE
from hftf_sim.episodes import Episodes
from hftf_sim.market import MarketData
from hftf_sim.tick import TickData, make_env

CONTEXT_MIN = 30  # market minutes shown before the order starts
STEP_FIELDS = (
    "minute", "pace", "passive", "q", "posted", "passive_filled", "price", "remaining",
    "cum_is_bps", "cum_fees_bps", "cum_risk_bps", "cum_cost_bps",
)  # fmt: skip


def clock(mso: int) -> str:
    """IST clock time of the start of minute `mso` (minutes since 09:15)."""
    m = 9 * 60 + 15 + int(mso)
    return f"{m // 60:02d}:{m % 60:02d}"


def _policy_trace(
    policy: Policy, market: MarketData, cfg: EnvConfig, episode: Episodes, ticks: TickData | None
) -> dict:
    env = make_env(market, cfg, ticks, n_envs=1, autoreset=False, randomise_impact=False)
    env.reset(episode)
    pace_space = getattr(policy, "action_space", "shares") == "pace"
    explain = getattr(policy, "explain", None)
    steps: dict[str, list] = {f: [] for f in (*STEP_FIELDS, "regime", "phi")}
    while not env.done.all():
        minute = int(env._fill_row()[0])
        regime = env.observe()[0, REGIME_SLICE]
        why = explain(env) if explain is not None else None
        phi = why["phi"][0] if why is not None else None
        action = np.asarray(policy(env), dtype=np.float64)
        if pace_space:
            act = action.reshape(1, -1)
            pace, passive = float(act[0, 0]), float(act[0, 1]) if act.shape[1] > 1 else 0.0
            _, _, _, info = env.step(action)
        else:
            pace, passive = float(env.implied_pace(action)[0]), 0.0
            _, _, _, info = env.step_shares(action)
        t = env.totals
        row = {
            "minute": minute, "pace": pace, "passive": passive, "q": float(info["q"][0]),
            "posted": float(info["posted"][0]), "passive_filled": float(info["passive_filled"][0]),
            "price": float(info["price"][0]), "remaining": float(env.remaining[0]),
            "cum_is_bps": float(t["is_bps"][0]), "cum_fees_bps": float(t["fees_bps"][0]),
            "cum_risk_bps": float(t["risk_bps"][0]),
        }  # fmt: skip
        row["cum_cost_bps"] = row["cum_is_bps"] + row["cum_fees_bps"] + row["cum_risk_bps"]
        for f in STEP_FIELDS:
            steps[f].append(row[f])
        steps["regime"].append([float(x) for x in regime])
        steps["phi"].append(None if phi is None else [float(x) for x in phi])
    summary = {k: float(v[0]) for k, v in env.summary().items()}
    if all(p is None for p in steps["phi"]):
        del steps["phi"]
    return {"name": getattr(policy, "name", type(policy).__name__), "summary": summary, "steps": steps}


def run_trace(
    policies: Sequence[Policy],
    market: MarketData,
    cfg: EnvConfig,
    episode: Episodes,
    ticks: TickData | None = None,
) -> dict:
    """The order, the market around it (from `CONTEXT_MIN` before its start to its end) and one trace per
    policy. Prices in ₹, costs in bps of arrival notional, minutes since 09:15 (`clock` gives IST)."""
    if len(episode) != 1:
        raise ValueError("run_trace takes a single episode")
    s, row0, h = int(episode.session[0]), int(episode.row0[0]), int(episode.horizon[0])
    lo, hi = max(row0 - CONTEXT_MIN, 0), min(row0 + h + 1, SESSION_MINUTES)
    rows = np.arange(lo, hi)
    # A bar's closing quote is known at its end: the state a decision at minute m + 1 sees.
    mid, spread = market.mid[s, rows], market.spread[s, rows]
    sess = market.sessions.iloc[s]
    traces = [_policy_trace(p, market, cfg, episode, ticks) for p in policies]
    side = int(episode.side[0])
    # The arrival mid the costs are measured from (TickSim reads it from the 1-second grid; DATA_AUDIT F12).
    arrival = float(market.mid[s, row0] if ticks is None else ticks.mid[ticks.row[s], 60 * (row0 + 1)])
    return {
        "order": {
            "security": str(sess["security"]), "date": str(sess["date"])[:10], "split": str(sess["split"]),
            "start": clock(row0 + 1), "end": clock(row0 + 1 + h), "horizon": h,
            "side": "buy" if side > 0 else "sell", "qty": float(episode.qty[0]), "arrival_mid": arrival,
            "sim": "tick" if ticks is not None else "bar",
        },
        "market": {
            "minute": rows.tolist(),
            "time": [clock(m) for m in rows],
            "mid": mid.tolist(),
            "bid": (mid * (1 - spread / 2)).tolist(),
            "ask": (mid * (1 + spread / 2)).tolist(),
            "volume": market.volume[s, rows].tolist(),
            "expected_volume": market.prof[s, rows].tolist(),
        },
        "regime_inputs": list(market.obs_columns[REGIME_SLICE]),
        "policies": traces,
    }  # fmt: skip
