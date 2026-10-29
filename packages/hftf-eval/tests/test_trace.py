import numpy as np
import pytest

from hftf_eval.backtest import run_policy
from hftf_eval.baselines import TWAP, VWAP, AlmgrenChriss, PassiveVWAP
from hftf_eval.trace import CONTEXT_MIN, clock, run_trace
from hftf_sim import EnvConfig, MarketData, TickData
from hftf_sim.episodes import order_episode
from hftf_sim.testing import STATS, synthetic_frame, synthetic_ticks


def _setup():
    frame = synthetic_frame(n_sessions=2)
    mk = MarketData.from_frame(frame, stats=STATS)
    return mk, EnvConfig.load(), TickData.from_frame(mk, synthetic_ticks(frame, depth=30.0))


@pytest.mark.parametrize("tick", [False, True])
def test_trace_totals_match_the_backtest_summary(tick):
    mk, cfg, ticks = _setup()
    ticks = ticks if tick else None
    ep = order_episode(mk, cfg.episode, 1, "11:05", 60, "sell", 0.08)
    policies = [VWAP(), TWAP(), AlmgrenChriss(), PassiveVWAP(1.0)]
    out = run_trace(policies, mk, cfg, ep, ticks)
    assert (
        out["order"]["start"] == "11:05" and out["order"]["end"] == "12:05" and out["order"]["side"] == "sell"
    )
    assert out["order"]["sim"] == ("tick" if tick else "bar")
    assert len(out["market"]["minute"]) == CONTEXT_MIN + 60 + 1
    for p, trace in zip(policies, out["policies"], strict=True):
        steps = trace["steps"]
        assert len(steps["q"]) == 60 and sum(steps["q"]) == ep.qty[0] and steps["remaining"][-1] == 0
        assert steps["minute"] == list(range(ep.row0[0] + 1, ep.row0[0] + 61))
        assert "phi" not in steps and all(len(r) == 3 for r in steps["regime"])
        row = run_policy(p, mk, cfg, ep, ticks=ticks).iloc[0]
        np.testing.assert_allclose(steps["cum_cost_bps"][-1], row["cost_bps"], rtol=1e-12)
        np.testing.assert_allclose(trace["summary"]["cost_bps"], row["cost_bps"], rtol=1e-12)
    passive = out["policies"][3]["steps"]
    assert all(f <= p for f, p in zip(passive["passive_filled"], passive["posted"], strict=True))
    # Synthetic bars trade at the touch; the synthetic tick grid has no prints, so nothing fills there.
    assert (sum(passive["passive_filled"]) > 0) != tick


def test_order_ticket_validation_and_clock():
    mk, cfg, _ = _setup()
    assert clock(0) == "09:15" and clock(359) == "15:14"
    ep = order_episode(mk, cfg.episode, 0, "09:20", 30, "buy", 0.1)
    assert ep.row0[0] == 4 and ep.side[0] == 1 and ep.qty[0] == np.rint(0.1 * mk.prof[0, 5:35].sum())
    for bad in (("09:19", 30, "buy", 0.1), ("14:50", 30, "buy", 0.1), ("10:00", 45, "buy", 0.1),
                ("10:00", 30, "hold", 0.1), ("10:00", 30, "buy", 0.0)):  # fmt: skip
        with pytest.raises(ValueError):
            order_episode(mk, cfg.episode, 0, *bad)
