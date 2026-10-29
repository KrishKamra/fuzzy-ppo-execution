import numpy as np
import pytest

from hftf_sim import BatchExecutionEnv, EnvConfig, EpisodeSampler, FeeModel, MarketData, TickData
from hftf_sim.config import FeeConfig
from hftf_sim.testing import STATS, fixed_episodes, synthetic_frame, synthetic_ticks
from hftf_sim.tick import TickExecutionEnv


def _setup(depth=1e9, flat=False, **tick):
    frame = synthetic_frame(n_sessions=2, flat=flat)
    mk = MarketData.from_frame(frame, stats=STATS)
    cfg = EnvConfig.load()
    cfg.tick = cfg.tick.model_copy(update=tick)
    return mk, cfg, TickData.from_frame(mk, synthetic_ticks(frame, depth=depth))


@pytest.mark.parametrize(
    ("mode", "drift"), [("vs_vwap", "expected"), ("vs_vwap", "realised"), ("arrival", "realised")]
)
def test_one_slice_without_depth_reproduces_barsim(mode, drift):
    """On a tick grid built from the bars, one order per decision and no depth fills is exactly BarSim."""
    mk, cfg, ticks = _setup(depth_fills=False, sub_slices=1, latency_s=1)
    cfg.reward.mode, cfg.reward.price_drift = mode, drift
    ep = EpisodeSampler(mk, cfg.episode, "train").sample(128, np.random.default_rng(0))
    actions = np.random.default_rng(1).uniform(0, cfg.action.a_max, size=(120, 128))
    out = []
    for env in (
        BatchExecutionEnv(mk, cfg, n_envs=128, autoreset=False),
        TickExecutionEnv(mk, cfg, ticks, n_envs=128, autoreset=False),
    ):
        env.reset(ep)
        rewards = [env.step(a)[1] for a in actions[: int(ep.horizon.max())]]
        out.append((np.array(rewards), env.summary()))
    np.testing.assert_allclose(out[0][0], out[1][0], rtol=1e-9, atol=1e-12)
    for k in out[0][1]:
        np.testing.assert_allclose(out[0][1][k], out[1][1][k], rtol=1e-9, atol=1e-12)


def test_child_within_the_touch_pays_only_the_half_spread():
    mk, cfg, ticks = _setup(depth=1e9, flat=True, depth_fills=True)
    cfg.impact = cfg.impact.model_copy(update={"gamma": 0.0})
    cfg.reward.risk_aversion = 0.0
    env = TickExecutionEnv(
        mk, cfg, ticks, fees=FeeModel.zero(), n_envs=1, autoreset=False, randomise_impact=False
    )
    env.reset(fixed_episodes(1, horizon=30, qty=3000.0))
    while not env.done.all():
        env.step(np.ones(1))
    np.testing.assert_allclose(env.summary()["is_bps"], [1.0])  # spread 2 bps -> 1 bp


def test_excess_over_the_touch_walks_the_book():
    """With depth d, a child of q shares pays the square-root term on q − d only, averaged over q."""
    mk, cfg, ticks = _setup(depth=40.0, flat=True, depth_fills=True)
    env = TickExecutionEnv(
        mk, cfg, ticks, fees=FeeModel.zero(), n_envs=1, autoreset=False, randomise_impact=False
    )
    env.reset(fixed_episodes(1, horizon=30, qty=3000.0))
    _, _, _, info = env.step(np.ones(1))
    q, m = info["q"][0], mk.mid[0, 5]
    imp = cfg.impact
    expected = m * (1 + 1e-4 + (q - 40) / q * imp.eta * mk.sigma[0, 5] * ((q - 40) / 1000.0) ** imp.beta)
    np.testing.assert_allclose(info["price"][0], expected, rtol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_sub_slices_conserve_shares_telescope_and_pay_fees_per_order(n):
    mk, cfg, ticks = _setup(depth=25.0, sub_slices=n)
    cfg.reward.mode, cfg.reward.price_drift, cfg.reward.risk_aversion = "mark_to_market", "realised", 0.0
    zero_side = {"delivery": {"buy": 0.0, "sell": 0.0}}
    fixed = FeeModel(
        FeeConfig(stt=zero_side, exchange_txn=0.0, sebi_fee=0.0, stamp_duty=zero_side, gst_rate=0.0,
                  brokerage={"fixed_per_order_inr": 20.0})
    )  # fmt: skip
    ep = EpisodeSampler(mk, cfg.episode, "train").sample(64, np.random.default_rng(3))
    env = TickExecutionEnv(mk, cfg, ticks, fees=FeeModel.zero(), n_envs=64, autoreset=False)
    fenv = TickExecutionEnv(mk, cfg, ticks, fees=fixed, n_envs=64, autoreset=False)
    env.reset(ep), fenv.reset(ep)
    rng = np.random.default_rng(4)
    total, ret, orders = np.zeros(64), np.zeros(64), np.zeros(64)
    while not env.done.all():
        a = rng.uniform(0, cfg.action.a_max, size=64)
        _, r, _, info = env.step(a)
        fenv.step(a)
        total += info["q"]
        ret += r
        orders += np.minimum(info["q"], n)  # whole-share slices: min(q, n) non-empty orders
    s = env.summary()
    np.testing.assert_array_equal(total, ep.qty)
    np.testing.assert_allclose(-ret, s["is_bps"], rtol=1e-9, atol=1e-9)  # mark-to-market telescopes to IS
    notional0 = ep.qty * env.m0
    np.testing.assert_allclose(fenv.summary()["fees_bps"], orders * 20.0 / notional0 * 1e4, rtol=1e-9)


def test_sub_slice_kicks_add_up_to_the_barsim_kick_when_impact_is_permanent():
    mk, cfg, ticks = _setup(depth_fills=False, sub_slices=4)
    cfg.impact = cfg.impact.model_copy(update={"tau_res_min": 1e12})
    ep = fixed_episodes(1, horizon=30, qty=3000.0)  # 100-share children: 4 equal orders
    bar = BatchExecutionEnv(mk, cfg, n_envs=1, autoreset=False, randomise_impact=False)
    tick = TickExecutionEnv(mk, cfg, ticks, n_envs=1, autoreset=False, randomise_impact=False)
    for env in (bar, tick):
        env.reset(ep)
        env.step(np.ones(1))
    np.testing.assert_allclose(tick.disp, bar.disp, rtol=1e-9)


def test_tick_env_rejects_sessions_without_ticks():
    frame = synthetic_frame(n_sessions=2)
    mk = MarketData.from_frame(frame, stats=STATS)
    ticks = TickData.from_frame(mk, synthetic_ticks(frame[frame["date"] == frame["date"].min()]))
    assert list(ticks.session_ids(mk)) == [0]
    env = TickExecutionEnv(mk, EnvConfig.load(), ticks, n_envs=1, autoreset=False)
    ep = fixed_episodes(1)
    ep.session[:] = 1
    with pytest.raises(ValueError, match="tick sessions"):
        env.reset(ep)
    assert list(EpisodeSampler(mk, EnvConfig.load().episode, "train", ticks.session_ids(mk)).sessions) == [0]
