import numpy as np
import pytest

from hftf_sim import BatchExecutionEnv, EnvConfig, EpisodeSampler, FeeModel, MarketData, TickData
from hftf_sim.limit import bar_passive_fill, queue_passive_fill
from hftf_sim.testing import STATS, fixed_episodes, synthetic_frame, synthetic_ticks
from hftf_sim.tick import TickExecutionEnv

NAN = np.nan


def test_bar_fill_through_touched_and_missed():
    side = np.array([1, 1, 1, -1, -1, -1])
    price = np.full(6, 100.0)
    low = np.array([99.9, 100.0, 100.1, 99.0, 99.0, 99.0])
    high = np.array([101.0, 101.0, 101.0, 100.1, 100.0, 99.9])
    post = np.full(6, 50.0)
    filled = bar_passive_fill(post, side, price, low, high, np.full(6, 400.0), 0.05, 0.1)
    np.testing.assert_array_equal(filled, [40, 20, 0, 40, 20, 0])  # through: 0.1·400; touched: 0.05·400


def test_queue_fill_consumes_the_queue_ahead_first():
    side, price, queue, post = np.array([1, 1, -1]), np.full(3, 100.0), np.full(3, 50.0), np.full(3, 40.0)
    px_min = np.array([[100.0, NAN, 100.0], [100.0, 99.9, NAN], [100.0, 100.0, 100.0]])
    px_max = np.array([[100.0, NAN, 100.2], [100.0, 100.0, NAN], [100.0, 100.0, 100.0]])
    sell = np.array([[30.0, 0.0, 40.0], [10.0, 5.0, 0.0], [99.0, 99.0, 99.0]])
    buy = np.array([[999.0, 0.0, 999.0], [0.0, 0.0, 0.0], [20.0, 25.0, 0.0]])
    filled = queue_passive_fill(post, side, price, queue, px_min, px_max, buy, sell)
    # buy: 70 sold at our price − 50 ahead = 20 | buy: a print through 100 clears the queue, but only 15
    # traded at or through it | sell: 45 bought − 50 ahead → 0
    np.testing.assert_array_equal(filled, [20, 15, 0])


def _market(flat=True):
    frame = synthetic_frame(n_sessions=2, flat=flat)
    return frame, MarketData.from_frame(frame, stats=STATS)


def test_zero_passive_fraction_is_the_pace_only_action():
    _, mk = _market(flat=False)
    cfg = EnvConfig.load()
    ep = EpisodeSampler(mk, cfg.episode, "train").sample(64, np.random.default_rng(0))
    a = np.random.default_rng(1).uniform(0, 3, size=(120, 64))
    out = []
    for two_d in (False, True):
        env = BatchExecutionEnv(mk, cfg, n_envs=64, autoreset=False)
        env.reset(ep)
        r = [env.step(np.stack([x, np.zeros(64)], 1) if two_d else x)[1] for x in a[: int(ep.horizon.max())]]
        out.append((np.array(r), env.summary()["cost_bps"]))
    np.testing.assert_array_equal(out[0][0], out[1][0])
    np.testing.assert_array_equal(out[0][1], out[1][1])


def test_passive_child_fills_at_the_decision_bid_on_a_flat_market():
    _, mk = _market()
    cfg = EnvConfig.load()
    cfg.limit.touch_fill_frac = cfg.limit.through_fill_frac = 1.0
    env = BatchExecutionEnv(mk, cfg, fees=FeeModel.zero(), n_envs=1, autoreset=False, randomise_impact=False)
    env.reset(fixed_episodes(1, horizon=30, qty=3000.0))
    _, _, _, info = env.step(np.array([[1.0, 1.0]]))
    assert info["posted"][0] == 100 and info["passive_filled"][0] == 100
    np.testing.assert_allclose(info["price"][0], 1000.0 * (1 - 1e-4))  # bid of a 2 bps market
    np.testing.assert_allclose(env.summary()["is_bps"], -1.0 * 100 / 3000)


@pytest.mark.parametrize("tick", [False, True])
def test_passive_orders_conserve_shares_and_mark_to_market_telescopes(tick):
    frame, mk = _market(flat=False)
    cfg = EnvConfig.load()
    cfg.reward.mode, cfg.reward.price_drift, cfg.reward.risk_aversion = "mark_to_market", "realised", 0.0
    kw = dict(fees=FeeModel.zero(), n_envs=64, autoreset=False)
    if tick:
        ticks_df = synthetic_ticks(frame, depth=30.0)
        rng = np.random.default_rng(5)  # random prints at the touch, some through it
        ticks_df["sell_vol"] = rng.integers(0, 40, len(ticks_df)).astype(float)
        ticks_df["buy_vol"] = rng.integers(0, 40, len(ticks_df)).astype(float)
        ticks_df["px_min"] = ticks_df["bid"] - rng.choice([0.0, 0.1], len(ticks_df), p=[0.97, 0.03])
        ticks_df["px_max"] = ticks_df["ask"] + rng.choice([0.0, 0.1], len(ticks_df), p=[0.97, 0.03])
        env = TickExecutionEnv(mk, cfg, TickData.from_frame(mk, ticks_df), **kw)
    else:
        env = BatchExecutionEnv(mk, cfg, **kw)
    ep = EpisodeSampler(mk, cfg.episode, "train").sample(64, np.random.default_rng(3))
    env.reset(ep)
    rng = np.random.default_rng(4)
    total, ret, passive = np.zeros(64), np.zeros(64), np.zeros(64)
    while not env.done.all():
        _, r, _, info = env.step(np.stack([rng.uniform(0, 4, 64), rng.uniform(0, 1, 64)], 1))
        assert (info["passive_filled"] <= info["posted"]).all()
        total += info["q"]
        ret += r
        passive += info["passive_filled"]
    s = env.summary()
    np.testing.assert_array_equal(total, ep.qty)
    assert (env.remaining == 0).all() and passive.sum() > 0
    np.testing.assert_allclose(s["passive_frac"], passive / ep.qty)
    np.testing.assert_allclose(-ret, s["is_bps"], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(("fraction", "posted"), [(0.05, 0), (0.1, 0), (0.5, 50), (0.9, 100), (0.95, 100)])
def test_dead_zone_snaps_the_passive_fraction_to_one_order(fraction, posted):
    _, mk = _market()
    cfg = EnvConfig.load()
    assert cfg.limit.dead_zone == 0.1
    env = BatchExecutionEnv(mk, cfg, fees=FeeModel.zero(), n_envs=1, autoreset=False, randomise_impact=False)
    env.reset(fixed_episodes(1, horizon=30, qty=3000.0))
    _, _, _, info = env.step(np.array([[1.0, fraction]]))  # a 100-share child
    assert info["posted"][0] == posted
