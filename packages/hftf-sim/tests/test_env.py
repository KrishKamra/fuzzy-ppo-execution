import numpy as np
import pytest

from hftf_data.paths import REPO_ROOT
from hftf_sim import BatchExecutionEnv, EnvConfig, EpisodeSampler, ExecutionEnv, FeeModel, MarketData
from hftf_sim.config import FeeConfig
from hftf_sim.market import MARKET_COLUMNS, REGIME_COLUMNS
from hftf_sim.testing import STATS, synthetic_frame
from hftf_sim.testing import fixed_episodes as episodes
from hftf_sim.testing import synthetic_market as market


def test_shares_conserved_under_random_pace():
    cfg = EnvConfig.load()
    mk = market()
    sampler = EpisodeSampler(mk, cfg.episode, "train")
    ep = sampler.sample(256, np.random.default_rng(1))
    env = BatchExecutionEnv(mk, cfg, sampler=sampler, n_envs=256, autoreset=False)
    env.reset(ep)
    rng = np.random.default_rng(2)
    total = np.zeros(256)
    while not env.done.all():
        _, reward, _, info = env.step(rng.uniform(0, cfg.action.a_max, size=256))
        assert (info["q"] >= 0).all() and np.isfinite(reward).all()
        assert np.array_equal(info["q"], np.rint(info["q"]))
        total += info["q"]
    assert np.array_equal(total, ep.qty)
    assert (env.remaining == 0).all() and (env.k == ep.horizon).all()


def test_observation_uses_only_past_rows():
    """Changing every row after the decision row leaves that decision's observation unchanged."""
    cfg = EnvConfig.load()
    base = synthetic_frame(n_sessions=1)
    ep = episodes(4, horizon=60, row0=100)
    actions = np.random.default_rng(3).uniform(0, 2, size=(60, 4))
    for k in (0, 1, 17, 59):
        cut = ep.row0[0] + k
        later = base.copy()
        future = later["mso"] > cut
        for c in (*REGIME_COLUMNS, *MARKET_COLUMNS, "mid", "volume", "value", "spread_bps", "rv_30m"):
            later.loc[future, c] = later.loc[future, c] * 1.7 + 0.3
        obs = []
        for frame in (base, later):
            env = BatchExecutionEnv(MarketData.from_frame(frame, stats=STATS), cfg, n_envs=4, autoreset=False)
            o = env.reset(ep)
            for a in actions[:k]:
                o, *_ = env.step(a)
            obs.append(o)
        np.testing.assert_array_equal(obs[0], obs[1])


def test_buy_and_sell_are_symmetric_on_a_flat_market():
    cfg, mk = EnvConfig.load(), market(flat=True)
    costs = []
    for side in (1, -1):
        env = BatchExecutionEnv(
            mk, cfg, fees=FeeModel.zero(), n_envs=1, autoreset=False, randomise_impact=False
        )
        env.reset(episodes(1, side=side))
        while not env.done.all():
            env.step(np.array([1.3]))
        costs.append(env.summary()["cost_bps"][0])
    assert costs[0] == pytest.approx(costs[1])


def test_fee_model_delivery_buy():
    fees = FeeModel(FeeConfig.load())
    got = fees(np.array([1e6]), np.array([1]), np.array([True]))[0]
    # STT 0.1% + stamp 0.015% + GST on (exchange 0.00297% + SEBI Rs10/cr + Rs20 per order)
    assert got == pytest.approx(1000 + 150 + (29.7 + 1.0 + 20.0) * 1.18)


def test_autoreset_and_domain_randomisation():
    cfg, mk = EnvConfig.load(), market()
    sampler = EpisodeSampler(mk, cfg.episode, "train")
    env = BatchExecutionEnv(mk, cfg, sampler=sampler, n_envs=16, seed=4)
    obs = env.reset()
    finished = 0
    for _ in range(300):
        obs, reward, done, info = env.step(np.ones(16))
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        finished += done.sum()
        if done.any():
            assert (info["final"]["executed"] > 0).all()
    assert finished > 16
    lo, hi = cfg.impact.randomise["eta"]
    assert ((env.impact.eta >= lo) & (env.impact.eta <= hi)).all() and env.impact.eta.std() > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_gymnasium_wrapper_passes_env_checker():
    from gymnasium.utils.env_checker import check_env

    cfg, mk = EnvConfig.load(), market()
    check_env(ExecutionEnv(mk, cfg, EpisodeSampler(mk, cfg.episode, "train")), skip_render_check=True)


FEATURES = REPO_ROOT / "data/processed/features_1m_v3.parquet"


@pytest.mark.skipif(not FEATURES.exists(), reason="licensed feature table not present")
def test_real_sessions_give_finite_observations():
    cfg = EnvConfig.load()
    mk = MarketData.load(cfg.features_path)
    for split in ("train", "val", "test"):
        sampler = EpisodeSampler(mk, cfg.episode, split)
        assert set(mk.sessions["split"].iloc[sampler.sessions]) == {split}
        env = BatchExecutionEnv(mk, cfg, sampler=sampler, n_envs=512, autoreset=False)
        obs = env.reset(sampler.sample(512, np.random.default_rng(5)))
        while not env.done.all():
            assert np.isfinite(obs).all()
            obs, reward, *_ = env.step(np.ones(512))
            assert np.isfinite(reward).all()


def test_mark_to_market_reward_sums_to_arrival_is():
    rewards = {}
    for mode in ("arrival", "mark_to_market"):
        cfg, mk = EnvConfig.load(), market()
        cfg.reward.mode = mode
        env = BatchExecutionEnv(mk, cfg, n_envs=8, autoreset=False, randomise_impact=False)
        env.reset(episodes(8, horizon=60, qty=4000.0, row0=50))
        rng, total = np.random.default_rng(6), np.zeros(8)
        while not env.done.all():
            total += env.step(rng.uniform(0, 3, size=8))[1]
        np.testing.assert_allclose(total, -env.summary()["cost_bps"], rtol=1e-9, atol=1e-9)
        rewards[mode] = total
    np.testing.assert_allclose(rewards["arrival"], rewards["mark_to_market"], rtol=1e-9, atol=1e-9)


def test_vs_vwap_reward_is_cost_relative_to_a_separate_vwap_run():
    cfg, mk = EnvConfig.load(), market()
    ep = episodes(8, horizon=60, qty=4000.0, row0=50)
    costs = {}
    for name, act in (("vwap", lambda rng: np.ones(8)), ("agent", lambda rng: rng.uniform(0, 3, size=8))):
        cfg.reward.mode = "vs_vwap"
        env = BatchExecutionEnv(mk, cfg, n_envs=8, autoreset=False, randomise_impact=True)
        env.reset(ep)
        rng, total = np.random.default_rng(7), np.zeros(8)
        while not env.done.all():
            total += env.step(act(rng))[1]
        costs[name] = (env.summary()["cost_bps"], total)
    np.testing.assert_allclose(costs["vwap"][1], 0.0, atol=1e-9)  # VWAP against itself
    np.testing.assert_allclose(costs["agent"][1], costs["vwap"][0] - costs["agent"][0], rtol=1e-9, atol=1e-9)
