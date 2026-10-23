import numpy as np
import torch

from hftf_rl.buffer import RolloutBuffer
from hftf_rl.config import PPOConfig
from hftf_rl.normalize import RunningMeanStd
from hftf_rl.policies import MLPActorCritic
from hftf_rl.ppo import PPOTrainer, evaluate, load_policy
from hftf_sim import EnvConfig, MarketData
from hftf_sim.testing import STATS, synthetic_frame


def test_gae_with_lambda_one_is_reward_to_go_within_episodes():
    b = RolloutBuffer(steps=4, n_envs=1, obs_dim=1)
    for r, d in ((1.0, False), (2.0, True), (3.0, False), (4.0, False)):
        b.add(np.zeros(1), 0.5, 0.0, 0.0, r, d)
    adv, ret = b.gae(last_value=np.array([10.0]), gamma=1.0, lam=1.0)
    np.testing.assert_allclose(ret[:, 0], [3.0, 2.0, 17.0, 14.0])  # no bootstrap across the done
    np.testing.assert_allclose(adv, ret)  # values are zero


def test_running_mean_std_matches_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 2))
    rms = RunningMeanStd((2,))
    for chunk in np.array_split(x, 7):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, x.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(rms.var, x.var(axis=0), rtol=1e-4)


def test_initial_policy_is_noisy_vwap():
    model = MLPActorCritic(obs_dim=15, hidden=[64, 64], a_max=4.0, init_pace=1.0, init_concentration=8.0)
    obs = np.random.default_rng(1).normal(size=(256, 15)).astype(np.float32)
    np.testing.assert_allclose(model.mean_pace(obs), 1.0, atol=0.05)
    u, logp, _ = model.sample(obs)
    assert ((u > 0) & (u < 1)).all() and np.isfinite(logp).all()
    dist, _ = model(torch.as_tensor(obs))
    assert torch.isfinite(dist.log_prob(torch.tensor(1e-6))).all()


def test_trainer_runs_and_checkpoint_round_trips(tmp_path):
    df = synthetic_frame(n_sessions=3)
    df.loc[df["date"] == df["date"].max(), "split"] = "val"
    market = MarketData.from_frame(df, stats=STATS)
    env_cfg = EnvConfig.load()
    cfg = PPOConfig(
        n_envs=8, rollout_steps=40, total_timesteps=8 * 40 * 3, epochs=2, minibatches=2,
        eval_every=1, eval_episodes=16, torch_threads=1,
    )  # fmt: skip
    best = PPOTrainer(cfg, env_cfg, market, tmp_path).train(log=lambda *_: None)
    assert {"train_vs_vwap", "val_vs_vwap"} <= best.keys()
    assert (tmp_path / "best.pt").exists() and (tmp_path / "progress.csv").exists()
    policy, cfg_loaded, ckpt = load_policy(tmp_path / "best.pt")
    assert ckpt["obs_names"][:3] == ["imb_bar_proxy_ema5", "log_spread_var_30m", "log_rv_30m"]
    from hftf_sim.episodes import EpisodeSampler

    ep = EpisodeSampler(market, cfg_loaded.episode, "val").sample(16, np.random.default_rng(cfg.eval_seed))
    summary = evaluate(policy, market, cfg_loaded, ep)
    assert np.array_equal(summary["executed"], ep.qty)
