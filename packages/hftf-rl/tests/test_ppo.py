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


def test_fuzzy_policy_starts_as_noisy_vwap_and_respects_antecedent_grad():
    from hftf_rl.policies import FuzzyActorCritic

    obs = torch.as_tensor(np.random.default_rng(2).normal(size=(128, 15)).astype(np.float32))
    for mode in ("all", "value_only", "none"):
        model = FuzzyActorCritic(15, [64, 64], 4.0, n_rules=4, antecedent_grad=mode)
        np.testing.assert_allclose(model.mean_pace(obs.numpy()), 1.0, atol=0.05)
        with torch.no_grad():  # different antecedents, same consequents: same output
            model.fuzzy.centres.normal_()
        np.testing.assert_allclose(model.mean_pace(obs.numpy()), 1.0, atol=0.05)
        with torch.no_grad():
            model.fuzzy.actor_b.normal_()  # rules now disagree, so phi matters
        actor_grad = torch.autograd.grad(model(obs)[0].mean.sum(), model.fuzzy.centres, allow_unused=True)[0]
        critic_grad = torch.autograd.grad(model(obs)[1].sum(), model.fuzzy.centres, allow_unused=True)[0]
        assert (actor_grad is not None and actor_grad.abs().sum() > 0) == (mode == "all")
        assert (critic_grad is not None) == (mode != "none")


def test_fuzzy_trainer_logs_events_and_checkpoint_keeps_the_evolved_structure(tmp_path):
    from hftf_rl.config import EFNNConfig, EvolutionSettings

    df = synthetic_frame(n_sessions=3)
    df.loc[df["date"] == df["date"].max(), "split"] = "val"
    market = MarketData.from_frame(df, stats=STATS)
    cfg = PPOConfig(
        policy="fuzzy", n_envs=8, rollout_steps=40, total_timesteps=8 * 40 * 4, epochs=2, minibatches=2,
        eval_every=1, eval_episodes=16, torch_threads=1,
    )  # fmt: skip
    efnn = EFNNConfig(evolution=EvolutionSettings(tau_add=0.99, n_min_uncovered=1, cooldown_iters=0))
    trainer = PPOTrainer(cfg, EnvConfig.load(), market, tmp_path, efnn=efnn)
    trainer.train(log=lambda *_: None)
    events = (tmp_path / "events.csv").read_text().strip().splitlines()
    assert events[0].startswith("iteration,kind") and len(events) > 1
    assert all(float(line.split(",")[5]) >= 0 for line in events[1:])
    policy, _, ckpt = load_policy(tmp_path / "last.pt")
    assert policy.name == "ppo_efnn"
    assert policy.model.fuzzy.n_rules == ckpt["arch"]["n_rules"] == trainer.model.fuzzy.n_rules
    assert policy.obs_rms.passthrough == 3


def test_hard_gate_uses_one_rule_per_decision():
    from hftf_rl.policies import FuzzyActorCritic

    obs = torch.as_tensor(np.random.default_rng(3).normal(size=(64, 15)).astype(np.float32))
    model = FuzzyActorCritic(15, [64, 64], 4.0, n_rules=3, gate="hard")
    with torch.no_grad():
        model.fuzzy.centres.normal_()
        model.fuzzy.actor_b.normal_()
    dist, _ = model(obs)
    h, x = model.trunk(obs), model.regime(obs)
    best = model.fuzzy.phi(x).argmax(-1)
    own = model.fuzzy.consequents("actor", h)[torch.arange(64), best]
    expected = 1.0 + torch.nn.functional.softplus(own)
    torch.testing.assert_close(dist.concentration1, expected[:, 0])
    assert torch.autograd.grad(dist.mean.sum(), model.fuzzy.centres, allow_unused=True)[0] is None


def test_limit_head_starts_at_vwap_pace_and_initial_passive_fraction():
    from hftf_rl.policies import FuzzyActorCritic

    obs = np.random.default_rng(4).normal(size=(256, 15)).astype(np.float32)
    for model in (
        MLPActorCritic(15, [64, 64], 4.0, limit=True, init_passive=0.2),
        FuzzyActorCritic(15, [64, 64], 4.0, n_rules=4, limit=True, init_passive=0.2),
    ):
        a = model.mean_pace(obs)
        assert a.shape == (256, 2)
        np.testing.assert_allclose(a.mean(0), [1.0, 0.2], atol=0.05)
        u, logp, _ = model.sample(obs)
        assert u.shape == (256, 2) and logp.shape == (256,)  # one joint log-prob per decision
        dist, _ = model(torch.as_tensor(obs))
        np.testing.assert_allclose(
            dist.log_prob(torch.as_tensor(u)).detach().numpy(), logp, rtol=1e-5, atol=1e-5
        )


def test_limit_trainer_runs_for_mlp_and_efnn_and_checkpoints_round_trip(tmp_path):
    from hftf_rl.config import EFNNConfig, EvolutionSettings
    from hftf_sim.episodes import EpisodeSampler

    df = synthetic_frame(n_sessions=3)
    df.loc[df["date"] == df["date"].max(), "split"] = "val"
    market = MarketData.from_frame(df, stats=STATS)
    env_cfg = EnvConfig.load()
    env_cfg.limit.enabled = True
    efnn = EFNNConfig(evolution=EvolutionSettings(tau_add=0.99, n_min_uncovered=1, cooldown_iters=0))
    for policy_kind in ("mlp", "fuzzy"):
        run = tmp_path / policy_kind
        cfg = PPOConfig(
            policy=policy_kind, n_envs=8, rollout_steps=40, total_timesteps=8 * 40 * 3, epochs=2,
            minibatches=2, eval_every=1, eval_episodes=16, torch_threads=1,
        )  # fmt: skip
        PPOTrainer(cfg, env_cfg, market, run, efnn=efnn).train(log=lambda *_: None)
        assert "mean_passive" in (run / "progress.csv").read_text().splitlines()[0]
        policy, cfg_loaded, ckpt = load_policy(run / "last.pt")
        assert ckpt["arch"]["limit"] and cfg_loaded.limit.enabled
        ep = EpisodeSampler(market, cfg_loaded.episode, "val").sample(16, np.random.default_rng(0))
        summary = evaluate(policy, market, cfg_loaded, ep)
        assert np.array_equal(summary["executed"], ep.qty) and (summary["passive_frac"] > 0).any()


def test_passive_head_can_start_at_the_naive_always_passive_rule():
    import pytest

    from hftf_rl.policies import FuzzyActorCritic

    obs = np.random.default_rng(5).normal(size=(512, 15)).astype(np.float32)
    kw = dict(limit=True, init_passive=0.95, passive_concentration=30.0)
    for model in (
        MLPActorCritic(15, [64, 64], 4.0, **kw),
        FuzzyActorCritic(15, [64, 64], 4.0, n_rules=4, **kw),
    ):
        np.testing.assert_allclose(model.mean_pace(obs).mean(0), [1.0, 0.95], atol=0.03)
        u, _, _ = model.sample(obs)
        assert (u[:, 1] >= 0.9).mean() > 0.85  # most initial children are a single passive order
    with pytest.raises(ValueError):  # the shared concentration (8) cannot put the mean at 0.95
        MLPActorCritic(15, [64, 64], 4.0, limit=True, init_passive=0.95)


def test_fuzzy_policy_explains_its_decisions_in_the_trace():
    from hftf_eval.trace import run_trace
    from hftf_rl.policies import FuzzyActorCritic
    from hftf_rl.ppo import PacePolicy
    from hftf_sim.episodes import order_episode

    market = MarketData.from_frame(synthetic_frame(n_sessions=2), stats=STATS)
    cfg = EnvConfig.load()
    dim = 16  # the env's observation: regime (3) + private (7) + market (5) + next-5-min share
    fuzzy = FuzzyActorCritic(dim, [64, 64], 4.0, n_rules=4)
    mlp = MLPActorCritic(dim, [64, 64], 4.0)
    rms = RunningMeanStd((dim,), passthrough=3)
    ep = order_episode(market, cfg.episode, 0, "10:00", 30, "buy", 0.05)
    out = run_trace([PacePolicy(fuzzy, rms, "efnn"), PacePolicy(mlp, rms, "mlp")], market, cfg, ep)
    phi = np.array(out["policies"][0]["steps"]["phi"])
    assert phi.shape == (30, 4) and np.allclose(phi.sum(1), 1.0, atol=1e-5)
    assert "phi" not in out["policies"][1]["steps"]  # an MLP has nothing to explain
