"""PPO trainer (ARCHITECTURE §6): clipped surrogate, GAE, value loss, entropy bonus, target-KL early stop.

Training samples train-split episodes with domain-randomised impact. Every `eval_every` iterations the
deterministic policy (Beta mean) and adaptive VWAP (a = 1) run on the same fixed train and val episodes,
at default impact. `best.pt` keeps the iteration with the lowest val objective (early stopping on val).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from hftf_rl.buffer import RolloutBuffer
from hftf_rl.config import PPOConfig
from hftf_rl.normalize import RewardScaler, RunningMeanStd
from hftf_rl.policies import POLICIES, MLPActorCritic
from hftf_sim.config import EnvConfig
from hftf_sim.env import BatchExecutionEnv
from hftf_sim.episodes import Episodes, EpisodeSampler
from hftf_sim.market import MarketData


class PacePolicy:
    """A trained actor-critic as a deterministic env policy (Beta mean); frozen observation normaliser."""

    action_space = "pace"

    def __init__(self, model: MLPActorCritic, obs_rms: RunningMeanStd, name: str = "ppo_mlp") -> None:
        self.model, self.obs_rms, self.name = model, obs_rms, name

    def __call__(self, env: BatchExecutionEnv) -> np.ndarray:
        return self.model.mean_pace(self.obs_rms.normalize(env.observe()))


def evaluate(policy, market: MarketData, env_cfg: EnvConfig, episodes: Episodes) -> dict[str, np.ndarray]:
    """Per-episode costs of a pace policy on a fixed batch at default impact (no hftf-eval dependency)."""
    env = BatchExecutionEnv(market, env_cfg, n_envs=len(episodes), autoreset=False, randomise_impact=False)
    env.reset(episodes)
    while not env.done.all():
        env.step(policy(env))
    return env.summary()


def vwap_policy(env: BatchExecutionEnv) -> np.ndarray:
    return np.ones(env.n)


def save_checkpoint(path: Path, model, obs_rms, ppo_cfg, env_cfg, names, extra: dict) -> None:
    torch.save(
        {
            "policy": "mlp",
            "arch": model.arch,
            "model": model.state_dict(),
            "obs_rms": obs_rms.state_dict(),
            "ppo_cfg": ppo_cfg.model_dump(),
            "env_cfg": env_cfg.model_dump(),
            "obs_names": list(names),
            **extra,
        },
        path,
    )


def load_policy(path: str | Path, name: str = "ppo_mlp") -> tuple[PacePolicy, EnvConfig, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = POLICIES[ckpt["policy"]](**ckpt["arch"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    policy = PacePolicy(model, RunningMeanStd.from_state(ckpt["obs_rms"]), name)
    return policy, EnvConfig.model_validate(ckpt["env_cfg"]), ckpt


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, env_cfg: EnvConfig, market: MarketData, run_dir: str | Path) -> None:
        self.cfg, self.env_cfg, self.market = cfg, env_cfg, market
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.torch_threads)
        sampler = EpisodeSampler(market, env_cfg.episode, "train")
        self.env = BatchExecutionEnv(
            market,
            env_cfg,
            sampler=sampler,
            n_envs=cfg.n_envs,
            randomise_impact=cfg.randomise_impact,
            seed=cfg.seed,
        )
        dim = self.env.obs_dim
        self.model = MLPActorCritic(
            dim,
            cfg.trunk.hidden,
            env_cfg.action.a_max,
            cfg.trunk.activation,
            cfg.init_pace,
            cfg.init_concentration,
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd((dim,))
        self.scaler = RewardScaler(cfg.n_envs, cfg.gamma)
        self.buffer = RolloutBuffer(cfg.rollout_steps, cfg.n_envs, dim)
        rng = np.random.default_rng(cfg.eval_seed)
        self.eval_sets = {
            s: EpisodeSampler(market, env_cfg.episode, s).sample(cfg.eval_episodes, rng)
            for s in ("train", "val")
        }
        self.vwap = {s: evaluate(vwap_policy, market, env_cfg, ep) for s, ep in self.eval_sets.items()}
        self.obs = self.env.reset()
        self.episode_costs: list[float] = []
        self.best = np.inf

    # ---------------------------------------------------------------- rollout
    def collect(self) -> None:
        self.buffer.reset()
        a_max = self.env_cfg.action.a_max
        for _ in range(self.cfg.rollout_steps):
            self.obs_rms.update(self.obs)
            obs_n = self.obs_rms.normalize(self.obs).astype(np.float32)
            u, logp, value = self.model.sample(obs_n)
            self.obs, reward, done, info = self.env.step(u * a_max)
            self.buffer.add(obs_n, u, logp, value, self.scaler(reward, done), done)
            if "final" in info:
                self.episode_costs.extend(info["final"]["cost_bps"].tolist())
        last = self.obs_rms.normalize(self.obs).astype(np.float32)
        with torch.no_grad():
            self.last_value = self.model.value(torch.as_tensor(last)).numpy()

    # ----------------------------------------------------------------- update
    def update(self) -> dict[str, float]:
        cfg, b = self.cfg, self.buffer
        adv, ret = b.gae(self.last_value, cfg.gamma, cfg.gae_lambda)
        flat = {
            "obs": torch.as_tensor(b.obs.reshape(-1, b.obs.shape[-1])),
            "u": torch.as_tensor(b.u.reshape(-1)),
            "logp": torch.as_tensor(b.logp.reshape(-1)),
            "adv": torch.as_tensor(adv.reshape(-1)),
            "ret": torch.as_tensor(ret.reshape(-1)),
        }
        n = len(flat["u"])
        mb = n // cfg.minibatches
        stats: dict[str, list[float]] = {k: [] for k in ("pg_loss", "v_loss", "entropy", "kl", "clipfrac")}
        stop = False
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for i in range(cfg.minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                dist, v = self.model(flat["obs"][idx])
                logp = dist.log_prob(flat["u"][idx])
                log_ratio = logp - flat["logp"][idx]
                ratio = log_ratio.exp()
                a = flat["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v_loss = 0.5 * (flat["ret"][idx] - v).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + cfg.value_coef * v_loss - cfg.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = float(((ratio - 1) - log_ratio).mean())
                    clipfrac = float(((ratio - 1).abs() > cfg.clip).float().mean())
                for k, x in zip(stats, (pg.item(), v_loss.item(), ent.item(), kl, clipfrac), strict=True):
                    stats[k].append(x)
                if kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(x)) for k, x in stats.items()}
        y, pred = ret.reshape(-1), b.value.reshape(-1)
        out["explained_var"] = float(1 - np.var(y - pred) / (np.var(y) + 1e-8))
        out["mean_pace"] = float(b.u.mean() * self.env_cfg.action.a_max)
        out["early_stop"] = float(stop)
        return out

    # ------------------------------------------------------------- evaluation
    def evaluate(self) -> dict[str, float]:
        policy = PacePolicy(self.model, self.obs_rms)
        out = {}
        for split, ep in self.eval_sets.items():
            cost = evaluate(policy, self.market, self.env_cfg, ep)["cost_bps"]
            d = cost - self.vwap[split]["cost_bps"]
            out[f"{split}_cost"] = float(cost.mean())
            out[f"{split}_vwap_cost"] = float(self.vwap[split]["cost_bps"].mean())
            out[f"{split}_vs_vwap"] = float(d.mean())
            out[f"{split}_vs_vwap_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        return out

    def _save(self, name: str, iteration: int, metrics: dict) -> None:
        extra = {"iteration": iteration, "metrics": metrics}
        save_checkpoint(
            self.run_dir / name, self.model, self.obs_rms, self.cfg, self.env_cfg, self.env.names, extra
        )

    def train(self, log=print) -> dict[str, float]:
        cfg = self.cfg
        per_iter = cfg.n_envs * cfg.rollout_steps
        iterations = max(1, cfg.total_timesteps // per_iter)
        (self.run_dir / "config.json").write_text(
            json.dumps({"ppo": cfg.model_dump(), "env": self.env_cfg.model_dump()}, indent=2)
        )
        log_path = self.run_dir / "progress.csv"
        writer, fh, best_metrics = None, log_path.open("w", newline=""), {}
        t_start = time.time()
        try:
            for it in range(1, iterations + 1):
                self.episode_costs = []
                if cfg.lr_anneal:
                    for g in self.opt.param_groups:
                        g["lr"] = cfg.lr * (1 - (it - 1) / iterations)
                self.collect()
                row = {"iteration": it, "timesteps": it * per_iter, **self.update()}
                costs = self.episode_costs
                row["train_episode_cost"] = float(np.mean(costs)) if costs else np.nan
                if it % cfg.eval_every == 0 or it == iterations:
                    ev = self.evaluate()
                    row.update(ev)
                    if ev["val_cost"] < self.best:
                        self.best, best_metrics = ev["val_cost"], {"iteration": it, **ev}
                        self._save("best.pt", it, best_metrics)
                    self._save("last.pt", it, ev)
                    log(
                        f"it {it:4d}/{iterations} steps {it * per_iter:>9,} pace {row['mean_pace']:.2f} "
                        f"kl {row['kl']:.4f} ev {row['explained_var']:.2f} | vs VWAP (bps) "
                        f"train {ev['train_vs_vwap']:+.3f}±{ev['train_vs_vwap_se']:.3f} "
                        f"val {ev['val_vs_vwap']:+.3f}±{ev['val_vs_vwap_se']:.3f} "
                        f"| {time.time() - t_start:.0f}s"
                    )
                row["elapsed_s"] = time.time() - t_start
                if writer is None:
                    keys = [*row.keys(), *(k for k in self.evaluate_keys() if k not in row)]
                    writer = csv.DictWriter(fh, fieldnames=keys, restval="")
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
        finally:
            fh.close()
        (self.run_dir / "best.json").write_text(json.dumps(best_metrics, indent=2))
        return best_metrics

    @staticmethod
    def evaluate_keys() -> list[str]:
        return [f"{s}_{k}" for s in ("train", "val") for k in ("cost", "vwap_cost", "vs_vwap", "vs_vwap_se")]
