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

from hftf_fuzzy.evolution import Evolver, kmeans
from hftf_fuzzy.layer import remap_optimizer_state
from hftf_rl.buffer import RolloutBuffer
from hftf_rl.config import EFNNConfig, PPOConfig
from hftf_rl.normalize import RewardScaler, RunningMeanStd
from hftf_rl.policies import POLICIES, FuzzyActorCritic, MLPActorCritic
from hftf_sim.config import EnvConfig
from hftf_sim.env import REGIME_SLICE, BatchExecutionEnv
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
            "policy": model.kind,
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


def policy_name(ckpt: dict) -> str:
    """ppo_mlp, ppo_hard_gate (one-hot rules), ppo_anfis (fuzzy, evolution off) or ppo_efnn."""
    if ckpt["policy"] == "mlp":
        return "ppo_mlp"
    if ckpt["arch"].get("gate") == "hard":
        return "ppo_hard_gate"
    evolving = (ckpt.get("efnn_cfg") or {}).get("evolution", {}).get("enabled", True)
    return "ppo_efnn" if evolving else "ppo_anfis"


def load_policy(path: str | Path, name: str | None = None) -> tuple[PacePolicy, EnvConfig, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = POLICIES[ckpt["policy"]](**ckpt["arch"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    policy = PacePolicy(model, RunningMeanStd.from_state(ckpt["obs_rms"]), name or policy_name(ckpt))
    return policy, EnvConfig.model_validate(ckpt["env_cfg"]), ckpt


class EnsemblePolicy:
    """Mean pace of several trained policies (e.g. one per seed). Each member normalises its own input."""

    action_space = "pace"

    def __init__(self, members: list[PacePolicy], name: str) -> None:
        self.members, self.name = members, name

    def __call__(self, env: BatchExecutionEnv) -> np.ndarray:
        return np.mean([m(env) for m in self.members], axis=0)


def load_ensemble(paths: list[str | Path], name: str) -> EnsemblePolicy:
    return EnsemblePolicy([load_policy(p)[0] for p in paths], name)


class PPOTrainer:
    def __init__(
        self,
        cfg: PPOConfig,
        env_cfg: EnvConfig,
        market: MarketData,
        run_dir: str | Path,
        efnn: EFNNConfig | None = None,
    ) -> None:
        self.cfg, self.env_cfg, self.market = cfg, env_cfg, market
        self.efnn = efnn or EFNNConfig()
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
        common = (dim, cfg.trunk.hidden, env_cfg.action.a_max, cfg.trunk.activation)
        common = (*common, cfg.init_pace, cfg.init_concentration)
        limit = {
            "limit": env_cfg.limit.enabled,
            "init_passive": cfg.init_passive,
            "passive_concentration": cfg.passive_concentration,
        }
        self.fuzzy = cfg.policy == "fuzzy"
        if self.fuzzy:
            self.model = FuzzyActorCritic(
                *common,
                n_rules=self.efnn.init_rules,
                regime_dim=REGIME_SLICE.stop,
                antecedent_grad=self.efnn.antecedent_grad,
                gate=self.efnn.gate,
                **limit,
            )
            if self.efnn.gate == "hard" and self.efnn.evolution.enabled:
                raise ValueError("gate: hard needs evolution off (merges and adds assume soft firing)")
            ante = self.model.fuzzy.antecedent_parameters()
            rest = [p for p in self.model.parameters() if all(p is not a for a in ante)]
            groups = [{"params": rest}, {"params": ante, "lr_scale": self.efnn.antecedent_lr_scale}]
            self.evolver = Evolver(self.efnn.evolution_config(), self.efnn.init_rules)
            self.events_path = self.run_dir / "events.csv"
            self.events_path.write_text("iteration,kind,rules,n_rules,accepted,kl,detail\n")
            self.initialised = False
        else:
            self.model = MLPActorCritic(*common, **limit)
            groups = [{"params": list(self.model.parameters())}]
        for g in groups:
            g["lr"] = cfg.lr * g.get("lr_scale", 1.0)
        self.opt = torch.optim.Adam(groups, lr=cfg.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd((dim,), passthrough=REGIME_SLICE.stop)
        self.scaler = RewardScaler(cfg.n_envs, cfg.gamma)
        action_shape = (2,) if env_cfg.limit.enabled else ()
        self.buffer = RolloutBuffer(cfg.rollout_steps, cfg.n_envs, dim, action_shape)
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
        for _ in range(self.cfg.rollout_steps):
            self.obs_rms.update(self.obs)
            obs_n = self.obs_rms.normalize(self.obs).astype(np.float32)
            u, logp, value = self.model.sample(obs_n)
            self.obs, reward, done, info = self.env.step(self.model.to_action(u))
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
            "u": torch.as_tensor(b.u.reshape(-1, *b.u.shape[2:])),
            "logp": torch.as_tensor(b.logp.reshape(-1)),
            "adv": torch.as_tensor(adv.reshape(-1)),
            "ret": torch.as_tensor(ret.reshape(-1)),
        }
        n = len(flat["logp"])
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
        u = b.u.reshape(-1, *b.u.shape[2:])
        out["mean_pace"] = float((u[:, 0] if u.ndim == 2 else u).mean() * self.env_cfg.action.a_max)
        if u.ndim == 2:
            out["mean_passive"] = float(u[:, 1].mean())
        out["early_stop"] = float(stop)
        return out

    # ---------------------------------------------------------- fuzzy layer
    def _regime_samples(self) -> torch.Tensor:
        x = self.buffer.obs[..., REGIME_SLICE].reshape(-1, REGIME_SLICE.stop)
        return torch.as_tensor(x)

    def init_rules(self) -> None:
        """R0 rules from k-means on the first rollout's regime inputs. Function-preserving, because every
        rule starts with the same consequents."""
        centres, std = kmeans(self._regime_samples(), self.efnn.init_rules, seed=self.cfg.seed)
        layer = self.model.fuzzy
        snap = layer.snapshot()
        snap["centres"] = centres
        width = (std * self.efnn.init_width_scale).clamp_min(self.efnn.evolution.min_width)
        snap["rho"] = torch.log(torch.expm1(width))
        layer.restore(snap)
        self.initialised = True

    def evolve(self, iteration: int) -> dict[str, float]:
        """One structure-evolution pass after the PPO update, before the next rollout (ARCHITECTURE §5.3).
        The pass's policy change is measured as KL(pi_before || pi_after) on buffer observations; a pass
        above `edit_kl_max` is undone. Accepted edits keep Adam moments for surviving rules."""
        layer, n = self.model.fuzzy, self.buffer.obs.shape[0] * self.buffer.obs.shape[1]
        obs = torch.as_tensor(self.buffer.obs.reshape(n, -1))
        idx = torch.randperm(n)[: self.efnn.edit_kl_samples]
        with torch.no_grad():
            before, _ = self.model(obs[idx])
        snap, state = layer.snapshot(), self.evolver.state()
        events = self.evolver.step(layer, self._regime_samples())
        kl, accepted = 0.0, True
        if events:
            with torch.no_grad():
                after, _ = self.model(obs[idx])
                kl = float(torch.distributions.kl_divergence(before, after).mean())
            accepted = kl <= self.efnn.edit_kl_max
            if accepted:
                remap_optimizer_state(self.opt, layer)
            else:
                layer.restore(snap)
                self.evolver.restore(state)
                self.evolver.cooldown = self.evolver.cfg.cooldown_iters
            with self.events_path.open("a") as fh:
                for e in events:
                    rules = " ".join(map(str, e.rules))
                    detail = json.dumps(e.detail).replace(",", ";")
                    fh.write(f"{iteration},{e.kind},{rules},{e.n_rules},{int(accepted)},{kl:.6f},{detail}\n")
        return {"n_rules": float(layer.n_rules), "edit_kl": kl, "edits": float(len(events) * accepted)}

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
        extra = {
            "iteration": iteration,
            "metrics": metrics,
            "market_stats": self.market.stats,  # the observation scaler: reused when evaluating
            "securities": sorted(self.market.sessions["security"].unique()),
        }
        if self.fuzzy:
            extra["efnn_cfg"] = self.efnn.model_dump()
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
                        g["lr"] = cfg.lr * g.get("lr_scale", 1.0) * (1 - (it - 1) / iterations)
                self.collect()
                if self.fuzzy and not self.initialised:
                    self.init_rules()
                row = {"iteration": it, "timesteps": it * per_iter, **self.update()}
                if self.fuzzy:
                    row.update(self.evolve(it))
                costs = self.episode_costs
                row["train_episode_cost"] = float(np.mean(costs)) if costs else np.nan
                if it % cfg.eval_every == 0 or it == iterations:
                    ev = self.evaluate()
                    row.update(ev)
                    if ev["val_cost"] < self.best:
                        self.best, best_metrics = ev["val_cost"], {"iteration": it, **ev}
                        self._save("best.pt", it, best_metrics)
                    self._save("last.pt", it, ev)
                    passive = f"passive {row['mean_passive']:.2f} " if "mean_passive" in row else ""
                    log(
                        f"it {it:4d}/{iterations} steps {it * per_iter:>9,} pace {row['mean_pace']:.2f} "
                        f"{passive}kl {row['kl']:.4f} ev {row['explained_var']:.2f} | vs VWAP (bps) "
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
