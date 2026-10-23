"""`hftf` command-line entry point."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from hftf_data.paths import DataPaths

app = typer.Typer(no_args_is_help=True, help="Regime-aware execution (Fuzzy PPO) toolkit.")
data_app = typer.Typer(no_args_is_help=True, help="Data validation and feature building.")
app.add_typer(data_app, name="data")
eval_app = typer.Typer(no_args_is_help=True, help="Backtests of baselines and policies in the simulator.")
app.add_typer(eval_app, name="eval")
train_app = typer.Typer(no_args_is_help=True, help="Train policies (PPO).")
app.add_typer(train_app, name="train")
console = Console()


@data_app.command("audit")
def audit(
    strict: bool = typer.Option(False, help="Exit with code 1 if any readiness check fails."),
) -> None:
    """Validate the raw pull and the final feature table (R01-R12); write docs/data/data_audit.json."""
    import pandas as pd
    from rich.table import Table

    from hftf_data.features import FEATURES_FILE
    from hftf_data.readiness import evaluate, load_thresholds, summary
    from hftf_data.validate import run_audit

    paths = DataPaths()
    report = run_audit(paths)
    feat_path = paths.processed / FEATURES_FILE
    features = pd.read_parquet(feat_path) if feat_path.exists() else None
    checks = evaluate(report, load_thresholds(), features)
    report["readiness"] = summary(checks)
    paths.reports.mkdir(parents=True, exist_ok=True)
    out = paths.reports / "data_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    ticks = report["ticks"]
    full = [r for r in ticks if r["clean_over_bar"] == r["clean_over_bar"]]
    console.print(f"[bold]audit written:[/bold] {out}")
    console.print(
        f"tick days: {len(ticks)}  clean/bar volume ratio range: "
        f"{min(r['clean_over_bar'] for r in full):.4f} - {max(r['clean_over_bar'] for r in full):.4f}"
    )
    for r in report["imbalance_proxy"]:
        console.print(f"imbalance proxy {r['security']}: pearson={r['pearson']} spearman={r['spearman']}")
    for r in report["tick_size"]:
        rule = r["nse_rule_tick_at_median_px"]
        console.print(f"tick size {r['security']}: inferred={r['inferred_tick']} nse_rule={rule}")

    table = Table(title="Data readiness")
    for col in ("id", "check", "result", "measured", "required", "detail"):
        table.add_column(col, overflow="fold")
    for c in checks:
        result = "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]"
        table.add_row(c.id, c.name, result, c.measured, c.required, c.detail)
    console.print(table)
    ready = report["readiness"]["ready"]
    n_fail = sum(not c.passed for c in checks)
    console.print(
        "[bold green]READY[/bold green]" if ready else f"[bold red]NOT READY[/bold red] ({n_fail} failing)"
    )
    if strict and not ready:
        raise typer.Exit(code=1)


@data_app.command("build-features")
def build_features() -> None:
    """Build the final causal 1-minute feature table (v3) and its manifest into data/processed/."""
    from hftf_data.features import FEATURES_FILE, build_feature_table, manifest
    from hftf_data.readiness import load_thresholds

    paths = DataPaths()
    df = build_feature_table(paths=paths)
    paths.processed.mkdir(parents=True, exist_ok=True)
    out = paths.processed / FEATURES_FILE
    df.to_parquet(out, index=False)
    th = load_thresholds()
    m = manifest(df, paths, th["episode_columns"], th["first_decision_mso"])
    out.with_suffix(".manifest.json").write_text(json.dumps(m, indent=2, default=str))
    console.print(f"[bold]{out}[/bold]: {len(df):,} rows x {df.shape[1]} cols")
    console.print(f"sessions by split: {m['sessions_by_split']}")
    console.print(f"dropped (incomplete) sessions: {m['dropped_sessions']}")
    console.print(
        f"episode rows: {m['episode_rows']:,}; NaN in episode columns: {sum(m['episode_nan'].values())}"
    )


@eval_app.command("baselines")
def eval_baselines(
    split: str = typer.Option("val", help="train | val | test"),
    episodes: int = typer.Option(4000, help="Parent orders, shared by every baseline (paired)."),
    seed: int = typer.Option(0, help="Episode seed."),
    randomise_impact: bool = typer.Option(False, help="Draw impact per episode from the training ranges."),
) -> None:
    """Immediate, TWAP, VWAP, POV10 and Almgren-Chriss on identical episodes (costs in bps of arrival)."""
    import pandas as pd
    from rich.table import Table

    from hftf_eval.backtest import run_policy, sample_episodes
    from hftf_eval.baselines import default_baselines
    from hftf_eval.metrics import summarize
    from hftf_sim import EnvConfig, MarketData

    cfg = EnvConfig.load()
    market = MarketData.load(cfg.features_path)
    ep = sample_episodes(market, cfg, split, episodes, seed)
    rows = [run_policy(p, market, cfg, ep, randomise_impact=randomise_impact) for p in default_baselines()]
    table_df = summarize(pd.concat(rows, ignore_index=True))
    table = Table(title=f"Baselines on {split}: {episodes:,} episodes, seed {seed}")
    heads = ("n", "IS", "IS med", "IS sd", "CVaR95", "fees", "obj", "vs VWAP", "kids")
    table.add_column("policy")
    for h in heads:
        table.add_column(h, justify="right")
    for name, r in table_df.iterrows():
        table.add_row(name, *(f"{v:,.0f}" if c == "episodes" else f"{v:.1f}" for c, v in r.items()))
    console.print(table)
    console.print("bps of arrival notional; obj = IS + fees + risk; vs VWAP = slippage to market VWAP")


@train_app.command("ppo")
def train_ppo(
    run: str = typer.Option("ppo_mlp", help="Run name; outputs go to runs/<run>/."),
    config: str = typer.Option("configs/ppo.yaml", help="PPO config."),
    timesteps: int = typer.Option(0, help="Override total_timesteps (0 = config)."),
    seed: int = typer.Option(-1, help="Override the seed (-1 = config)."),
    policy: str = typer.Option("", help="mlp | fuzzy (default: the config's policy)."),
    evolve: bool = typer.Option(True, help="Fuzzy only: structure evolution (off = static ANFIS)."),
    antecedent_grad: str = typer.Option("", help="Fuzzy only: all | value_only | none (default: efnn.yaml)."),
) -> None:
    """Train PPO (MLP, static fuzzy or eFNN) on the train split; best.pt is picked by val objective."""
    from hftf_data.paths import REPO_ROOT
    from hftf_rl.config import EFNNConfig, PPOConfig
    from hftf_rl.ppo import PPOTrainer
    from hftf_sim import EnvConfig, MarketData

    cfg = PPOConfig.load(REPO_ROOT / config)
    if timesteps:
        cfg.total_timesteps = timesteps
    if seed >= 0:
        cfg.seed = seed
    if policy:
        cfg.policy = policy
    efnn = EFNNConfig.load()
    efnn.evolution.enabled = evolve
    if antecedent_grad:
        efnn.antecedent_grad = antecedent_grad
    env_cfg = EnvConfig.load()
    market = MarketData.load(env_cfg.features_path)
    trainer = PPOTrainer(cfg, env_cfg, market, REPO_ROOT / "runs" / run, efnn=efnn)
    best = trainer.train(log=console.print)
    console.print(
        f"[bold]best (val):[/bold] iteration {best.get('iteration')}  {json.dumps(best, indent=None)}"
    )


@eval_app.command("policy")
def eval_policy(
    checkpoint: str = typer.Argument(..., help="Path to a checkpoint, e.g. runs/ppo_mlp/best.pt."),
    split: str = typer.Option("val", help="train | val | test"),
    episodes: int = typer.Option(4000, help="Parent orders, shared by every policy (paired)."),
    seed: int = typer.Option(1, help="Episode seed (differs from the trainer's eval seed by default)."),
    randomise_impact: bool = typer.Option(False, help="Draw impact per episode from the training ranges."),
) -> None:
    """A trained policy vs the baselines on identical episodes, with the paired difference to VWAP."""
    import pandas as pd
    from rich.table import Table

    from hftf_eval.backtest import run_policy, sample_episodes
    from hftf_eval.baselines import default_baselines
    from hftf_eval.metrics import paired_diff, summarize
    from hftf_rl.ppo import load_policy
    from hftf_sim import MarketData

    policy, cfg, ckpt = load_policy(checkpoint)
    market = MarketData.load(cfg.features_path)
    ep = sample_episodes(market, cfg, split, episodes, seed)
    policies = [*default_baselines(), policy]
    df = pd.concat([run_policy(p, market, cfg, ep, randomise_impact=randomise_impact) for p in policies])
    table_df = summarize(df)
    table = Table(title=f"{checkpoint} (iteration {ckpt.get('iteration')}) on {split}: {episodes:,} episodes")
    table.add_column("policy")
    for h in ("n", "IS", "IS med", "IS sd", "CVaR95", "fees", "obj", "vs VWAP", "kids"):
        table.add_column(h, justify="right")
    for name, r in table_df.iterrows():
        table.add_row(name, *(f"{v:,.0f}" if c == "episodes" else f"{v:.2f}" for c, v in r.items()))
    console.print(table)
    for col in ("cost_bps", "is_bps", "fees_bps"):
        d = paired_diff(df, policy.name, "vwap", col)
        console.print(
            f"{policy.name} - vwap, {col}: {d['mean']:+.3f} bps (95% CI {d['lo']:+.3f}, {d['hi']:+.3f}); "
            f"better on {d['win']:.0%} of episodes"
        )


@eval_app.command("rules")
def eval_rules(
    checkpoint: str = typer.Argument(..., help="A fuzzy checkpoint, e.g. runs/ppo_efnn_s0/best.pt."),
    split: str = typer.Option("val", help="Episodes whose observations drive the firing statistics."),
    episodes: int = typer.Option(2000, help="Parent orders."),
    seed: int = typer.Option(1, help="Episode seed."),
) -> None:
    """Linguistic rule report: IF imbalance ∧ spread-var ∧ RV THEN pace, with each rule's firing share."""
    from rich.table import Table

    from hftf_eval.backtest import sample_episodes
    from hftf_rl.interpret import rule_report
    from hftf_rl.ppo import load_policy
    from hftf_sim import MarketData

    policy, cfg, ckpt = load_policy(checkpoint)
    if ckpt["policy"] != "fuzzy":
        raise typer.BadParameter("not a fuzzy checkpoint")
    market = MarketData.load(cfg.features_path)
    rows = rule_report(policy, market, cfg, sample_episodes(market, cfg, split, episodes, seed))
    table = Table(title=f"{checkpoint}: {len(rows)} rules, firing on {split} ({episodes:,} episodes)")
    for col in ("rule", "IF imbalance", "spread-var", "RV", "THEN pace", "share"):
        table.add_column(col, justify="right" if col in ("rule", "THEN pace", "share") else "left")
    for r in rows:
        table.add_row(
            str(r["rule"]),
            f"{r['imbalance_term']} ({r['imbalance_c']:+.2f})",
            f"{r['spread_var_term']} ({r['spread_var_c']:+.2f})",
            f"{r['rv_term']} ({r['rv_c']:+.2f})",
            f"{r['pace']:.2f}",
            f"{r['share']:.1%}",
        )
    console.print(table)
    console.print(
        "Centres in train-standardised units; terms from train terciles. Imbalance is signed by side."
    )


if __name__ == "__main__":
    app()
