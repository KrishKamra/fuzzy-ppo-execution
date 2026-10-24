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
    price_drift: str = typer.Option("", help="Reward: realised | expected (default: env.yaml)."),
    gate: str = typer.Option("", help="Fuzzy only: soft | hard (k-means gate ablation; no evolution)."),
    hold_out: str = typer.Option("", help="Leave this security out of training (and of the scaler)."),
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
    if gate:
        efnn.gate = gate
    env_cfg = EnvConfig.load()
    if price_drift:
        env_cfg.reward.price_drift = price_drift
    keep = None
    if hold_out:
        names = MarketData.load(env_cfg.features_path).sessions["security"].unique().tolist()
        if hold_out not in names:
            raise typer.BadParameter(f"{hold_out!r} is not one of {names}")
        keep = [n for n in names if n != hold_out]
    market = MarketData.load(env_cfg.features_path, securities=keep)
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
    from hftf_eval.report import LABEL, eval_labels
    from hftf_eval.stats import adjusted_mutual_info
    from hftf_rl.interpret import dominant_rules, rule_report
    from hftf_rl.ppo import load_policy
    from hftf_sim import MarketData

    policy, cfg, ckpt = load_policy(checkpoint)
    if ckpt["policy"] != "fuzzy":
        raise typer.BadParameter("not a fuzzy checkpoint")
    market = MarketData.load(cfg.features_path)
    ep = sample_episodes(market, cfg, split, episodes, seed)
    rows = rule_report(policy, market, cfg, ep)
    rule, session = dominant_rules(policy, market, cfg, ep)
    label = market.sessions.merge(eval_labels(cfg.features_path), on=["security", "date"], how="left")[LABEL]
    label = label.to_numpy()[session]
    known = label == label  # drop sessions without a previous-day label (NaN)
    ami = adjusted_mutual_info(rule[known], label[known])
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
    console.print(f"AMI(dominant rule, {LABEL}) = {ami:.3f} over {known.sum():,} decisions")


if __name__ == "__main__":
    app()


def _load_groups(groups: list[str], members: bool):
    """`NAME=GLOB` arguments -> policies: each group's seed ensemble (named NAME) and, optionally, every
    member alone (named after its run directory)."""
    import glob
    from pathlib import Path

    from hftf_data.paths import REPO_ROOT
    from hftf_rl.ppo import EnsemblePolicy, load_policy

    policies, cfg, stats = [], None, None
    for g in groups:
        name, _, pattern = g.partition("=")
        paths = sorted(glob.glob(str(REPO_ROOT / pattern)))
        if not name or not paths:
            raise typer.BadParameter(f"{g!r}: expected NAME=GLOB matching at least one checkpoint")
        loaded = [load_policy(p, name=Path(p).parent.name) for p in paths]
        cfg = cfg or loaded[0][1]
        for _, _, ckpt in loaded:  # every policy must see observations scaled as in its training
            s = ckpt.get("market_stats")
            if stats is not None and s is not None and s != stats:
                raise typer.BadParameter("checkpoints were trained with different observation scalers")
            stats = stats or s
        policies.append(EnsemblePolicy([p for p, _, _ in loaded], name))
        if members:
            policies += [p for p, _, _ in loaded]
    return policies, cfg, stats


def _print_stats(table, title: str) -> None:
    from rich.table import Table

    ref = table["reference"].iloc[0]
    t = Table(title=f"{title}: policy - {ref} (bps), 95% CI over {table['sessions'].iloc[0]} sessions")
    for col in ("policy", "diff", "CI", "Wilcoxon p", "win"):
        t.add_column(col, justify="left" if col == "policy" else "right", no_wrap=True)
    for _, r in table.iterrows():
        t.add_row(
            r["policy"], f"{r['mean']:+.3f}", f"({r['lo']:+.3f}, {r['hi']:+.3f})",
            f"{r['wilcoxon_p']:.3f}", f"{r['win']:.0%}",
        )  # fmt: skip
    console.print(t)


@eval_app.command("compare")
def eval_compare(
    groups: list[str] = typer.Argument(
        ..., help="NAME=GLOB of checkpoints, e.g. mlp=runs/m4_mlp_s*/best.pt."
    ),
    split: str = typer.Option("val", help="train | val | test"),
    episodes: int = typer.Option(4000, help="Parent orders, shared by every policy (paired)."),
    seed: int = typer.Option(1, help="Episode seed (not the trainer's selection seed)."),
    members: bool = typer.Option(True, help="Also evaluate every seed alone, not only the seed ensemble."),
    vs: str = typer.Option("", help="Also compare every group's ensemble against this group's ensemble."),
    by_regime: bool = typer.Option(False, help="Break the ensembles down by the eval-only regime label."),
    out: str = typer.Option("", help="Write the backtest rows and tables to runs/reports/<out>_*.csv."),
    security: str = typer.Option("", help="Only this security (e.g. the held-out one of a LOSO run)."),
) -> None:
    """Seed ensembles (and seeds) vs VWAP on fresh paired episodes, with session-cluster bootstrap CIs."""
    from hftf_data.paths import REPO_ROOT
    from hftf_eval.report import backtest, eval_labels, regime_table, stats_table
    from hftf_sim import MarketData

    policies, cfg, stats = _load_groups(groups, members)
    market = MarketData.load(cfg.features_path, securities=[security] if security else None, stats=stats)
    df = backtest(policies, market, cfg, split, episodes, seed, eval_labels(cfg.features_path))
    names = [p.name for p in policies]
    ensembles = [g.partition("=")[0] for g in groups]
    tables = {"vs_vwap": stats_table(df, names)}
    if vs:
        tables["vs_group"] = stats_table(df, [e for e in ensembles if e != vs], reference=vs)
    for table in tables.values():
        _print_stats(table, f"{split}, {episodes:,} episodes")
    if by_regime:
        tables["regime"] = regime_table(df, ensembles)
        console.print(
            tables["regime"][["policy", "regime_eval_prevday", "episodes", "mean", "lo", "hi"]].round(3)
        )
    if out:
        base = REPO_ROOT / "runs" / "reports"
        base.mkdir(parents=True, exist_ok=True)
        df.to_csv(base / f"{out}_rows.csv", index=False)
        for key, table in tables.items():
            table.to_csv(base / f"{out}_{key}.csv", index=False)
        console.print(f"wrote {base / out}_*.csv")


@eval_app.command("robustness")
def eval_robustness(
    groups: list[str] = typer.Argument(..., help="NAME=GLOB of checkpoints (evaluated as seed ensembles)."),
    param: str = typer.Option("eta", help="Impact parameter to sweep: eta | gamma | beta | tau_res_min."),
    values: str = typer.Option(
        "0.25,0.5,1,2,4", help="Comma-separated values (training range is in env.yaml)."
    ),
    split: str = typer.Option("val", help="train | val | test"),
    episodes: int = typer.Option(4000, help="Parent orders per setting."),
    seed: int = typer.Option(1, help="Episode seed."),
    out: str = typer.Option("", help="Write runs/reports/<out>.csv."),
    vs: str = typer.Option("", help="Also compare every group against this group."),
) -> None:
    """Ensembles vs VWAP as one impact parameter moves inside and outside the training range."""
    from hftf_data.paths import REPO_ROOT
    from hftf_eval.report import impact_sweep
    from hftf_sim import MarketData

    policies, cfg, stats = _load_groups(groups, members=False)
    market = MarketData.load(cfg.features_path, stats=stats)
    grid = [float(v) for v in values.split(",")]
    table = impact_sweep(policies, market, cfg, split, episodes, seed, param, grid, vs=vs or None)
    console.print(f"training range of {param}: {cfg.impact.randomise.get(param, 'fixed')}")
    cols = [param, "policy", "reference", "vwap_cost", "mean", "lo", "hi", "wilcoxon_p"]
    console.print(table[cols].round(3).to_string(index=False))
    if out:
        (REPO_ROOT / "runs" / "reports").mkdir(parents=True, exist_ok=True)
        table.to_csv(REPO_ROOT / "runs" / "reports" / f"{out}.csv", index=False)
