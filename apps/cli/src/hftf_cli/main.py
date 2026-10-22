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


if __name__ == "__main__":
    app()
