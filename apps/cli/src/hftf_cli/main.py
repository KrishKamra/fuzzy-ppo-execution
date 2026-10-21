"""`hftf` command-line entry point."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from hftf_data.paths import DataPaths

app = typer.Typer(no_args_is_help=True, help="Regime-aware execution (Fuzzy PPO) toolkit.")
data_app = typer.Typer(no_args_is_help=True, help="Data validation and feature building.")
app.add_typer(data_app, name="data")
console = Console()


@data_app.command("audit")
def audit(
    tier: str = typer.Option(
        "research", help="Readiness tier from configs/data_readiness.yaml: pilot | research"
    ),
    strict: bool = typer.Option(False, help="Exit with code 1 if any readiness check fails."),
) -> None:
    """Validate the raw Bloomberg pull, check readiness, and write docs/data/data_audit.json."""
    import pandas as pd
    from rich.table import Table

    from hftf_data.readiness import evaluate, load_thresholds, summary
    from hftf_data.validate import run_audit

    paths = DataPaths()
    report = run_audit(paths)
    feat_path = paths.processed / "features_1m_v2.parquet"
    features = pd.read_parquet(feat_path) if feat_path.exists() else None
    checks = evaluate(report, load_thresholds(tier), features)
    report["readiness"] = {"tier": tier, **summary(checks)}
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

    table = Table(title=f"Data readiness - tier: {tier}")
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
    """Build the causal 1-minute feature table v2 into data/processed/features_1m_v2.parquet."""
    from hftf_data.features import build_feature_table

    paths = DataPaths()
    df = build_feature_table(paths=paths)
    paths.processed.mkdir(parents=True, exist_ok=True)
    out = paths.processed / "features_1m_v2.parquet"
    df.to_parquet(out, index=False)
    console.print(f"[bold]{out}[/bold]: {len(df):,} rows x {df.shape[1]} cols")
    console.print(df.groupby("security")["l1_imb_source"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    app()
