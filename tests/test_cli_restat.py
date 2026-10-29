import numpy as np
import pandas as pd
from typer.testing import CliRunner

from hftf_cli.main import app


def test_restat_rewrites_tables_with_holm_and_date_columns(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    base = pd.DataFrame({
        "episode": np.arange(n), "security": rng.choice(["A", "B"], n),
        "date": rng.choice([f"2026-10-{d:02d}" for d in range(1, 13)], n),
        "regime_eval_prevday": rng.choice(["low", "high"], n),
    })  # fmt: skip
    rows = pd.concat(
        [base.assign(policy=p, cost_bps=16.0 + shift + rng.normal(0, 1, n))
         for p, shift in (("vwap", 0.0), ("mlp", -0.2), ("efnn", -0.3), ("efnn_s0", -0.3))],
        ignore_index=True,
    )  # fmt: skip
    rows.to_csv(tmp_path / "x_rows.csv", index=False)
    result = CliRunner().invoke(
        app, ["eval", "restat", str(tmp_path / "x_rows.csv"), "--groups", "mlp,efnn", "--vs", "mlp"]
    )
    assert result.exit_code == 0, result.output
    vs = pd.read_csv(tmp_path / "x_vs_vwap.csv").set_index("policy")
    assert {"holm_p", "date_lo", "date_hi", "date_holm_p"} <= set(vs.columns)
    assert np.isnan(vs.loc["efnn_s0", "holm_p"]) and not np.isnan(vs.loc["efnn", "holm_p"])  # family = groups
    assert (tmp_path / "x_vs_group.csv").exists() and (tmp_path / "x_regime.csv").exists()
