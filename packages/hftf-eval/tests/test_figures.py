import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from hftf_eval.figures import write_figures  # noqa: E402


def _stats(policies, reference="vwap", **extra):
    n = len(policies)
    return pd.DataFrame(
        {
            "policy": policies,
            "reference": reference,
            "mean": np.linspace(-0.3, -0.1, n),
            "lo": -0.4,
            "hi": 0.0,
            "date_lo": -0.45,
            "date_hi": 0.05,
            "holm_p": 0.01,
            **extra,
        }
    )


def test_every_figure_renders_from_result_tables(tmp_path):
    reports, runs, out = tmp_path / "reports", tmp_path / "runs", tmp_path / "figures"
    reports.mkdir()
    (runs / "r0").mkdir(parents=True)
    _stats(["mlp", "anfis", "efnn"]).to_csv(reports / "final_test_vs_vwap.csv", index=False)
    _stats(["anfis", "efnn"], reference="mlp").to_csv(reports / "final_test_vs_group.csv", index=False)
    for name in ("m5_val_bar_vs_vwap", "m5_val_tick_vs_vwap"):
        _stats(["mlp", "anfis", "efnn"]).to_csv(reports / f"{name}.csv", index=False)
    _stats(["efnn", "vwap_passive100", "lip_efnn", "lip_mlp"]).to_csv(
        reports / "m5_lip_val_ticksim_vs_vwap.csv", index=False
    )
    _stats(["lip_efnn", "lip_mlp"], reference="vwap_passive100").to_csv(
        reports / "m5_lip_val_ticksim_vs_group.csv", index=False
    )
    for param, file in (
        ("gamma", "robust_gamma"),
        ("tau_res_min", "robust_tau_res_min"),
        ("eta", "robust_eta"),
    ):
        pd.concat([_stats(["mlp", "efnn"], **{param: v}) for v in (0.5, 1.0, 2.0)]).to_csv(
            reports / f"{file}.csv", index=False
        )
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        {
            "episode": np.tile(np.arange(120), 4),
            "policy": np.repeat(["vwap", "mlp", "anfis", "efnn"], 120),
            "security": np.tile(np.repeat(["A", "B", "C"], 40), 4),
            "date": np.tile(np.arange(120) % 6, 4),
            "cost_bps": 16 + rng.normal(0, 1, 480),
        }
    )
    rows.to_csv(reports / "final_test_rows.csv", index=False)
    pd.DataFrame(
        {
            "iteration": [1, 2, 3],
            "val_vs_vwap": [0.1, -0.1, -0.2],
            "val_vs_vwap_se": 0.05,
            "train_vs_vwap": [0.0, -0.1, -0.2],
            "val_cost": [16.2, 16.0, 15.9],
            "n_rules": [4, 5, 4],
        }
    ).to_csv(runs / "r0" / "progress.csv", index=False)
    (runs / "r0" / "events.csv").write_text(
        "iteration,kind,rules,n_rules,accepted,kl,detail\n2,add,4,5,1,0.001,{}\n3,prune,1,4,1,0.0,{}\n"
    )
    done = write_figures(out, reports, runs, "r0")
    assert len(done) == 7
    for _, files in done:
        assert all(f.exists() and f.stat().st_size > 1000 for f in files)
    assert "## Test split vs VWAP" in (out / "README.md").read_text(encoding="utf-8")
