import numpy as np
import pandas as pd
import pytest

from hftf_eval.stats import adjusted_mutual_info, cluster_bootstrap_ci, paired_stats


def _frame(d: np.ndarray, sessions: np.ndarray) -> pd.DataFrame:
    base = {"episode": np.arange(len(d)), "security": "X", "date": sessions}
    return pd.concat(
        [pd.DataFrame({**base, "policy": "ref", "cost_bps": 10.0}),
         pd.DataFrame({**base, "policy": "p", "cost_bps": 10.0 + d})],
        ignore_index=True,
    )  # fmt: skip


def test_cluster_ci_is_wider_than_naive_when_sessions_share_a_shock():
    rng = np.random.default_rng(0)
    sessions = np.repeat(np.arange(40), 50)
    d = rng.normal(0, 1, 40)[sessions] + rng.normal(0, 0.2, len(sessions))  # mostly session-level noise
    s = paired_stats(_frame(d, sessions), "p", "ref")
    assert s["sessions"] == 40
    assert (s["hi"] - s["lo"]) > 3 * (s["naive_hi"] - s["naive_lo"])
    assert s["lo"] < s["mean"] < s["hi"]


def test_cluster_ci_covers_and_wilcoxon_detects_a_shift():
    rng = np.random.default_rng(1)
    sessions = np.repeat(np.arange(60), 20)
    d = -0.5 + rng.normal(0, 0.3, 60)[sessions] + rng.normal(0, 1, len(sessions))
    s = paired_stats(_frame(d, sessions), "p", "ref")
    assert s["hi"] < 0 and s["wilcoxon_p"] < 1e-3 and s["win"] > 0.5
    lo, hi = cluster_bootstrap_ci(np.zeros(10), np.arange(10))
    assert lo == hi == 0.0


def test_adjusted_mutual_info():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 3, 3000)
    assert adjusted_mutual_info(a, a) == pytest.approx(1.0)
    assert adjusted_mutual_info(a, (a + 1) % 3) == pytest.approx(1.0)  # label names do not matter
    assert abs(adjusted_mutual_info(a, rng.integers(0, 4, 3000))) < 0.01
    noisy = np.where(rng.random(3000) < 0.5, a, rng.integers(0, 3, 3000))
    assert 0.1 < adjusted_mutual_info(a, noisy) < 0.5


def test_impact_sweep_shares_episodes_and_leaves_the_config_alone():
    from hftf_eval.baselines import TWAP
    from hftf_eval.report import impact_sweep, with_impact
    from hftf_sim import EnvConfig
    from hftf_sim.testing import synthetic_market

    cfg, mk = EnvConfig.load(), synthetic_market(n_sessions=4)
    assert with_impact(cfg, eta=2.0).impact.eta == 2.0 and cfg.impact.eta != 2.0
    table = impact_sweep([TWAP()], mk, cfg, "train", 64, 0, "eta", [0.0, 1.0])
    assert list(table["eta"]) == [0.0, 1.0] and (table["sessions"] == 4).all()
    assert table["vwap_cost"].iloc[1] > table["vwap_cost"].iloc[0]  # more impact costs more
