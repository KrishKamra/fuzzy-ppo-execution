import numpy as np
import pandas as pd
import pytest

from hftf_eval.backtest import run_policy
from hftf_eval.baselines import POV, TWAP, VWAP, AlmgrenChriss, Immediate, ac_expected_cost, ac_trajectory
from hftf_eval.metrics import cvar, summarize
from hftf_sim import BatchExecutionEnv, EnvConfig, FeeModel
from hftf_sim.testing import fixed_episodes as episodes
from hftf_sim.testing import linear_cfg
from hftf_sim.testing import synthetic_market as market


@pytest.mark.parametrize(("gamma", "lam"), [(0.0, 0.0), (0.3, 0.0), (0.3, 1e-4)])
def test_twap_matches_analytic_almgren_chriss_cost(gamma, lam):
    cfg, mk, H, Q = linear_cfg(gamma, lam), market(flat=True), 30, 3000.0
    env = BatchExecutionEnv(mk, cfg, fees=FeeModel.zero(), n_envs=2, autoreset=False, randomise_impact=False)
    env.reset(episodes(2, horizon=H, qty=Q))
    while not env.done.all():
        env.step_shares(TWAP()(env))
    sigma, V = 0.003 / np.sqrt(30), 1000.0
    expected = ac_expected_cost(
        1 - np.arange(H + 1) / H,
        half_spread=1.0,  # 2 bps spread
        eta_lin=1e4 * 0.5 * sigma * Q / V,
        perm_lin=1e4 * gamma * sigma * Q / V,
        lam_step=lam * (sigma * 1e4) ** 2,
    )
    np.testing.assert_allclose(env.summary()["cost_bps"], expected, rtol=1e-6)


def test_almgren_chriss_beats_twap_on_its_objective():
    np.testing.assert_allclose(ac_trajectory(np.array([0.0]), np.array([10]))[0], 1 - np.arange(11) / 10)
    cfg, mk = linear_cfg(lam=1e-2), market(flat=True)
    ep = episodes(4, horizon=60, qty=20000.0)
    cost = {
        p.name: run_policy(p, mk, cfg, ep, fees=FeeModel.zero())["cost_bps"].mean()
        for p in (TWAP(), AlmgrenChriss())
    }
    assert cost["almgren_chriss"] < cost["twap"]


def test_every_baseline_completes_every_episode():
    cfg, mk = EnvConfig.load(), market()
    ep = episodes(8, horizon=60, qty=5000.0)
    rows = [run_policy(p, mk, cfg, ep) for p in (Immediate(), TWAP(), VWAP(), POV(0.1), AlmgrenChriss())]
    for df in rows:
        assert np.array_equal(df["executed"].to_numpy(), ep.qty)
    assert rows[0]["n_children"].eq(1).all()
    table = summarize(pd.concat(rows))
    assert list(table.index) == ["immediate", "twap", "vwap", "pov10", "almgren_chriss"]


def test_cvar_is_the_tail_mean():
    assert cvar(np.arange(100.0)) == pytest.approx(97.0)


def test_passive_vwap_posts_its_fraction_and_completes():
    from hftf_eval.baselines import PassiveVWAP

    mk, cfg = market(), EnvConfig.load()
    rows = {f: run_policy(PassiveVWAP(f), mk, cfg, episodes(8)) for f in (0.0, 1.0)}
    np.testing.assert_array_equal(rows[0.0]["passive_frac"], 0.0)
    assert (rows[1.0]["passive_frac"] > 0).all()
    assert (rows[1.0]["executed"] == 3000.0).all() and (rows[0.0]["executed"] == 3000.0).all()
    pd.testing.assert_series_equal(  # f = 0 is plain VWAP
        rows[0.0]["cost_bps"], run_policy(VWAP(), mk, cfg, episodes(8))["cost_bps"], check_names=False
    )
