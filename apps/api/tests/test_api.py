import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hftf_api.app import create_app
from hftf_api.service import DemoService, PolicyEntry
from hftf_rl.normalize import RunningMeanStd
from hftf_rl.policies import FuzzyActorCritic, MLPActorCritic
from hftf_rl.ppo import PacePolicy
from hftf_sim import EnvConfig, MarketData, TickData
from hftf_sim.testing import STATS, synthetic_frame, synthetic_ticks

DIM = 16


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    frame = synthetic_frame(n_sessions=3)
    days = sorted(frame["date"].unique())
    frame.loc[frame["date"] == days[1], "split"] = "val"
    frame.loc[frame["date"] == days[2], "split"] = "test"
    market = MarketData.from_frame(frame, stats=STATS)
    ticks = TickData.from_frame(market, synthetic_ticks(frame[frame["date"] == days[1]], depth=30.0))
    rms = RunningMeanStd((DIM,), passthrough=3)
    policies = {
        "efnn_s0": PolicyEntry("efnn_s0", "eFNN s0", "", PacePolicy(FuzzyActorCritic(DIM, [32], 4.0), rms)),
        "mlp": PolicyEntry("mlp", "MLP", "", PacePolicy(MLPActorCritic(DIM, [32], 4.0), rms)),
    }
    runs = tmp_path_factory.mktemp("runs")
    (runs / "r0").mkdir()
    pd.DataFrame({"iteration": [1, 2], "val_vs_vwap": [0.1, np.nan], "n_rules": [4, 5]}).to_csv(
        runs / "r0" / "progress.csv", index=False
    )
    (runs / "r0" / "events.csv").write_text(
        "iteration,kind,rules,n_rules,accepted,kl,detail\n2,add,4,5,1,0.001,{}\n"
    )
    table = runs / "table.csv"
    pd.DataFrame({"policy": ["mlp"], "mean": [-0.2], "holm_p": [np.nan]}).to_csv(table, index=False)
    service = DemoService(
        market=market, cfg=EnvConfig.load(), policies=policies, ticks=ticks,
        default_policies=("efnn_s0", "vwap"),
        runs_dir=runs, runs=("r0",), results=[{"id": "t", "title": "T", "file": str(table)}],
    )  # fmt: skip
    return TestClient(create_app(service))


def _ticket(client, split="val", sim="bar"):
    r = client.get("/api/order/random", params={"split": split, "sim": sim, "seed": 3})
    assert r.status_code == 200, r.text
    return r.json()


def test_meta_lists_sessions_policies_and_baselines(client):
    m = client.get("/api/meta").json()
    assert {s["split"] for s in m["sessions"]} == {"val", "test"}
    assert [s["ticks"] for s in m["sessions"]] == [True, False]
    kinds = {p["name"]: (p["kind"], p["explains"]) for p in m["policies"]}
    assert (
        kinds["efnn_s0"] == ("ppo", True)
        and kinds["mlp"] == ("ppo", False)
        and kinds["vwap"][0] == "baseline"
    )
    assert m["default_policies"] == ["efnn_s0", "vwap"] and m["runs"] == ["r0"]


@pytest.mark.parametrize("sim", ["bar", "tick"])
def test_order_returns_a_paired_trace_that_matches_the_trace_function(client, sim):
    from hftf_eval.trace import run_trace
    from hftf_sim.episodes import order_episode

    ticket = _ticket(client, sim=sim)
    r = client.post(
        "/api/order", json={**ticket, "policies": ["efnn_s0", "mlp", "vwap", "vwap_passive100"], "sim": sim}
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["order"]["sim"] == sim and out["order"]["start"] == ticket["start"]
    names = [p["name"] for p in out["policies"]]
    assert names == ["efnn_s0", "mlp", "vwap", "vwap_passive100"]
    for p in out["policies"]:
        assert sum(p["steps"]["q"]) == out["order"]["qty"] and p["steps"]["remaining"][-1] == 0
    assert out["policies"][0]["steps"]["phi"] is not None and out["policies"][1]["steps"]["phi"] is None
    service = client.app.state.service  # recompute with the library: the API must only serialise it
    session, _ = service.session_id(ticket["security"], ticket["date"])
    ep = order_episode(service.market, service.cfg.episode, session, ticket["start"], ticket["horizon"],
                       ticket["side"], ticket["participation"])  # fmt: skip
    ref = run_trace([service.policies[n].policy for n in names], service.market, service.cfg, ep,
                    service.ticks if sim == "tick" else None)  # fmt: skip
    for got, want in zip(out["policies"], ref["policies"], strict=True):
        np.testing.assert_allclose(got["steps"]["cum_cost_bps"], want["steps"]["cum_cost_bps"], rtol=1e-12)
        assert got["summary"]["cost_bps"] == pytest.approx(want["summary"]["cost_bps"], rel=1e-12)


def test_bad_tickets_and_unknown_names(client):
    ticket = _ticket(client)
    assert client.post("/api/order", json={**ticket, "start": "15:00", "horizon": 120}).status_code == 400
    assert client.post("/api/order", json={**ticket, "horizon": 45}).status_code == 400
    assert client.post("/api/order", json={**ticket, "date": "2030-01-01"}).status_code == 400
    assert client.post("/api/order", json={**ticket, "participation": 0.9}).status_code == 422  # schema bound
    assert client.post("/api/order", json={**ticket, "policies": ["nope"]}).status_code == 404
    test_ticket = _ticket(client, split="test")
    assert client.post("/api/order", json={**test_ticket, "sim": "tick"}).status_code == 400  # no ticks there
    assert client.get("/api/order/random", params={"split": "train"}).status_code == 400


def test_rules_runs_and_results(client):
    rules = client.get("/api/policies/efnn_s0/rules").json()
    assert len(rules["rules"]) == 4 and len(rules["terciles"]) == 3 and len(rules["input_labels"]) == 3
    assert abs(sum(r["share"] for r in rules["rules"]) - 1.0) < 1e-6
    assert all(len(r["centres"]) == len(r["widths"]) == len(r["terms"]) == 3 for r in rules["rules"])
    assert client.get("/api/policies/mlp/rules").status_code == 400  # an MLP has no rules
    run = client.get("/api/runs/r0").json()
    assert run["progress"]["val_vs_vwap"] == [0.1, None] and run["events"][0]["kind"] == "add"
    assert client.get("/api/runs/..%2F..%2Fsecrets").status_code == 404  # only configured run names
    assert client.get("/api/runs/r1").status_code == 404
    res = client.get("/api/results").json()
    assert res["tables"][0]["rows"] == [{"policy": "mlp", "mean": -0.2, "holm_p": None}]
