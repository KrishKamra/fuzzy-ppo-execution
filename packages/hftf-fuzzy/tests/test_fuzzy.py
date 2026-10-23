import numpy as np
import torch

from hftf_fuzzy import EvolutionConfig, Evolver, FuzzyTSK, bhattacharyya, kmeans, remap_optimizer_state
from hftf_fuzzy.interpret import linguistic, rule_table, terciles

torch.manual_seed(0)


def layer(n_rules: int = 3, hidden: int = 5) -> FuzzyTSK:
    lay = FuzzyTSK(d=2, n_rules=n_rules, heads={"actor": 2, "critic": 1}, hidden=hidden)
    with torch.no_grad():
        lay.centres.copy_(torch.randn(n_rules, 2))
        for n in lay.consequent_parameters():
            n.copy_(torch.randn_like(n))
    return lay


def out(lay: FuzzyTSK, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return lay.head("actor", h, lay.phi(x))


def test_phi_is_a_partition_of_unity_even_far_from_every_rule():
    lay = layer()
    x = torch.tensor([[0.0, 0.0], [1e3, -1e3]])
    phi = lay.phi(x)
    assert torch.isfinite(phi).all()
    torch.testing.assert_close(phi.sum(-1), torch.ones(2))


def test_add_is_function_preserving_when_consequents_agree():
    lay = layer()
    with torch.no_grad():
        for n in lay.consequent_parameters():
            n.copy_(n[:1].expand_as(n))  # every rule has the same consequent
    x, h = torch.randn(200, 2) * 3, torch.randn(200, 5)
    before = out(lay, x, h)
    Evolver(EvolutionConfig(tau_add=0.99, n_min_uncovered=1), lay.n_rules).step(lay, x)
    assert lay.n_rules == 4
    torch.testing.assert_close(out(lay, x, h), before)


def test_add_moves_outputs_less_than_a_random_new_rule():
    x, h = torch.randn(400, 2) * 3, torch.randn(400, 5)
    lay = layer()
    before = out(lay, x, h)
    Evolver(EvolutionConfig(tau_add=0.05, n_min_uncovered=8), lay.n_rules).step(lay, x)
    preserved = (out(lay, x, h) - before).abs().mean()
    with torch.no_grad():
        lay.actor_W[-1].copy_(torch.randn_like(lay.actor_W[-1]))
        lay.actor_b[-1].copy_(torch.randn_like(lay.actor_b[-1]))
    random = (out(lay, x, h) - before).abs().mean()
    assert preserved < 0.5 * random


def test_merge_of_identical_rules_preserves_the_output():
    lay = layer(n_rules=3)
    with torch.no_grad():
        for p in lay.parameters():
            p[2].copy_(p[1])
    x, h = torch.randn(100, 2), torch.randn(100, 5)
    before = out(lay, x, h)
    assert bhattacharyya(lay.centres[1], lay.widths()[1], lay.centres[2], lay.widths()[2]) > 0.999
    ev = Evolver(EvolutionConfig(min_age_iters=0, tau_add=0.0), 3)
    events = ev.step(lay, x)
    assert [e.kind for e in events][:1] == ["merge"] and lay.n_rules == 2
    torch.testing.assert_close(out(lay, x, h), before, atol=1e-5, rtol=1e-5)


def test_prune_drops_the_unused_rule_and_keeps_adam_state_for_survivors():
    lay = layer(n_rules=3)
    with torch.no_grad():
        lay.centres[0] = torch.tensor([50.0, 50.0])  # never fires
    x, h = torch.randn(256, 2), torch.randn(256, 5)
    opt = torch.optim.Adam(lay.parameters(), lr=1e-3)
    out(lay, x, h).sum().backward()
    opt.step()
    survivor_moment = opt.state[lay.actor_W]["exp_avg"][1:].clone()
    ev = Evolver(EvolutionConfig(min_age_iters=0, tau_add=0.0, tau_merge=1.1, utility_ema=0.0), 3)
    events = ev.step(lay, x)
    assert [e.kind for e in events] == ["prune"] and events[0].rules == [0] and lay.n_rules == 2
    remap_optimizer_state(opt, lay)
    torch.testing.assert_close(opt.state[lay.actor_W]["exp_avg"], survivor_moment)
    out(lay, x, h).sum().backward()
    opt.step()  # shapes line up after the edit


def test_guards_cooldown_and_snapshot_restore():
    lay = layer(n_rules=2)
    cfg = EvolutionConfig(tau_add=0.99, n_min_uncovered=1, max_rules=3, cooldown_iters=2, tau_merge=1.1)
    ev = Evolver(cfg, 2)
    x = torch.randn(64, 2) * 10
    snap, state = lay.snapshot(), ev.state()
    assert [e.kind for e in ev.step(lay, x)] == ["add"]
    assert ev.step(lay, x) == [] and ev.step(lay, x) == []  # cooldown
    assert ev.step(lay, x) == [] and lay.n_rules == 3  # max_rules
    lay.restore(snap)
    ev.restore(state)
    assert lay.n_rules == 2 and len(ev.utility) == 2


def test_kmeans_finds_separated_clusters():
    g = torch.Generator().manual_seed(1)
    true = torch.tensor([[-5.0, 0.0], [5.0, 0.0], [0.0, 6.0]])
    x = torch.cat([c + 0.3 * torch.randn(200, 2, generator=g) for c in true])
    centres, std = kmeans(x, 3)
    d = torch.cdist(true, centres).min(1).values
    assert (d < 0.2).all() and (std < 0.5).all()


def test_rule_table_terms():
    x = np.random.default_rng(0).normal(size=(3000, 2))
    cuts = terciles(x)
    assert linguistic(-2.0, cuts[:, 0]) == "LOW" and linguistic(2.0, cuts[:, 0]) == "HIGH"
    rows = rule_table(layer(), cuts, ["a", "b"])
    assert len(rows) == 3 and {"a_term", "b_c", "b_w"} <= rows[0].keys()


def test_terciles_ignore_nan_rows():
    x = np.random.default_rng(1).normal(size=(3000, 2))
    x[:100, 1] = np.nan
    assert np.isfinite(terciles(x)).all()
