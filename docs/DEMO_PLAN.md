# M6 plan: demo backend and frontend

Status: **built** (M6a-M6f, 2026-10-30); the frontend design is in [UI_DESIGN.md](UI_DESIGN.md) and the walkthrough in [DEMO_SCRIPT.md](DEMO_SCRIPT.md). This is the plan for the demo: a local web app that shows the
system working on a real parent order, from the order ticket through the eFNN's regime reasoning to the fills
and the cost against VWAP. Design decisions made while building go in `ARCHITECTURE.md` (decision log), as
for M0-M5.

## 1. Goal and audience

The project is about an intelligent system, so the demo has to make **the intelligent part visible**,
not just the result:

1. **Fuzzy inference:** the three regime inputs moving over the session, the Gaussian membership functions
   with the current value marked, each rule's firing strength, and the linguistic rule currently in charge
   ("IF spread-var HIGH ∧ RV HIGH THEN pace 0.83, slow down").
2. **Evolution:** how the rule base grew, merged and pruned during training (events, rule count, KL at edits).
3. **Decisions:** the policy's pace and passive fraction at each minute, and the child orders that result.
4. **Outcome:** fills on the price chart, and the cost against VWAP, TWAP and Almgren-Chriss on the *same*
   order, plus the statistically tested results (M4/M5 tables).

What a viewer should be able to do in under a minute: pick (or randomise) an order, press play, and watch
the eFNN slow down when spreads and volatility rise, while VWAP keeps trading.

## 2. Constraints that shape the design

| Constraint | Consequence |
|---|---|
| Bloomberg data is licensed (`data/` never committed or published) | **Local only.** The API runs on localhost and serves derived, per-order views (one episode's minute bars and fills), never bulk exports. No public deployment. Screenshots for the report are fine. |
| `runs/` and `*.pt` are git-ignored | The demo reads checkpoints from local `runs/`. A `demo` manifest (`configs/demo.yaml`) names the checkpoint globs per policy. Teammates either train (the `hftf train ppo` commands in `docs/RESULTS.md`, ~8-11 min per run on CPU) or copy `runs/` privately. |
| `apps/*` is a uv workspace glob | The Python API is a new workspace member `apps/api` (`hftf-api`); the JS frontend lives in **`web/`** (outside the glob). |
| An episode runs in milliseconds; training takes ~10 min | The backend **computes a whole episode trace at once** and the frontend animates it. There is no live training in the UI; training runs are shown from their recorded `progress.csv` / `events.csv`. |
| CPU only, small models | No GPU or job queue needed. One process, synchronous endpoints, an LRU cache for loaded data and checkpoints. |
| Test split was used once (M4) | The UI offers **val** sessions by default. Test sessions are behind an explicit toggle, labelled as out-of-sample showcase only. |

## 3. Architecture

```
web/ (React + Vite + TS)  ──HTTP/JSON──▶  apps/api (FastAPI, hftf-api)  ──▶  hftf-eval.trace  ──▶  hftf-sim / hftf-rl
   order ticket                             /meta /order /rules /runs          per-step trace          BarSim, TickSim,
   replay + fuzzy panel                     /results                           (no I/O, tested)        checkpoints
   learning + results views
```

### 3.1 Core addition: `hftf_eval.trace` (reusable, unit-tested; no web code)
`run_trace(policies, market, cfg, episode, ticks=None) -> dict` runs every policy (and the baselines) on one
identical episode and records, per decision step:
- market: time, mid, bid, ask, minute volume, and the expected-volume schedule;
- agent input: the 3 regime inputs (raw and standardised); for fuzzy policies, the firing `φ_r` of every
  rule, the dominant rule and its linguistic text;
- action: pace `a`, passive fraction `f`, child `q`, passive posted/filled, fill price, remaining inventory;
- cost: cumulative IS, fees, risk and objective, and the same for VWAP.

Invariant tests: the trace's cumulative cost equals `env.summary()`; shares sum to `Q`; the policies see
identical episodes. This is the only new logic. The API only serialises it.

### 3.2 Backend: `apps/api` (FastAPI + uvicorn; `uv run hftf serve`)

| Endpoint | Returns |
|---|---|
| `GET /api/meta` | securities, sessions per split (with a tick flag), horizons, available policy groups (from `configs/demo.yaml`) |
| `POST /api/order` | body: security, date, start time, horizon, side, participation, policies, `sim: bar\|tick`. Returns the §3.1 trace for every policy and a summary row each. `GET /api/order/random` gives a random ticket |
| `GET /api/policies/{name}/rules` | rule table (terms, centres, widths, pace, passive, share on val) and the MF parameters for plotting |
| `GET /api/runs/{name}` | training curves (`progress.csv`: val vs VWAP, KL, rule count) and structure events (`events.csv`) |
| `GET /api/results` | the M4/M5 tables from `runs/reports/*.csv`, computed by `stats_table` / `regime_table`: ensembles vs VWAP with session CI, Holm p and date CI; architecture pairs; breakdowns by regime, security and horizon; robustness sweep; the replication run alongside ours |

Pydantic response models give the frontend a typed contract (exported as OpenAPI, from which TS types are
generated with `openapi-typescript`). Market data and checkpoints load once at start-up. Tests use FastAPI's
`TestClient` on the synthetic market (`hftf_sim.testing`), so CI needs no licensed data.

**As built (M6a-M6b):** `hftf_eval.trace.run_trace` (a policy may expose `explain(env)` for its rule firing,
so hftf-eval stays torch-free); `hftf_sim.episodes.order_episode` builds an order from a ticket; `apps/api`
(`hftf-api`: `schemas.py` contract, `service.py` logic, `app.py` routes) with `uv run hftf serve`, bound to
127.0.0.1. On the real data, `/api/order` answers in 0.28 s (BarSim) / 0.33 s (TickSim) for 8 policies,
`/api/policies/{name}/rules` in 0.2 s (then cached), the others in < 20 ms. `hftf eval restat` recomputes the
report tables (Holm, date clusters) from saved backtest rows without simulating.

### 3.3 Frontend: `web/` (React 19 + Vite + TypeScript; ECharts for charts). Design spec: [UI_DESIGN.md](UI_DESIGN.md)
1. **Order ticket:** stock, session date, start time, horizon (30/60/120), side, size (% of expected
   volume), policies to compare, BarSim/TickSim, limit orders on/off, and a "random order" button.
2. **Execution replay** (the main screen, with a play/pause scrubber over the minutes):
   - price panel: mid with the bid/ask band, fills as markers sized by `q` (aggressive vs passive coloured);
   - inventory panel: remaining shares for each policy against the VWAP schedule;
   - cost panel: running objective of each policy minus VWAP (bps).
3. **Fuzzy brain** (synchronised with the replay): the three inputs as sparklines; per input, the rules'
   Gaussian MFs with the current value as a vertical line; a bar chart of rule firing `φ_r`; and the active
   rule rendered as a sentence with its pace and passive fraction.
4. **Learning:** rule count over iterations with add/merge/prune markers, edit KL against the 0.02 guard,
   and val objective vs VWAP with the `best.pt` iteration marked.
5. **Results:** a forest plot per question (adapted from the replication's figures): ensembles vs VWAP
   with session CI and date CI side by side, with Holm p next to each; architecture pairs; per-security and
   per-horizon breakdowns; the impact robustness sweep. A toggle shows **our run vs the independent
   replication** on the same axes, so the audience sees the result reproduce. A BarSim/TickSim toggle shows
   the M5 fill-model check, and a limit-orders panel shows the learned policy next to the naive
   always-passive rule, which is the honest comparison.

## 4. Milestones

| | Deliverable | Exit criterion |
|---|---|---|
| **M6a** ✅ | `hftf_eval.trace` + tests; `configs/demo.yaml` | trace totals equal `summary()` for every policy type (MLP, ANFIS, eFNN, limit), BarSim and TickSim |
| **M6b** ✅ | `apps/api` with the five endpoints + `hftf serve`; OpenAPI types | TestClient tests on synthetic data; `/order` < 300 ms for 5 policies on CPU |
| **M6c** ✅ | `web/` scaffold: design tokens and themes, shell, routing, command palette, typed API client | Lighthouse ≥ 95 on the shell (measured: 100 / 100 / 100) |
| **M6d** ✅ | Execute screen: order ticket, price / inventory / cost-race charts, transport dock, leaderboard | one real val order replays end to end, all panels in sync |
| **M6e** ✅ | fuzzy brain rail + Brain screen (3D regime space, 2D fallback) | rule text matches `hftf eval rules`; the 3D chunk loads only on that screen |
| **M6f** ✅ | Evidence and Training screens, hero, presentation mode, demo script, README | a 3-minute scripted demo runs from a clean checkout plus local `data/` and `runs/` |

The frontend milestones follow [UI_DESIGN.md](UI_DESIGN.md) §5 (the single source for M6c-M6f).
Suggested split for three people: one owns M6a-M6b (Python), one owns M6d and M6f (replay and results),
one owns M6e (fuzzy panel and regime space, the core of the project). The API contract (§3.2) is fixed first so the
three can work in parallel against mocked JSON.

## 5. What the demo claims (and does not)

The UI should say exactly what the evidence supports (docs/RESULTS.md):

| Claim | Evidence | How the UI shows it |
|---|---|---|
| PPO with an expected-cost reward beats VWAP, TWAP and Almgren-Chriss out of sample | val and test, two independent runs, Holm and date-cluster checks, LOSO | results screen, both runs toggled |
| The edge survives replaying fills against the real order book | M5 TickSim, paired with BarSim | BarSim/TickSim toggle on replay and results |
| The edge is transient-impact management | zero-impact and robustness sweeps | robustness chart; replay shows slower pacing when displacement builds |
| The fuzzy layer is interpretable and its rules follow execution theory | rule reports for every seed; calm → pace ≥ 1, stressed → pace < 1 | the fuzzy brain panel |
| eFNN is better than MLP | **not established** (Holm p ≥ 0.32 in both runs) | stated plainly on the architecture plot |
| Learned limit orders help | **yes**: started at the naive always-passive rule, the agent beats it by 0.43-0.46 bps in both simulators and has a lighter tail (CVaR 69 vs 75); absolute passive gains are an upper bound (counterfactual fills) | limit-orders panel: naive rule vs learned policy on the same order, passive fills marked on the price chart |
| eFNN beats MLP with limit orders | **no** (−0.03, n.s.) | stated on the panel |

The story is therefore *interpretable* intelligent execution with honest statistics, not "fuzzy beats
neural nets". It is also the more defensible story.

## 6. Decisions to confirm

1. **Chart library:** ECharts (rich interaction, handles 360-point series and synchronised panels) is
   recommended over Recharts (simpler, weaker for linked brushing) and Plotly (heavy bundle).
2. **Test sessions in the UI:** off by default and labelled. Recommended, since the test split is already used.
3. **Streaming:** none. The trace is computed at once and animated client-side. A WebSocket only becomes
   worth it if live training is ever shown, which is out of scope.
4. **Packaging for the demo day:** two processes (`uv run hftf serve` and `npm run dev`), or the API serving
   the built `web/dist`. The latter is recommended for the final demo (one command, one port).
