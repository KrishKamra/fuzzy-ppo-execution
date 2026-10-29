# M6 plan: demo backend and frontend

Status: **planned** (2026-10-29). This is the plan for the course demo: a local web app that shows the
system working on a real parent order, from the order ticket through the eFNN's regime reasoning to the fills
and the cost against VWAP. Design decisions made while building go in `ARCHITECTURE.md` (decision log), as
for M0-M5.

## 1. Goal and audience

The project is for an Intelligent Systems course, so the demo has to make **the intelligent part visible**,
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
| `GET /meta` | securities, sessions per split (with a tick flag), horizons, available policy groups (from `configs/demo.yaml`) |
| `POST /order` | body: security, date, start time, horizon, side, participation (or "random"), policies, `sim: bar\|tick`. Returns the §3.1 trace for every policy + VWAP/TWAP/AC and a summary row each |
| `GET /policies/{name}/rules` | rule table (terms, centres, widths, pace, passive, share on val) and the MF parameters for plotting |
| `GET /runs/{name}` | training curves (`progress.csv`: val vs VWAP, KL, rule count) and structure events (`events.csv`) |
| `GET /results` | the M4/M5 tables from `runs/reports/*.csv` (ensembles, CIs, regime breakdown, robustness sweep) |

Pydantic response models give the frontend a typed contract (exported as OpenAPI, from which TS types are
generated with `openapi-typescript`). Market data and checkpoints load once at start-up. Tests use FastAPI's
`TestClient` on the synthetic market (`hftf_sim.testing`), so CI needs no licensed data.

### 3.3 Frontend: `web/` (React 18 + Vite + TypeScript; ECharts for charts)
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
5. **Results:** the headline tables with CIs (M4 val/test, LOSO, M5 TickSim, limit orders), the impact
   robustness sweep, and the regime breakdown.

## 4. Milestones

| | Deliverable | Exit criterion |
|---|---|---|
| **M6a** | `hftf_eval.trace` + tests; `configs/demo.yaml` | trace totals equal `summary()` for every policy type (MLP, ANFIS, eFNN, limit), BarSim and TickSim |
| **M6b** | `apps/api` with the five endpoints + `hftf serve`; OpenAPI types | TestClient tests on synthetic data; `/order` < 300 ms for 5 policies on CPU |
| **M6c** | `web/` order ticket + execution replay | one real val order replays end to end for eFNN vs VWAP |
| **M6d** | fuzzy brain panel | MFs, firing and the active rule animate in sync; rule text matches `hftf eval rules` |
| **M6e** | learning + results views; demo script | a 3-minute scripted demo runs from a clean checkout plus local `data/` and `runs/` |
| **M6f** | report figures exported from the UI; README "run the demo" section | figures for the report regenerate from the UI |

Suggested split for three people: one owns M6a-M6b (Python), one owns M6c and M6e (replay and results),
one owns M6d (fuzzy panel, the course-specific part). The API contract (§3.2) is fixed first so the
three can work in parallel against mocked JSON.

## 5. Decisions to confirm

1. **Chart library:** ECharts (rich interaction, handles 360-point series and synchronised panels) is
   recommended over Recharts (simpler, weaker for linked brushing) and Plotly (heavy bundle).
2. **Test sessions in the UI:** off by default and labelled. Recommended, since the test split is already used.
3. **Streaming:** none. The trace is computed at once and animated client-side. A WebSocket only becomes
   worth it if live training is ever shown, which is out of scope.
4. **Packaging for the demo day:** two processes (`uv run hftf serve` and `npm run dev`), or the API serving
   the built `web/dist`. The latter is recommended for the final demo (one command, one port).
