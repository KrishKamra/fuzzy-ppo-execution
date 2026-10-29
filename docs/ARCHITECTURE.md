# Architecture Plan (v1)

This plan is definite so we can build against it. It is open to change: when something changes, edit
this file in the same PR and log the change in the [Decision log](#decision-log).

## 1. Problem statement

**Optimal execution of a parent order under regime shifts.** At a decision time `t0` we receive a parent order
(side, `Q` shares, horizon `H` minutes) for one NSE stock. We split it into child orders over the
continuous session to minimise **implementation shortfall (IS)** against the arrival mid-price,
including NSE transaction costs, subject to finishing by `t0 + H`.

The agent's state includes a **regime representation** from an **evolving fuzzy neural network (eFNN)**.
Its inputs are L1 order-book imbalance, bid-ask spread variance and realized volatility. The eFNN adds,
merges and prunes fuzzy rules as the regime distribution shifts. The membership functions are trained
end-to-end by the PPO actor-critic losses, which include the Bellman (TD/GAE) value loss.

> Scope: this is intraday **execution at 1-minute (later sub-minute) decision frequency**. We optimise
> slicing, not latency. "HFT" here means microstructure-aware execution, not colocated market-making.

## 2. System overview

```mermaid
flowchart LR
  subgraph data[hftf-data]
    RAW[(Bloomberg raw<br/>bars, ticks, daily)] --> CLEAN[clean + derive<br/>F1-F11 fixes] --> FEAT[(features_1m_v3<br/>final causal table)]
  end
  subgraph sim[hftf-sim]
    FEAT --> SAMP[EpisodeSampler] --> ENV[ExecutionEnv<br/>vectorised]
    IMP[ImpactModel] --> ENV
    FEE[NSE FeeModel] --> ENV
  end
  subgraph agent[hftf-rl + hftf-fuzzy]
    ENV -- obs --> SPLIT{split obs}
    SPLIT -- x_regime --> EFNN[Evolving fuzzy layer<br/>Gaussian MFs → φ]
    SPLIT -- z_private+market --> TRUNK[MLP trunk h]
    EFNN --> TSK[TSK heads<br/>Σφ_r·head_r h]
    TRUNK --> TSK
    TSK --> PI[π: Beta pace] & V[V critic]
    PI -- action --> ENV
    BUF[(rollout buffer)] --> PPO[PPO update<br/>clip + GAE + value]
    PPO -- ∇ incl. MF centres/widths --> EFNN
    BUF -- x samples --> EVO[structure evolution<br/>add / merge / prune] --> EFNN
  end
  subgraph eval[hftf-eval]
    BASE[TWAP · VWAP · POV · Almgren-Chriss] --> BT[backtest runner]
    PI --> BT --> MET[IS, VWAP slippage, CVaR,<br/>per-regime, rule analytics]
  end
```

## 3. Data layer (`hftf-data`), built

- Raw Bloomberg pull (read-only) → `loaders` → `ticks` (cleaning) / `derive` (missing quantities) →
  `features.build_feature_table` → `data/processed/features_1m_v3.parquet` (+ manifest), the final dataset.
- **Invariant:** a row stamped `time` (bar start, UTC) uses only information available at
  `decision_time = time + 1 min`. Rolling windows reset each session. The volume profile uses previous
  sessions only. Daily features are lagged one session.
- **Grid and gaps:** 09:15-15:15 IST (360 min) every day, because there is a closing call auction from
  2026-08-03. Incomplete sessions are dropped whole and never imputed. Within-session fills are flagged
  (`fill_trade`, `fill_quote`), as are warm-up estimates (`dq_warmup`). Each row has a `split` label (DATA_AUDIT F10-F11).
- See `docs/DATA_AUDIT.md` for the quirks and fixes, and `project_context.md §Data` for the column dictionary.

## 4. Execution environment (`hftf-sim`)

### 4.1 Episode
| Item | v1 choice | Notes |
|---|---|---|
| Unit | (stock, session, `t0`, `H`, side, `Q`) | sampled by `EpisodeSampler` |
| Horizon `H` | {30, 60, 120} min | `t0` uniform on [09:20, 15:15 − H]; avoid the first 5 min; the grid ends at 15:15 (closing auction) |
| Size `Q` | `ρ · Σ_{t0}^{t0+H} prof_avg_vol` with `ρ ~ LogUniform[1%, 20%]` | expected participation |
| Side | buy / sell, 50/50 | the env is side-symmetric through signed prices |
| Decision step Δ | 1 min (BarSim) | TickSim later: 5-15 s on tick days |
| Termination | `t = t0 + H` with the residual forced as a market order | the forced cost is in the final reward |

### 4.2 Observation (all standardised by train-split statistics)
- **Regime input `x` → eFNN (d = 3):** `imb_bar_proxy` (5-min EMA), `log(spread_var_30m + ε)`, `log(rv_30m + ε)`.
  Use the **bar proxy on every split**, not the mixed `l1_imb` column. Train has only proxy days and test
  has only tick days, and the two sources have different scales (std 0.42 vs 0.32). Mixing them would hand
  the eFNN a source shift that looks like a regime shift. Tick L1 imbalance is reserved for analysis and TickSim.
  Optional extension (config): `rel_volume`, `ofi_norm_5m` (tick days only).
- **Private `z_p`:** remaining inventory fraction, remaining time fraction, IS so far (bps), last pace
  action, schedule deviation (executed − VWAP-schedule fraction), and (added in M2) `log_participation` =
  log(Q / expected horizon volume) and minutes left / 120. Without the last two the policy cannot tell order
  size or horizon apart, and these set the impact vs per-child-fee trade-off.
- **Market `z_m`:** `spread_bps`, `dev_vwap_bps` (signed), `ret_5m` (signed by side), expected share of the
  remaining volume in the next 5 min (from the profile), `ctx_nifty_ret_30m` (signed), `ctx_vix`.
- **Side symmetry (built):** the policy never sees `side`, so every directional column is multiplied by it:
  `imb_bar_proxy_ema5` (so the eFNN sees "imbalance with/against us"), `dev_vwap_bps`, `ret_5m`,
  `ctx_nifty_ret_30m`. Signed columns are scaled by the train std but **not centred**, so the sign flip is an
  exact mirror; the others are z-scored. Stats come from train rows from the 09:20 decision (as in the
  manifest), and z-scores are clipped to ±5. Layout: `hftf_sim.obs_names()` = x (3) · z_p (5) · z_m (6).

### 4.3 Action: pace multiplier
`a_t ∈ [0, a_max]` (`a_max = 4`), sampled from a **Beta** distribution scaled to that range.
Child size `q_t = clip(a_t · X_t · w_t / Σ_{s≥t} w_s, 0, min(X_t, π_max · V̂_t))`, where `X_t` is the
remaining inventory, `w` is the expected volume profile, and `π_max = 0.3` caps participation.
`a = 1` reproduces adaptive VWAP, `a = 0` waits, and `a > 1` front-loads. This parameterisation makes the
naive policy a strong baseline and keeps the action meaning stable across `Q` and `H`.
*v2:* add a discrete head for order type {market, limit@touch} with a fill-probability model on tick days.

### 4.4 Fill model (BarSim)
Children execute at minute `t+1` as aggressive orders:

```
p_exec = m̃_{t+1} + side · ( s_{t+1}/2  +  η · σ_{t+1} · (q_t / V_{t+1})^β )      temporary
m̃_t   = m_t + side · Σ_{u<t} γ · σ_u · (q_u / V_u)^β · exp(−(t−u)/τ_res)          transient (OW-style)
```
- `m`, `s`, `V` are the historical mid, spread and realised minute volume. `σ` is the 1-min vol from `rv_30m`
  (`rv_30m/√30`). As built, impact terms are fractions of the mid: `p = m·(1 + side·(D + s/2 + η·σ·(q/V)^β))`
  with `D` the decayed transient displacement (`hftf_sim/impact.py`).
- `V_eff = max(V, 1% of expected volume, 1 share)` guards (near-)empty minutes (1 zero-volume usable minute).
- Children are whole shares. The last step force-executes the residual and ignores the participation cap.
- *Deferred to TickSim (M5):* children with `q ≤` L1 depth paying half the spread only. BarSim always adds
  the temporary impact term.
- Defaults: `β = 0.5` (square-root law), `η = 0.5`, `γ = 0.3`, `τ_res = 5 min`. **Domain randomisation**
  draws `η ∈ [0.25, 1.0]`, `γ ∈ [0.1, 0.5]` and `τ_res ∈ [2, 15]` per episode, because our own impact cannot
  be identified from public data. Robustness to these parameters is a reported result, not a hidden assumption.
  Draws come from each episode's own seed, so paired policies face identical impact. Evaluation defaults to
  the fixed defaults (`hftf eval baselines --randomise-impact` to draw).
- Counterfactual caveat: historical prices are not affected by our past trades except through `m̃`.

### 4.5 Costs (`configs/fees_nse.yaml`, configurable, verify against current circulars)
STT (0.1% buy and sell, delivery), NSE transaction charge (0.00297%), SEBI fee (₹10/crore), stamp
duty (0.015% buy), GST 18% on (brokerage + exchange + SEBI), and brokerage (bps, plus an optional fixed
amount **per child order**).
Proportional fees barely depend on how the order is sliced. **The per-child fixed fee is what penalises
over-slicing**, so the slice count becomes a real trade-off.

### 4.6 Reward (bps of arrival notional, per step)
```
r_t = − side · (p_exec,t − m_0)/m_0 · (q_t/Q) · 1e4   − fees_t(bps)   − λ · (σ_t·1e4)² · (X_{t+1}/Q)² · Δ
```
**Per-step credit (`env.yaml reward.mode`, M2).** `arrival` is the formula above. `mark_to_market` splits the
same IS into each step's execution cost `q_t(p_t − m_t)` plus inventory P&L `X_t(m_t − m_{t−1})`, giving the
same episode return. `vs_vwap` (the training default) subtracts the same step of a **shadow adaptive VWAP** (a = 1,
own impact state) run on the same path. Its cost does not depend on our actions, so the optimal policy and
the policy gradient are unchanged. But the price-drift noise, an episode IS std of ~28 bps against a learnable
edge of ~0.2 bps, now scales with our deviation from VWAP rather than with the whole order. Reported costs
are always absolute (`summary()`).
**Expected price drift (`reward.price_drift`, M4; default `expected`).** The inventory term `X_t(m_t − m_{t−1})`
is replaced by its martingale expectation, 0, so the training return is the expected cost plus the risk
penalty, which is the Almgren-Chriss mean-variance objective. With `realised` (M2/M3), the whole train edge was
timing against the 180 train price paths: re-evaluated at zero impact, it was unchanged, and it vanished on val.
With `expected`, the edge is impact reduction and transfers to val (docs/RESULTS.md, M4). Evaluation always
charges the realised IS.
`m_0` is the mid of the row observed at `t0`. The risk term uses the inventory **left after** the child
(`X_{t+1}`), as in discrete Almgren-Chriss (`Var = σ²Σ x_k²`, `k ≥ 1`), so immediate execution carries no
timing risk.
Summed over an episode, this gives `−(IS + fees + λ·risk)`, the Almgren-Chriss objective. Defaults are
`λ = 1e-4` (swept), and running-std reward scaling is applied in the trainer.

### 4.7 Implementation
A pure-numpy **batched env** (`BatchExecutionEnv`, N episodes stepped together) with a thin
`gymnasium.Env` wrapper for compatibility. There is no GPU, so environment throughput dominates
training time. Market arrays are held per session as `(n_sessions, 360)` arrays (`MarketData`) and
indexed per step by `(session, row0 + k)`, which is equivalent to pre-slicing and cheaper. `autoreset=True`
(training) re-samples finished envs; `autoreset=False` (evaluation) runs a fixed batch. Throughput: about
4,000 episodes per baseline in 0.2 s. Baselines (`hftf-eval`) act in shares through `step_shares` and are
not participation-capped, except VWAP, which is the pace action `a = 1`.

### 4.8 TickSim (M5, built: evaluation on tick sessions)
Ticks exist only on val (8) and test (12) sessions per stock, and **none on train**. TickSim is therefore a
higher-fidelity **evaluation** simulator: policies trained in BarSim are replayed against the real L1 state
on the same episodes. The agent, its observation (1-minute features) and its decision clock are unchanged;
only the fill changes (`hftf_sim.tick.TickExecutionEnv`, a subclass of the BarSim env).
- **Input:** `l1_1s_v1.parquet`, a 1-second L1 grid (DATA_TRANSFORMS T23). Bars and ticks share one clock:
  bar `m` = seconds `60m+1 … 60m+60` (DATA_AUDIT F12), so the arrival mid is grid row `60(row0+1)`.
- **Fill:** the child for bar `j` is sent at the bar boundary and fills against the touch at row `60j + L`
  (latency `L = 1 s`). With displayed depth `d` on the side we take, `x = (q − d)⁺`:
  `p = m·(1 + side·(D + s/2 + (x/q)·η·σ·(x/V)^β))`, so shares within the touch pay the half spread only
  and the excess walks the book with BarSim's square-root term (depth beyond L1 is unknown). Transient
  impact `D` is BarSim's OW state.
- **Sub-minute orders (`tick.sub_slices`, default 1):** n orders per decision at offsets `L + ⌊60i/n⌋` s, with
  `V/n` per order and kick `/n`, decayed continuously. Each order pays the per-order fee.
- **Exact reduction (test):** one order and no depth fills, on a grid built from the bars, reproduces
  BarSim's rewards and summaries exactly in every reward mode.
- **CLI:** `hftf eval baselines|policy|compare|robustness --sim tick` (tick sessions of the split);
  `--sim bar --tick-sessions` runs BarSim on the same episodes for paired sim-gap measurements.
- **Measured gap (VWAP, val tick sessions):** timing alone (tick mid at bar start vs bar-close mid)
  −0.004 bps (n.s.), depth −0.32 bps, latency 0-5 s nil, 4 orders per decision +2.8 bps (fees).

### 4.9 Limit orders (M5, built)
With `limit.enabled` (`hftf train ppo --limit`), the action is `(a, f)`: the pace and the **passive fraction**
`f ∈ [0, 1]` of the child, from a second, independent Beta head (MLP and fuzzy TSK heads alike). Each decision
posts `⌊f·q⌋` at the near touch (bid for a buy) and sends the rest aggressively. The passive order rests
until the end of the bar, and the unfilled part is cancelled and returns to the inventory (the schedule
re-plans it). The forced last step is all aggressive. **Dead zone** (`limit.dead_zone` = 0.1): a fraction ≤ 0.1 snaps to 0 and ≥ 0.9 to 1, so a child can be a single order of either kind (a Beta never reaches 0 or 1 exactly, and every mixed child pays the per-order fee twice). A passive fill pays the limit price shifted by our
transient displacement, with no spread, no temporary impact and no kick, plus fees. Each non-empty passive
or aggressive order pays the per-order fee. `hftf_sim.limit` holds both fill models:
- **BarSim (all days, used for training):** the decision's quote against the fill bar's trade range. If the
  range trades through our price, `min(posted, θ_through·V)` fills; if it trades at our price,
  `min(posted, θ_touch·V)`; if it never reaches our price, nothing fills.
- **TickSim (tick sessions, used for evaluation):** a queue model. The order joins the back of the displayed
  queue at row `60j + L`. Prints at our price (seller-initiated for a buy) consume the queue ahead, then us.
  A print through our price clears the queue, and we fill up to the volume traded at or through our price.
  **A fill never exceeds the volume that actually traded there.**
- **Calibration:** θ_touch = 0.0656 and θ_through = 0.2942 each minimise the squared fill error of the bar
  rule against the queue model on their own minutes, on the **val** tick sessions (`hftf eval
  calibrate-limit`; it refuses `test`). Over 68k synthetic orders (1-30% of expected minute volume, both
  sides), the bar rule fills 48.7% of shares and the queue model 47.1% (trade-through minutes 65.6 / 61.9%,
  touched 22.1 / 20.6%). The bar trades through the decision quote in 66% of minutes.
- **Naive reference:** `PassiveVWAP(f)` (`hftf_eval.baselines`) is adaptive VWAP with a fixed passive
  fraction. A learned passive fraction is compared with it, not only with VWAP.
- Adverse selection is not modelled separately; it comes from the replayed prices (passive buys fill when
  the price trades down). Our resting order does not move the historical book.

## 5. Evolving fuzzy layer (`hftf-fuzzy`)

### 5.1 Inference (first-order TSK, product t-norm)
For standardised `x ∈ ℝ^d` and rules `r = 1..R` with centres `c_r` and widths `σ_r = softplus(ρ_r)` (diagonal):
```
τ_r(x) = exp(−½ Σ_j ((x_j − c_rj)/σ_rj)²)      firing strength
φ_r(x) = τ_r / (Σ_k τ_k + ε)                    normalised, a soft regime assignment
```
Heads are fuzzy-gated mixtures of linear experts over the trunk embedding `h = MLP([z_p, z_m, x])`:
```
actor:  (α, β) = 1 + softplus( Σ_r φ_r · (A_r h + b_r) )
critic: V      =             Σ_r φ_r · (u_r·h + d_r)
```
**As built (M3, `hftf_fuzzy/layer.py`):**
- `φ` is a softmax of `log m_r + log τ_r`, so it never underflows far from every rule; such points go to the nearest rule.
- `m_r` is a per-rule **support mass**, a buffer that is not trained: 1 for a new rule, and the sum of the two
  masses for a merge, so merging two identical rules is exactly function-preserving.
- `x` is the env's train-standardised regime input: the running observation normaliser passes those 3 dims
  through, so the MFs see a fixed scale.
- All rules start with identical consequents, so the initial policy equals the MLP's noisy VWAP whatever the
  antecedents are.

### 5.2 Parameter learning: "Bellman updates shape the memberships"
`c_r` and `ρ_r` are `nn.Parameter`s. The PPO loss `L = L_clip − c_e·H + c_v·L_V` back-propagates into
them. `L_V` is the TD(λ)/GAE regression loss, so the critic's Bellman error moves the MF centres and
widths. The antecedents get their own optimizer param group with a lower learning rate (default 0.3× the
base LR). Config flag `antecedent_grad: {all, value_only, none}`. `value_only` ties MF shaping purely to
Bellman error and is an ablation.

### 5.3 Structure evolution (between PPO iterations)
Structure is edited **only after the update epochs and before the next rollout**. PPO's importance
ratio needs the stored `log π_old` to come from the exact network that collected the buffer.
1. **Add** (novelty): take buffer samples with `max_r τ_r(x) < τ_add` (default 0.15). If at least
   `n_min` (default 64) of them exist, add one rule at their mean with width = their std (floored).
   The new consequent is initialised **function-preserving**: it copies the firing-weighted average of the
   existing consequents at those samples, so the policy does not jump.
2. **Merge:** if two rules overlap (Bhattacharyya coefficient > `τ_merge`, default 0.8), replace them with
   a support-weighted merge.
3. **Prune:** utility `U_r` = EMA over iterations of `mean_x φ_r(x)`. Prune a rule if `U_r < τ_prune`
   (default 0.02) and its age ≥ `a_min` iterations.
4. Guards: `R ∈ [R_min=2, R_max=16]`, at most one add and one prune per iteration, and a cooldown.
   After each edit, rebuild the optimizer param groups (keep Adam state for surviving tensors) and log
   the event.
Initialisation: `R0 = 4` rules from k-means on the regime inputs of the training split's first rollout.
**As built:**
- Each evolution pass's policy change is measured as `KL(π_before ‖ π_after)` on 8,192 buffer observations,
  and a pass with KL > `edit_kl_max` (= `target_kl`, 0.02) is **undone** and logged as rejected. The exit
  criterion is therefore enforced, not just hoped for.
- Edits keep the same `Parameter` objects and remap Adam moments row by row (`remap_optimizer_state`), with new
  rows starting from zero.
- Events go to `runs/<run>/events.csv` (iteration, kind, rules, R, accepted, KL), and `n_rules` and `edit_kl` go to `progress.csv`.
- Initial widths are the k-means cluster std, floored at 0.1.

### 5.4 Interpretability
Each rule centre is mapped to linguistic terms per input (Low/Med/High from train terciles) and to its
mean pace action. `hftf eval rules` prints, for example, *"IF spread-var HIGH ∧ RV HIGH ∧ imbalance NEUTRAL
THEN pace 0.62 (slow down)"*. Rule activations are compared with the eval-only regime labels using
adjusted mutual information.

## 6. Learning (`hftf-rl`)

PPO uses the clipped surrogate (ε = 0.2), GAE (γ = 1.0 because episodes are finite and IS-additive,
λ = 0.95), 10 epochs × 8 minibatches, Adam 3e-4, entropy 1e-3, value-loss coefficient 0.5, gradient
norm clip 0.5, target-KL early stop at 0.02, advantage normalisation, running observation/reward
normalisation, and 64 parallel episodes × 120 steps per rollout. The trunk is 2×64 tanh.
**As tuned in M2:** 256 envs and lr 1e-3 (64 envs / 3e-4 did not learn within 3M steps), 6M steps, no LR
annealing (no gain over 3 seeds). The Beta head is initialised to mean pace 1 with α + β = 8, so the
untrained policy is a noisy adaptive VWAP. Every 3 iterations the deterministic policy (Beta mean) and VWAP run
on 2,000 fixed paired train and val episodes at default impact, and `best.pt` keeps the best val objective.
Report on **fresh** episodes (`hftf eval policy`, seed 1), never on the selection set. Policies are
`MLPActorCritic` (baseline) and `FuzzyActorCritic` (eFNN; also runs as static ANFIS with evolution off).
After each update, `EvolutionCallback` applies §5.3. Logging goes to TensorBoard and CSV, and checkpoints
include the fuzzy structure (R varies).

## 7. Evaluation (`hftf-eval`)

- **Splits (walk-forward by date, train statistics only):** train 2026-06-03 → 08-31; val 09-01 → 09-30;
  test 10-01 → 10-19, all of which are tick days. That gives 60 / 20 / 12 sessions per stock. 06-03 → 06-05
  (profile burn-in) and the partial 10-21 are `excluded`, and 09-10 is absent. A secondary generalisation
  check is leave-one-stock-out.
- **Baselines:** immediate (all at t0), TWAP, VWAP (profile), POV(10%), Almgren-Chriss (closed form,
  same σ/η/λ), PPO-MLP, PPO + static fuzzy (ANFIS, no evolution), and **PPO + eFNN (ours)**.
- **Ablations:** antecedent gradients off / value-only; evolution off; regime inputs swapped for raw
  features fed to the MLP; hard k-means gate instead of fuzzy.
- **Metrics:** IS mean/median/std (bps), CVaR95 of IS, VWAP slippage, fees, the risk-adjusted objective,
  completion rate, and participation profile. Results are broken down by regime (eval labels).
  Rule count over training, rule-regime AMI.
- **Statistics:** every policy runs on identical episode seeds (paired). We report paired bootstrap
  95% CIs and Wilcoxon signed-rank tests against VWAP and PPO-MLP. **As built (M4, `hftf_eval.stats`):** the
  bootstrap resamples whole (security, date) sessions, because episodes on one session share a price path; on
  val (60 sessions) this CI is about 2× the per-episode one. Wilcoxon runs on per-session mean differences.
  Policies are reported as **seed ensembles** (mean pace of the seeds' `best.pt`), with every seed listed too.
  **Added after the M4 replication:** Wilcoxon p-values are Holm-adjusted over the family of ensemble
  claims in a table, and every comparison also reports a date-cluster CI and Wilcoxon test (stocks share
  market moves on a date) and the unweighted session-mean t-interval, as sensitivity checks.
- **Leave-one-stock-out (as built):** `hftf train ppo --hold-out SEC` trains and fits the observation scaler on
  the other two names. The checkpoint stores the scaler, and `hftf eval compare --security SEC` evaluates on
  the unseen name.
- **Robustness:** sweep the impact parameters off the training range.

## 8. Repository layout (uv workspace monorepo)

```
hft_fuzzy/
├── pyproject.toml            # workspace root (virtual), dev tools, ruff/pytest/mypy config
├── packages/
│   ├── hftf-data/            # loaders, tick cleaning, derivations, features, audit       [built]
│   ├── hftf-sim/             # ExecutionEnv, BatchEnv, EpisodeSampler, ImpactModel, FeeModel
│   ├── hftf-fuzzy/           # MFs, EvolvingFuzzyLayer, evolution rules, TSK heads, rule report
│   ├── hftf-rl/              # PPO trainer, policies, buffer, EvolutionCallback, checkpoints
│   └── hftf-eval/            # baselines, backtest runner, metrics, stats tests, plots
├── apps/cli/                 # `hftf` CLI: data | train | eval | report
├── configs/                  # YAML configs (pydantic-validated): data, env, fees, efnn, ppo, experiments
├── data/                     # raw Bloomberg (is2/) + processed/ — git-ignored (licensed data)
├── docs/                     # ARCHITECTURE, DATA_AUDIT, prompts/, data/ (audit json)
├── notebooks/                # exploration only; no logic that others import
├── scripts/                  # one-off utilities
└── tests/                    # cross-package integration tests
```
Dependency direction: `data → sim → rl ← fuzzy`, and `eval` depends on `sim`/`data`/`rl`. `fuzzy` has no
dependency on `sim`, so it stays reusable as a standalone torch module.

## 9. Milestones

| | Deliverable | Exit criterion |
|---|---|---|
| **M0** ✅ | uv workspace, data audit, final features v3 | `hftf data audit --strict` passes (12/12); tests green |
| **M1** ✅ | `hftf-sim` + TWAP/VWAP/AC baselines | share conservation and causality tests; TWAP IS ≈ analytic AC cost under the linear-impact config |
| **M2** ✅ | PPO-MLP | beats VWAP on train; val gap reported |
| **M3** ✅ | static fuzzy → evolving fuzzy | rule add/merge/prune logged; no PPO instability at edits (KL spikes < target) |
| **M4** ✅ | experiments, ablations, robustness, report | paired stats tables, regime breakdowns, rule narratives |
| **M5** ✅ | TickSim (§4.8) and limit orders (§4.9) on the frozen data (full-universe data dropped: the pull is frozen) | M4 policies re-evaluated in TickSim with paired BarSim-TickSim gaps; limit-order head evaluated on tick sessions |
| **M6** | demo backend (`apps/api`) and frontend (`web/`): see `docs/DEMO_PLAN.md` | a real order replays end to end with the fuzzy reasoning visible |

## 10. Key risks

- **Small data → overfitting:** 3 stocks and 60 training sessions. Mitigations: many episodes per
  session (random `t0`, `H`, `Q`, side), domain randomisation, small networks, early stopping on val.
  The data is frozen, so state this limitation in the report and lean on the leave-one-stock-out check.
- **Impact-model dependence:** results are conditional on the simulator. We report robustness sweeps and
  never claim real-world savings without live data.
- **Non-stationary architecture in PPO:** edits happen only between iterations, are function-preserving,
  and are rate-limited.
- **CPU-only compute:** batched numpy env and small networks. Expect a few minutes per 1M env steps.

## Decision log
| Date | Decision | Why |
|---|---|---|
| 2026-10-21 | Action = Beta-distributed pace multiplier on adaptive-VWAP schedule | stable semantics across Q/H; strong naive prior at a=1 |
| 2026-10-21 | Bar-proxy imbalance on non-tick days | ρ ≈ 0.77-0.86 vs tick L1; lifts coverage 22% → 99.7% |
| 2026-10-21 | Structural fuzzy edits only between PPO iterations | keeps importance ratios valid |
| 2026-10-21 | Parametric impact (square-root + OW transient) with domain randomisation | no depth beyond L1; own impact unidentifiable |
| 2026-10-21 | Test split = Oct 1-19 (tick days) | highest-fidelity data reserved for final evaluation |
| 2026-10-21 | eFNN imbalance input = `imb_bar_proxy` on all splits | proxy/tick scale mismatch (std 0.42 vs 0.32) would masquerade as a regime shift between train and test |
| 2026-10-22 | Training-data pull is gap-driven per session (root notebook), on NSE `IS` tickers, with a fixed tick universe; no options, futures = generics only | the `LIMIT`/chunk-name collector re-downloaded everything, never repaired gaps or partial days, picked names alphabetically (weights are N/A sentinels) and mixed in the `IN` composite book; R05 needs the same names across pulls |
| 2026-10-22 | **Supersedes the row above.** The pull is frozen and the pull notebook is retired (it stays in git history at `de800d8`). The gate is the new `final` readiness tier | university scope; the full pull ran for 20+ min and returned far more data than needed; the research tier's 50 names and 40 tick days cannot be produced by processing |
| 2026-10-22 | Final dataset `features_1m_v3`: a 09:15-15:15 grid on every day; incomplete sessions dropped whole (Sep 10); flagged within-session last-observation fills; partial-window warm-up estimators; a `split` column | closing call auction from 2026-08-03 (DATA_AUDIT F10); imputing a whole session would fabricate data; R12 makes every episode row complete so the env never imputes |
| 2026-10-22 | One readiness gate (no tiers); the collector tooling leaves the repo (git history `de800d8`); every transformation is logged in `docs/DATA_TRANSFORMS.md` | the pull is frozen, so pilot/research tiers and pull tooling have no remaining use; the audit stays because R06-R12 guard the cleaning and feature code |
| 2026-10-23 | M1 simulator as built: every directional observation (incl. the imbalance regime input) is side-signed and scaled without centring; the risk term uses post-child inventory `X_{t+1}`; impact in mid fractions with `V_eff ≥ 1%` of expected volume; whole-share children; forced residual ignores the cap | the policy never sees `side`, so unsigned directional inputs would make buys and sells indistinguishable; post-trade inventory is the discrete AC variance, which makes TWAP match the analytic AC cost exactly (test); the floor stops a zero-volume minute from giving infinite impact |
| 2026-10-23 | Domain-randomised impact is drawn from each episode's seed; evaluation uses fixed defaults unless `--randomise-impact`; L1-depth half-spread fills are deferred to TickSim | paired comparisons need identical impact per episode across policies; BarSim has no depth on 78% of days |
| 2026-10-23 | M2: add `log_participation` and minutes-left to `z_p`; benchmark-relative (`vs_vwap`) reward for training; 256 envs, lr 1e-3 | the policy could not see order size or horizon; mark-to-market alone did worse (+0.2-0.4 bps vs VWAP) and arrival stayed at 0 ± 0.1 bps, because drift noise swamps a ~0.2 bps edge; the shadow-VWAP baseline is action-independent, so the optimum is unchanged (test) |
| 2026-10-23 | PPO-MLP result: beats VWAP on train in 3/3 seeds (−0.16 to −0.24 bps, CIs exclude 0); val −0.24 / +0.06 / −0.04 bps (docs/RESULTS.md) | the gain is timing (IS), not fees; the train→val gap of up to 0.27 bps is the small-data overfitting risk, so model selection and seeds matter for M3/M4 |
| 2026-10-23 | M3 eFNN as built: softmax-normalised firing with a per-rule support mass (merges of duplicates are exact); regime inputs bypass the running normaliser; edits keep Parameter objects and remap Adam moments row by row; an evolution pass whose policy KL on the buffer exceeds `target_kl` is undone | no underflow far from rules; MFs need a fixed input scale; the KL guard makes "no PPO instability at edits" enforced and auditable (0 rejections in 3 seeds, max edit KL 0.016) |
| 2026-10-23 | M3 result: fuzzy ≥ MLP on train (−0.20/−0.21 vs −0.17 bps vs VWAP, 3 seeds), but no policy beats VWAP reliably on val, and the M2 val gain did not replicate | the train→val gap dominates; M4 first reduces variance and overfitting (more seeds, ensembling, selection set), then compares architectures |
| 2026-10-24 | Training reward uses the expected price drift (`reward.price_drift: expected`): the inventory × mid-move term is dropped; evaluation still charges the realised IS | M2/M3 policies' train edge was entirely timing against the 180 train price paths (unchanged at zero impact) and did not transfer to val; with the martingale reward, the edge is impact reduction and transfers (MLP val −0.24 bps, cluster CI −0.34 to −0.14) |
| 2026-10-24 | Report session-cluster bootstrap CIs and session-level Wilcoxon tests, and seed ensembles as the headline policy | episodes on one session are correlated (per-episode CIs were ~2× too narrow on val); ensembling removes the seed lottery that made the M2 val result non-reproducible |
| 2026-10-24 | Leave-one-stock-out refits the observation scaler on the training names, and checkpoints store their scaler | no statistic of the held-out name reaches the policy |
| 2026-10-24 | M4 result: on the test split (run once), 5-seed ensembles beat VWAP by −0.16 (MLP), −0.24 (ANFIS) and −0.22 (eFNN) bps, all CIs excluding 0; they also beat TWAP/AC and win on held-out stocks (LOSO). Fuzzy − MLP on test −0.06 to −0.08 (Wilcoxon n.s.); eFNN ≈ ANFIS everywhere; eFNN > MLP under strong or persistent transient impact | the reward design, not the architecture, is the main effect; the fuzzy advantage is suggestive at default impact and clear only off the training range of transient impact (docs/RESULTS.md M4) |
| 2026-10-29 | M5 scope: full-universe data is dropped (the pull is frozen); M5 is TickSim plus limit orders on the 20 tick sessions per stock. TickSim is an evaluation simulator: BarSim-trained policies are replayed against the 1-second L1 grid on the val tick sessions, and never trained on tick days | ticks exist only on val (8) and test (12) sessions, so training on them would spend the selection or the test set |
| 2026-10-29 | TickSim fill: one order per decision at the bar boundary + 1 s, half spread only within displayed depth, BarSim's square-root walk on the excess; the bar/tick clock alignment comes from the data (bar `m` = seconds 60m+1…60m+60) | the measured gap to BarSim is entirely depth (−0.32 bps for VWAP; timing −0.004, n.s.), which validates BarSim's end-of-bar fill; 4 orders per decision cost +2.8 bps in per-order fees, so sub-minute slicing is a sensitivity option, not the default |
| 2026-10-29 | Limit orders: a passive-fraction action with a second Beta head; policies train in BarSim with a bar-range fill rule whose θ is fitted to a TickSim queue model on the val tick sessions, and are evaluated in TickSim with the queue model | train has no ticks, so training on tick days would spend val or test; the bar rule is causal (decision quote vs the next bar's range) and calibrated without touching test; a continuous fraction keeps PPO's log-probabilities simple and nests the pace-only policy at f = 0 |
| 2026-10-29 | Passive fills are capped by the volume that traded at or through our price (two calibrated BarSim caps, θ_touch and θ_through); the first limit runs (uncapped trade-through fills) are discarded | a naive 100%-passive VWAP beat VWAP by −3.0 bps in TickSim because any trade-through filled the whole order, however large and however small the print; with the cap the queue model fills 47% of posted shares instead of 72% |
| 2026-10-29 | Holm-adjusted p-values over each table's ensemble claims, plus date-cluster CIs/tests and the unweighted session t-interval, in every `eval compare` table; an independent 3-seed M4 replication is in RESULTS | three architectures tested against VWAP and each other is a family of claims; same-date sessions of different stocks are correlated; the replication confirmed the VWAP result in both runs and showed the architecture differences do not survive the correction |
| 2026-10-29 | Passive-fraction dead zone (`limit.dead_zone` = 0.1): fractions within 0.1 of 0 or 1 snap to it | the first limit policies settled at f ≈ 0.6, where every child is two orders and pays the ₹20 fee twice (+0.6-0.7 bps over VWAP); a continuous Beta cannot reach the single-order corners, so the reward had a cliff at f = 1 that gradient ascent could not cross |
| 2026-10-29 | Limit-order runs start the passive head at the naive always-passive rule (`init_passive` 0.95, `passive_concentration` 30) | from 0.2, PPO settled in a mixed-order local optimum 1.2-1.4 bps behind the naive rule; started at it, every seed improves on it (−0.43 to −0.46 bps in both simulators) with a lighter tail, as the pace head improves on VWAP from a = 1 |
| 2026-10-29 | M6 backend as built: per-order traces in `hftf_eval.trace` (policies may expose `explain(env)`), a FastAPI service `apps/api` on 127.0.0.1 with a typed OpenAPI contract, demo choices in `configs/demo.yaml`; UI direction in `docs/UI_DESIGN.md`: dark "quiet precision" with one signature gradient, glass only on floating chrome, 3D only for the eFNN's 3-D regime space (lazy-loaded) | hftf-eval stays torch-free; licensed data never leaves localhost; the frontend gets one typed contract; glass under charts costs contrast and frames, while the regime space is the one place 3D carries information |
