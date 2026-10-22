# Project Context: Regime-Aware Optimal Execution with Fuzzy PPO

> Read this before changing anything in the repo. It records what we are building, the domain knowledge
> behind each decision, and the exact meaning of the data. Binding design: `docs/ARCHITECTURE.md`.
> Data provenance and quirks: `docs/DATA_AUDIT.md`. Regenerate this file with
> `docs/prompts/project_context_prompt.md` after major changes.

---

## 1. TL;DR

- **Goal:** execute a parent order (buy or sell `Q` shares of an NSE stock within `H` minutes) at minimum
  **implementation shortfall** (slippage vs. arrival mid + market impact + fees + timing risk).
- **Method:** a PPO actor-critic whose state passes through an **evolving fuzzy neural network
  (eFNN)**. Its inputs are order-book imbalance, spread variance and realized volatility. It produces a
  soft regime assignment (normalised rule firing strengths) and adds, merges and prunes rules as regimes
  appear and fade. The actor outputs a **pace multiplier** on an adaptive-VWAP schedule (the slice size).
- **Learning:** the PPO policy loss and the Bellman/GAE value loss back-propagate into the Gaussian
  membership centres and widths. Structural rule edits happen between PPO iterations.
- **Data:** Bloomberg (blpapi) pull for 3 Nifty-50 stocks (ADE, ADSEZ, APHS). It has 96 sessions of 1-min
  bars, 20 sessions of conflated L1 ticks, and 15 years of daily data. **The pull is frozen.** The final
  table `features_1m_v3` has 92 usable sessions per stock (train 60 / val 20 / test 12) on a 09:15-15:15 grid.
- **State (2026-10-22):** M0 done. That covers the uv monorepo, the data audit, the final feature table
  `features_1m_v3` (readiness gate READY, 12/12), and tests. Next is M1: the execution simulator and
  classical baselines.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| Parent / child order | The full instruction (`Q` shares by `t0+H`) and the slices actually sent to the market. |
| Arrival price `m_0` | Mid-price at the decision time `t0`. The benchmark for implementation shortfall. |
| Implementation shortfall (IS) | `side · (avg exec price − m_0)/m_0`, in bps, plus fees. Positive = cost. Buy side = +1, sell = −1. |
| Slippage | The generic name for execution price minus benchmark. Always state which benchmark (arrival, VWAP, close). |
| TWAP | Equal slices per unit time. Optimal under zero risk aversion and linear impact with constant liquidity. |
| VWAP (benchmark) | Volume-weighted average market price over the interval. |
| VWAP (strategy) | Slices proportional to the *expected* intraday volume profile. |
| POV | Percentage of volume: trade a fixed fraction `ρ` of realised market volume. |
| Participation rate | Our child volume / market volume over the same interval. |
| Temporary impact | The price concession paid by our own child order. It vanishes for the next trade. |
| Permanent impact | The lasting shift in the mid caused by our trading (information leakage). |
| Transient impact | Impact that decays over time as the book refills (resilience). Obizhaeva-Wang. |
| L1 | Best bid/ask and the displayed sizes at them. We have no L2+ depth. |
| L1 imbalance | `(bid_sz − ask_sz)/(bid_sz + ask_sz) ∈ [−1, 1]`. Positive means a heavier bid, i.e. buy pressure. |
| OFI | Order-flow imbalance (Cont-Kukanov-Stoikov): net signed change in L1 liquidity. A strong short-horizon predictor of mid moves. |
| Microprice | `(ask·bid_sz + bid·ask_sz)/(bid_sz + ask_sz)`: the mid adjusted toward the thin side. |
| Realized vol (RV) | `√Σ r²` of 1-min log mid returns over a window. Not annualised in our features. |
| Spread variance | Rolling variance of the spread in bps. It measures how *unstable* liquidity is, not how expensive. |
| Regime | A persistent state of the market's liquidity and volatility. Here it is a soft (fuzzy) assignment, not a hard label. |
| Membership function (MF) | `μ(x) ∈ [0,1]`: the degree to which `x` belongs to a fuzzy set, e.g. "spread-var is High". We use Gaussians. |
| Firing strength | A rule's degree of activation, the product of its antecedent MFs (product t-norm). |
| TSK (Takagi-Sugeno-Kang) | Fuzzy rules whose consequents are functions (first-order = linear), blended by normalised firing. |
| eFNN / evolving fuzzy system | A fuzzy NN whose *structure* (number of rules) adapts online: rules are added on novelty and merged or pruned when redundant or stale. |
| PPO | Proximal Policy Optimization: a policy gradient with a clipped importance ratio `π/π_old`, giving stable on-policy updates. |
| GAE | Generalized Advantage Estimation: a λ-weighted sum of TD errors. It trades bias against variance. |
| Bellman / TD error | `δ_t = r_t + γV(s_{t+1}) − V(s_t)`. The critic's value loss is built from these errors. |

---

## 3. NSE market microstructure (what the agent trades into)

### 3.1 Session (IST, UTC+05:30; all stored timestamps are UTC)
| Phase | Time | Relevance |
|---|---|---|
| Pre-open order entry | 09:00-09:08 | Call auction. Bloomberg shows BID/ASK **price 0.0** (no visible book). |
| Pre-open matching + buffer | 09:08-09:15 | Opening price discovered. `AOB` print ≈ 09:08-09:09. |
| **Continuous trading** | **09:15-15:30 until 2026-07-31; 09:15-15:15 from 2026-08-03** | **The agent trades only 09:15-15:15 on every day** (360 one-minute bars; DATA_AUDIT F10). |
| Closing call auction | 15:15-~15:30 from 2026-08-03 | Measured in our data: indicative `T` prints, zero-price quotes, `AUC` uncross at ~15:29:40. Not modelled in v1. |
| Close / post-close | 15:30-16:00 | A post-close session trades at the close (not modelled). |

### 3.2 Tick size, lot size, price bands
- **Tick size follows price bands** (verify against current NSE circulars): `<₹250: 0.01`,
  `₹250-1000: 0.05`, `₹1000-5000: 0.10`, `₹5000-10000: 0.50`, `₹10000-20000: 1.00`, `>₹20000: 5.00`.
  Our data agrees: ADE and ADSEZ trade on a ₹0.10 grid, APHS on ₹0.50 (`hftf_data.derive`).
  Bloomberg did not return `TICK_SIZE`, so it is derived.
- **Cash-equity lot = 1 share.** Index futures: NIFTY lot 65, BANKNIFTY lot 30 (`static_futures`).
- F&O stocks have dynamic price bands and no fixed circuit. A breach is rare intraday and is not modelled.

### 3.3 Fee stack (configurable in `configs/fees_nse.yaml`; verify against current NSE/SEBI circulars)
| Component | Rate (delivery) | Type |
|---|---|---|
| STT | 0.1% on buy and sell | proportional |
| NSE transaction charge | 0.00297% of turnover | proportional |
| SEBI turnover fee | ₹10 per crore (0.0001%) | proportional |
| Stamp duty | 0.015% (buy side only) | proportional |
| GST | 18% on (brokerage + transaction charge + SEBI fee) | proportional |
| Brokerage | institutional bps and/or a fixed ₹ per order | **per child order (fixed part)** |

**Intuition.** The statutory costs total about 11.9 bps on a buy and about 10.4 bps on a sell. That is
5× the median spread (~2 bps). But they are proportional to notional, so **no slicing schedule can change
them**. What slicing *can* change is spread, impact and timing risk, plus any **fixed per-child fee**.
That fixed fee is the lever that penalises over-slicing; without it, the square-root law would push
toward infinitely many tiny slices. Report fees separately from IS so the learnable part stays visible.

### 3.4 Measured liquidity (our 3 stocks, Jun-Oct 2026)
| | ADE | ADSEZ | APHS |
|---|---|---|---|
| median price (₹) | 2999.6 | 1766.7 | 8759.3 |
| tick (₹) | 0.10 | 0.10 | 0.50 |
| median spread | 2.0 bps (6 ticks) | 1.8 bps (3 ticks) | 1.8 bps (3 ticks) |
| median ADV | 1.29M sh (₹385 cr) | 1.77M sh (₹316 cr) | 0.31M sh (₹267 cr) |
| median daily RV | 145 bps | 116 bps | 97 bps |
| median L1 depth (bid+ask) | 123 sh | 209 sh | 74 sh |

- **U-shaped intraday volume** (09:15-15:15 grid): the first 30 min ≈ 15% of the day's volume and the last
  hour ≈ 22%, both before and after the auction change. RV is highest at the open. **Spreads are widest in
  the first 5 min (3.5 bps).** Before August, 15:15-15:30 was wide too (3.8 bps), but those minutes are now
  the auction and are off the grid. A residual forced at the horizon pays impact on a thin touch.
- **L1 depth is tiny:** 1% of ADV is about 100× the displayed touch. Every non-trivial child walks the book,
  so price impact has to be modelled (§4.3), not read from data.

---

## 4. Optimal execution theory: the intuition the agent should rediscover

### 4.1 Decomposing implementation shortfall
`IS = spread cost + temporary impact + permanent/transient impact + timing drift + fees`
- Trading **fast** pays more impact (concentrated demand) and less timing risk.
- Trading **slow** pays less impact but leaves inventory exposed to adverse price moves (variance ∝ σ²·T·inventory²).
- Execution is a **cost-risk trade-off**. No schedule is best in every regime, and that is the reason for
  a regime-aware policy.

### 4.2 Almgren-Chriss (2000)
Minimise `E[cost] + λ·Var[cost]` with linear temporary impact `η` and volatility `σ`. The optimal
remaining inventory is `x(t) = X · sinh(κ(T−t)) / sinh(κT)` with `κ ≈ √(λσ²/η)`.
- As `λ → 0`, `x(t)` is linear, which is TWAP.
- Higher risk aversion or higher σ increases `κ`, which **front-loads** execution.
- AC is the classical baseline (`hftf-eval`). Our reward (§6.3) telescopes to exactly this objective,
  so a policy that beats AC is beating the textbook solution on its own terms.

### 4.3 Impact models used by the simulator
- **Square-root law** (Bouchaud et al.; Almgren et al.): cost per share ≈ `Y · σ · √(q/V)`. Impact is
  *concave* in size, so splitting reduces total cost. For `N` equal slices the impact cost scales like
  `Q · √(Q/(N·V))`, and it decreases in `N`. Our temporary term is `η · σ_1m · (q/V_1m)^β` with `β = 0.5`.
- **Obizhaeva-Wang transient impact:** each trade displaces the mid, and the displacement decays with
  resilience time `τ_res`. Trading again before the book refills compounds cost, so **bursts are
  penalised beyond the concave per-trade cost**. Implemented as an exponentially decaying sum.
- **Own impact cannot be identified from public data.** The parameters `η`, `γ`, `τ_res` are
  domain-randomised during training, and robustness to them is reported (ARCHITECTURE §4.4).

### 4.4 Regime hypotheses (priors the agent should discover; never hard-coded)
| Regime signal | Hypothesised optimal response | Mechanism |
|---|---|---|
| High **spread variance** | Be patient and trade opportunistically when the spread compresses | Crossing cost is unstable; waiting has option value |
| High **RV** | Accelerate if risk-averse; but impact also scales with σ | AC `κ ∝ σ`; the tension is exactly what must be learned |
| **Imbalance** favouring our side's direction (bid-heavy while buying) | Front-load | L1 imbalance predicts short-horizon mid moves; buy before the price lifts |
| Imbalance against us (ask-heavy while buying) | Slow down | Price is likely to drift in our favour |
| Low **relative volume** | Slow down | Impact ∝ `q/V` |
| Approaching the horizon | Avoid a large residual | The forced residual walks a thin L1 book; impact is concave but not free |

"High Volatility / Low Liquidity" and "Trend Bullish" are human names for regions of this input space.
The eFNN discovers its own regions, and §5.4 maps them back to such names.

---

## 5. Fuzzy systems and the eFNN

### 5.1 Inference (first-order TSK, product t-norm)
For a standardised regime input `x ∈ ℝ^d` (d = 3: imbalance, log spread-var, log RV) and rules `r = 1..R`:
```
τ_r(x) = Π_j exp(−½ ((x_j − c_rj)/σ_rj)²)       Gaussian MFs, product t-norm
φ_r(x) = τ_r / (Σ_k τ_k + ε)                     normalised firing (soft regime assignment)
y(x,h) = Σ_r φ_r(x) · (W_r h + b_r)              TSK consequent over the trunk embedding h
```
Read it as a **mixture of linear experts gated by fuzzy regimes**. The actor (Beta parameters) and the critic
(V) each have their own consequents but share the antecedents (`c`, `σ`), so both learn on one regime partition.

### 5.2 Lineage: what we borrow
| System | Idea we use |
|---|---|
| ANFIS (Jang, 1993) | Gradient training of MF parameters and consequents end-to-end |
| TSK (Takagi & Sugeno, 1985) | Linear consequents blended by normalised firing |
| eTS (Angelov & Filev, 2004) | Online rule creation driven by novelty/density of new data |
| DENFIS (Kasabov & Song, 2002), ePL (Lima et al., 2010) | Evolving clustering for antecedents |
| PANFIS / GENEFIS (Pratama et al., 2014) | Rule merging by overlap and pruning by statistical contribution (utility) |
| Lughofer, *Evolving Fuzzy Systems* (2011) | Stability-plasticity and rule-redundancy handling |

### 5.3 How the Bellman updates shape the membership functions
`c_r` and `σ_r = softplus(ρ_r)` are ordinary `nn.Parameter`s. The PPO loss is
`L = L_clip − c_e·Entropy + c_v·L_V`, where `L_V = (V(s) − V_target)²` and `V_target` comes from GAE/TD(λ).
The gradient of `L_V` with respect to `c_r` is non-zero because `V = Σ φ_r(x)·v_r(h)` depends on `φ`.
So a large Bellman error in some region of `x` moves the nearby MF centres and widths toward a partition
that makes values predictable. In other words, **regimes are defined by what matters for cost, not by
unsupervised density alone.** The antecedents get a smaller learning rate (default 0.3× base). The config
flag `antecedent_grad ∈ {all, value_only, none}` makes this an explicit ablation.

### 5.4 Structure evolution
- **Add** when a meaningful mass of buffer samples is poorly covered (`max τ < τ_add`): place a rule at their
  mean with their std as width.
- **Merge** two rules whose Gaussians overlap strongly (Bhattacharyya coefficient > `τ_merge`).
- **Prune** rules whose long-run utility (EMA of mean `φ_r`) falls below `τ_prune` after a minimum age.
- **Why only between PPO iterations?** PPO's ratio `π_θ(a|s)/π_old(a|s)` assumes `π_old` is exactly the
  network that generated the buffer. Changing `R` mid-update invalidates the stored log-probs. Edits
  therefore happen after the update epochs and before the next rollout.
- **Function-preserving insertion.** A new rule's consequent is initialised to the firing-weighted average
  of the existing consequents at the uncovered samples. The network's output barely changes at insertion,
  so the policy does not jump and the next iteration's KL stays under target. Merges use support weighting
  for the same reason.
- **Interpretation:** each rule centre is mapped to Low/Med/High per input (train terciles) and to its
  average pace, e.g. *IF spread-var HIGH ∧ RV HIGH ∧ imbalance NEUTRAL THEN pace 0.6*. Rule activations are
  compared with the eval-only regime labels using adjusted mutual information.

---

## 6. RL formulation (as specified in ARCHITECTURE §4 and §6)

### 6.1 Episodes
`(stock, session, t0, H ∈ {30,60,120} min, side ∈ {buy,sell}, Q = ρ·expected volume over the horizon, ρ ~ LogU[1%,20%])`,
with `t0 ∈ [09:20, 15:15−H]`. Decision step Δ = 1 min. The residual is force-executed at the horizon.

### 6.2 Observation
- `x` (to the eFNN): `imb_bar_proxy` (5-min EMA), `log(spread_var_30m+ε)`, `log(rv_30m+ε)`.
  **Use the proxy on all splits.** Train has only proxy days and test only tick days, and the two scales
  differ (std 0.42 vs 0.32). The mixed `l1_imb` column would look like a regime shift.
- Private: remaining inventory fraction, remaining time fraction, IS so far (bps), last action, schedule deviation.
- Market: `spread_bps`, `dev_vwap_bps`, side-signed `ret_5m`, expected next-5-min profile share,
  side-signed `ctx_nifty_ret_30m`, `ctx_vix`.
- All features are standardised with **train-split statistics only**.

### 6.3 Action and reward
- Action `a ∈ [0, 4]` from a scaled Beta distribution. Child
  `q = clip(a · X · w_t/Σ_{s≥t} w_s, 0, min(X, 0.3·V̂_t))`. `a = 1` is adaptive VWAP, `a = 0` waits.
  The Beta is chosen because it is bounded, so no clipping bias arises as with Gaussians. The
  multiplicative form keeps the meaning of `a` the same for any `Q` and `H`.
- Reward per step (bps of arrival notional):
  `r_t = −side·(p_exec − m_0)/m_0·(q_t/Q)·1e4 − fees_t − λ·(σ_t·1e4)²·(X_t/Q)²·Δ`.
  The sum over the episode is `−(IS + fees + λ·risk)`, the AC objective. The discount is `γ = 1` because
  episodes are finite and cost is additive.

### 6.4 PPO
Clip 0.2, GAE λ = 0.95, 10 epochs × 8 minibatches, Adam 3e-4, entropy 1e-3, value coefficient 0.5,
gradient clip 0.5, target KL 0.02, 64 parallel episodes × 120 steps, 2×64 tanh trunk. Observations and
rewards are normalised. **CPU only (no GPU on the dev machine):** the environment is a batched numpy
simulator, and networks stay small.

---

## 7. Data

### 7.1 Inventory (raw: `data/is2/bbg_data/`, pulled 2026-10-21 06:44-07:00 UTC)
| Dataset | Instruments | Range | Granularity |
|---|---|---|---|
| `intraday_bars/1m/TRADE` | ADE, ADSEZ, APHS; NIFTY, NSEBANK, INVIXN (India VIX); NZ1, NZ2 (Nifty fut), AF1 (BankNifty fut) | 2026-06-03 → 10-21 12:05 IST (96 sessions) | 1 min OHLCV, `numEvents`, `value` (₹ turnover) |
| `intraday_bars/1m/BID`, `/ASK` | 3 equities, 3 futures, USDINR, EURINR (DXY empty) | same | 1 min quote OHLC; `volume` = sum of displayed sizes |
| `ticks/<sec>/<YYYYMMDD>` | ADE, ADSEZ, APHS | 2026-09-21 → 10-21 (20 full + 1 partial) | ~1 s conflated TRADE/BID/ASK with sizes and condition codes |
| `daily/daily_<class>` | 6 equities (+ACC, ACEM, ARBP daily only), 3 indices, 3 futures, FX, rates, commodities, 3 Nifty options | 2011-10-21 → 2026-10-21 | OHLCV, vols, implied vol, adjusted + raw prices |
| `reference/` | static per class, Nifty membership, futures/options chains, field validation | snapshot | |
| `events/` | dividends, earnings dates | history | |
| `features/features_1m.parquet` | collector output | | **Defective: use `data/processed/features_1m_v3.parquet`** |

### 7.2 `features_1m_v3` column dictionary (103,680 rows × 91 cols; one row per security × session minute)
`time` is the bar **start** (UTC). Every value in a row is known at `decision_time = time + 1 min`.

| Group | Columns | Notes |
|---|---|---|
| Identity / time | `security`, `time`, `decision_time`, `date` (IST session date), `mso` (minutes since 09:15), `min_to_close`, `tick_size` | grid = 09:15-15:15 IST, 360 min/session |
| Split | `split ∈ {train, val, test, excluded}` | from `configs/env.yaml`; `excluded` = partial day or profile burn-in |
| Bars | `o`, `h`, `l`, `c`, `volume`, `n_events`, `value` | no-trade minutes: volume 0, OHLC = last close (`fill_trade`) |
| Quotes | `bid`, `ask`, `mid`, `px` (mid, else close), `spread_bps`, `spread_ticks` | last quote in the minute, ffilled within the day (`fill_quote`) |
| Returns / vol | `ret_1m`, `ret_{5,15,30,60}m`, `rv_{5,15,30,60}m`, `parkinson_30m` | log mid returns, not annualised; reset each session; in the first `w` min: `ret` since the open, `rv` rescaled by w/n (`dq_warmup`) |
| Regime inputs | `imb_bar_proxy_ema5`, `log_spread_var_30m`, `log_rv_30m` | the eFNN `x` (§6.2); log offset ε = 1e-8 |
| Spread dynamics | `spread_var_{15,30,60}m`, `spread_mean_30m` | variance of `spread_bps` (bps²) |
| Volume / VWAP | `turnover`, `cumvol`, `vwap_run`, `dev_vwap_bps` | session-to-date VWAP |
| Volume profile | `prof_share`, `prof_cum_share`, `prof_avg_vol`, `rel_volume` | previous 20 sessions only (causal) |
| L1 (all days) | `imb_bar_proxy` | **the eFNN imbalance input** |
| L1 (tick days only, ~22%) | `l1_imb_last`, `l1_imb_mean`, `microprice_dev_bps`, `tick_spread_bps_mean`, `tick_spread_bps_var`, `ofi`, `ofi_norm`, `ofi_5m`, `ofi_norm_5m`, `l1_depth_mean`, `n_quote_updates`, `bid_sz_last`, `ask_sz_last` | NaN on non-tick days, never zero-faked |
| Trade flow (tick days) | `n_trades`, `tick_volume`, `buy_vol`, `sell_vol`, `signed_vol`, `signed_vol_5m`, `trade_imb` | Lee-Ready sign vs. the prior quote |
| Unified L1 | `l1_imb`, `l1_imb_source ∈ {tick, bar_proxy, none}`, `has_ticks` | for analysis; **not** an eFNN input (§6.2) |
| Market context | `ctx_nifty_ret_1m`, `ctx_nifty_ret_30m`, `ctx_nifty_rv_30m`, `ctx_banknifty_ret_1m`, `ctx_banknifty_ret_30m`, `ctx_vix`, `ctx_vix_chg_30m` | same-minute index bars |
| Daily (lagged 1 session) | `d_ret_1d`, `d_vol_20d` (annualised), `d_adv_20d`, `d_range_pct`, `d_amihud_20d`, `d_ivol_30d`, `d_volume` | known before the open |
| Data quality | `dq_partial_day`, `dq_burnin`, `dq_warmup`, `dq_offbook_frac`, `fill_trade`, `fill_quote` | filter/flag only; not model inputs |
| Eval only | `regime_eval_prevday ∈ {low, mid, high}` | previous-day RV tercile. **Never a model input** |

### 7.3 Raw-data quirks and their fixes (details: DATA_AUDIT F1-F11)
| Quirk | Fix |
|---|---|
| Ticks are ~1 s conflated snapshots. `T` prints are cumulative and `ACB` is a day total | Keep only untagged + `BC` prints in 09:15-15:30. They reconcile to bar volume at 0.9996-1.0005 |
| BID and ASK rows share timestamps, so a row-wise ffill creates ~3% crossed books | Collapse to the last BID/ASK per timestamp, then ffill. Leaves 0 crossed |
| Pre-open quotes have price 0.0 | Masked to NaN; trading only in the continuous session |
| `-2.424536e-14` sentinel (weights, EPS) | → NaN (`loaders.clean_sentinels`) |
| `TICK_SIZE`, `TRADING_LOT_SIZE` not returned | Tick inferred from the price grid and checked against the NSE rule; equity lot = 1 |
| 2026-09-10 bars missing (8/9 instruments; ADE has no BID) | Collection gap. The session is dropped whole for every stock, never imputed |
| Closing call auction from 2026-08-03 (15:15-15:30) | Grid ends at 15:15 on every day |
| Missing minutes inside a session (5 rows in usable sessions) | Last observation carried within the session and flagged (`fill_trade`, `fill_quote`) |
| Rolling windows NaN after the open | Partial-window estimators over at least 3 same-session observations (`dq_warmup`) |
| Holidays Jun 26, Sep 14, Oct 2, Oct 20 | Legitimately empty |
| Oct 21 partial (to ~12:05 IST) | `dq_partial_day`; excluded |
| Aug 31, Sep 18, Sep 29: bars hold only 42-65% of daily volume | `dq_offbook_frac`; kept, flagged |
| Daily index files have all-NaN weekend rows | Dropped in the loader |

### 7.4 The imbalance proxy
A BID/ASK bar's `volume/numEvents` is the mean displayed size at the touch over that minute. The proxy
`(b̄ − ā)/(b̄ + ā)` correlates with the tick-derived L1 imbalance at **Pearson 0.771 (ADE), 0.838 (ADSEZ),
0.861 (APHS)** over about 7.4k overlapping minutes each. It is noisier and wider (std 0.42 vs 0.32) but
available on every bar day.

### 7.5 Splits (walk-forward by date; counts from the data)
| Split | Dates | Sessions | Tick days |
|---|---|---|---|
| train | 2026-06-08 → 08-31 | 60 | 0 |
| val | 2026-09-01 → 09-30 (09-10 absent) | 20 | 8 |
| test | 2026-10-01 → 10-19 | 12 | 12 |
| excluded | 2026-06-03 → 06-05 (profile burn-in), 2026-10-21 (partial) | 4 | 1 |

---

## 8. Invariants and conventions

1. **Causality:** use only columns observable at `decision_time`. No same-day daily fields (`d_*` are
   lagged). The volume profile comes from prior sessions. Normalisation statistics are fit on train only.
   `regime_eval_prevday` and `dq_*` are never model inputs.
2. **Time:** store in UTC, reason in IST. `date` is the IST session date. Never compare naive and
   tz-aware timestamps.
3. **Units:** prices in ₹; costs, IS and spreads in **bps of arrival notional**; volume in shares; `side`
   = +1 buy / −1 sell, and every cost formula is side-signed so buys and sells share one code path.
4. **Reproducibility:** every experiment has a YAML in `configs/`, a seed, and paired episode seeds across
   policies.
5. **Monorepo boundaries:** data logic lives in `hftf-data`, the market simulation in `hftf-sim`, torch fuzzy
   modules in `hftf-fuzzy` (no sim dependency), training in `hftf-rl`, and baselines/metrics in `hftf-eval`.
   Entry points go in `apps/cli`. **Notebooks never contain logic** that other code imports.
6. **Licensed data:** Bloomberg data is never committed (`data/` is git-ignored) and never published raw.
7. **Tooling:** `uv sync`, `uv run pytest`, `uv run ruff check`, `uv run hftf ...`. Python 3.12, pandas 3
   (copy-on-write; str dtype), CPU torch from the PyTorch CPU index.

---

## 9. Evaluation protocol

- **Baselines:** immediate execution, TWAP, VWAP (profile), POV 10%, Almgren-Chriss (same σ/η/λ),
  PPO-MLP, PPO + static fuzzy (ANFIS, no evolution), **PPO + eFNN (ours)**.
- **Ablations:** `antecedent_grad = none / value_only`; evolution off; raw regime features into the MLP;
  hard k-means gate.
- **Metrics:** IS mean/median/std, CVaR95(IS), VWAP slippage, fees, the risk-adjusted objective, completion,
  and participation profile. Everything is broken down by regime (eval labels and learned rules).
  Training diagnostics: rule count, add/merge/prune events, rule-regime AMI, KL at structural edits.
- **Statistics:** identical episode seeds across policies; paired bootstrap 95% CI; Wilcoxon signed-rank
  against VWAP and PPO-MLP.
- **What counts as a win:** a statistically significant reduction in mean IS **or** CVaR95 vs. VWAP and
  vs. PPO-MLP on the test split, with the advantage concentrated in the regimes where theory predicts it
  (§4.4), and robust across the impact-parameter sweep. A gain that appears under only one impact setting
  is not a win.

---

## 10. Known limitations and pitfalls

- **Tiny universe:** 3 stocks and 60 training sessions. Overfitting is the default outcome. Mitigations:
  episode augmentation, domain randomisation, small networks, early stopping on val, and a
  leave-one-stock-out check. The pull is frozen, so state this limit in the report.
- **Structural break inside train:** the closing auction starts 2026-08-03, so 39 train sessions come before
  it and 21 after. The common 15:15 grid removes the mechanical difference. Report val/test results knowing that.
- **Conflated ticks:** about 1 s snapshots. Queue dynamics, true OFI at event resolution, and sub-second
  effects are unobservable.
- **No L2 depth:** impact is parametric, and all results are conditional on the simulator.
- **Counterfactual backtests:** history did not see our orders. Permanent/transient impact enters only
  through the simulated mid `m̃`.
- **Rolling Bloomberg windows:** about 140 days of bars and about 1 month of ticks. Pull regularly or lose data.
- **Look-ahead traps specific to this data:**
  - using `time` instead of `decision_time` when joining fills;
  - using same-minute `volume` or `V_{t+1}` in the observation (the simulator may use realised volume,
    the agent may not);
  - fitting scalers on all dates;
  - using the eval-only regime label or `dq_offbook_frac` (which uses same-day daily volume) as an input;
  - mixing tick and proxy imbalance across splits.
- **PPO with a changing architecture:** edit only between iterations, preserve function, rate-limit
  edits, and monitor KL around edit events.
- **Fees dwarf the spread:** if headline IS includes STT, small learnable gains look negligible. Always
  report the components.

---

## 11. Status and next steps

| Milestone | Status |
|---|---|
| M0: uv workspace, data audit, final `features_1m_v3`, tests | **done** |
| M1: `hftf-sim` (BatchEnv, ImpactModel, FeeModel, EpisodeSampler) + TWAP/VWAP/AC | next |
| M2: PPO-MLP | planned |
| M3: static fuzzy → evolving fuzzy | planned |
| M4: experiments, ablations, robustness, report | planned |
| M5: full-universe data, TickSim, limit orders | planned |

**Data readiness:** `uv run hftf data build-features && uv run hftf data audit --strict` gives pass/fail
against `docs/DATA_READINESS.md`: **READY, 12/12**. Every transformation is logged in `docs/DATA_TRANSFORMS.md`.

**Data actions:** none. The pull is frozen (ARCHITECTURE decision log, 2026-10-22), and the retired pull
notebook and the collector are in git history at `de800d8`.

---

## 12. Open questions (recommended default in bold)

1. Decision frequency on tick days: **keep 1 min for v1**; try 10-15 s in TickSim (M5).
2. Should the eFNN input include `rel_volume`? **Not in v1** (keeps the 3 inputs from the vision); add it as an ablation.
3. Risk aversion `λ`: **sweep {0, 1e-5, 1e-4, 1e-3}**; report the frontier rather than one point.
4. Per-child fixed fee: **₹20/order in the default config** (discount-broker style); sweep 0-₹50.
5. Should the regime input use EMA smoothing of `imb_bar_proxy`? **Yes, 5-min EMA**, because the raw proxy is noisy.
6. Should one agent be trained per stock or pooled? **Pooled**, with per-stock standardisation; plus a leave-one-stock-out check.
7. The collector's `features_1m.parquet` is still on disk. **Leave it untouched** (raw is read-only) and ignore it.

---

## 13. References

- Almgren, R. & Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk*.
- Obizhaeva, A. & Wang, J. (2013). Optimal trading strategy and supply/demand dynamics. *Journal of Financial Markets*.
- Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*.
- Bouchaud, J.-P., Bonart, J., Donier, J. & Gould, M. (2018). *Trades, Quotes and Prices*. Cambridge University Press.
- Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Schulman, J. et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR.
- Nevmyvaka, Y., Feng, Y. & Kearns, M. (2006). Reinforcement learning for optimized trade execution. ICML.
- Ning, B., Lin, F. H. T. & Jaimungal, S. (2021). Double deep Q-learning for optimal execution. *Applied Mathematical Finance*.
- Takagi, T. & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. *IEEE Trans. SMC*.
- Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Trans. SMC*.
- Angelov, P. & Filev, D. (2004). An approach to online identification of Takagi-Sugeno fuzzy models. *IEEE Trans. SMC-B*.
- Kasabov, N. & Song, Q. (2002). DENFIS: Dynamic evolving neural-fuzzy inference system. *IEEE Trans. Fuzzy Systems*.
- Lughofer, E. (2011). *Evolving Fuzzy Systems: Methodologies, Advanced Concepts and Applications*. Springer.
- Pratama, M., Anavatti, S., Angelov, P. & Lughofer, E. (2014). PANFIS: A novel incremental learning machine. *IEEE Trans. Neural Networks and Learning Systems*.

---

```
generated:      2026-10-21 (Claude Code, from docs/prompts/project_context_prompt.md)
git commit:     none (repository initialised, no commits yet)
data snapshot:  Bloomberg pull 2026-10-21 06:44-07:00 UTC (LIMIT=3 pass)
sources read:   docs/ARCHITECTURE.md, docs/DATA_AUDIT.md, docs/data/data_audit.json,
                packages/hftf-data/src/hftf_data/*.py, data/processed/features_1m_v3.parquet,
                data/is2/bbg_collector.py, pyproject.toml, packages/*/pyproject.toml
```
