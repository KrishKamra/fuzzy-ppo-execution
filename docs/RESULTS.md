# Results log

Results as they are produced, with the commands that regenerate them. Costs are in **bps of arrival
notional** (higher is worse). All comparisons are **paired**: every policy runs on identical episodes with
identical impact. The test split was used once, at the end of M4, after every choice was fixed.

## M2: PPO-MLP vs baselines (2026-10-23)

Setup: `configs/ppo.yaml` (256 envs × 120 steps, lr 1e-3, 6M steps, 2×64 tanh trunk, Beta pace head) and
`configs/env.yaml` (`reward.mode: vs_vwap`; domain-randomised impact in training, default impact in evaluation).
`best.pt` is picked by val objective on the trainer's own 2,000 episodes (seed 12345). The numbers below are
on **fresh** episodes (seed 1, 4,000 per split).

```bash
uv run hftf train ppo --run ppo_mlp_s0 --seed 0          # ~2 min on CPU
uv run hftf eval policy runs/ppo_mlp_s0/best.pt --split train
uv run hftf eval policy runs/ppo_mlp_s0/best.pt --split val
```

### Seed 0 (best checkpoint: iteration 117 of 195)

| split | policy | IS | IS sd | CVaR95 | fees | objective | vs mkt VWAP | children |
|---|---|---|---|---|---|---|---|---|
| train | immediate | 8.77 | 8.99 | 32.76 | 11.15 | 19.92 | 8.24 | 1.0 |
| train | TWAP | 4.61 | 27.96 | 73.38 | 12.15 | 16.86 | 4.08 | 69.8 |
| train | VWAP | 4.60 | 27.64 | 72.00 | 12.15 | 16.84 | 4.07 | 69.8 |
| train | POV 10% | 5.34 | 23.25 | 63.14 | 11.52 | 16.92 | 4.81 | 42.0 |
| train | Almgren-Chriss | 4.61 | 26.99 | 70.78 | 12.15 | 16.85 | 4.08 | 69.8 |
| train | **PPO-MLP** | **4.36** | 26.43 | 69.97 | 12.15 | **16.60** | 3.83 | 69.8 |
| val | immediate | 8.77 | 9.66 | 34.84 | 11.15 | 19.92 | 7.69 | 1.0 |
| val | TWAP | 4.94 | 28.72 | 74.02 | 12.16 | 17.19 | 3.86 | 69.8 |
| val | VWAP | 4.92 | 28.26 | 72.54 | 12.16 | 17.17 | 3.84 | 69.8 |
| val | POV 10% | 6.02 | 23.57 | 64.89 | 11.53 | 17.61 | 4.95 | 42.5 |
| val | Almgren-Chriss | 4.91 | 27.54 | 71.06 | 12.16 | 17.15 | 3.83 | 69.8 |
| val | **PPO-MLP** | **4.68** | 27.59 | 72.53 | 12.16 | **16.92** | 3.60 | 69.8 |

PPO − VWAP (objective): train **−0.240** (95% CI −0.351, −0.130), val **−0.243** (−0.344, −0.142).
The gain is entirely IS (timing). Fees and child count equal VWAP's.

### Across seeds (PPO − VWAP objective, fresh episodes)

| seed | train | val | train → val gap |
|---|---|---|---|
| 0 | −0.240 (−0.351, −0.130) | −0.243 (−0.344, −0.142) | 0.00 |
| 1 | −0.204 (−0.384, −0.024) | +0.061 (−0.114, +0.237) | 0.27 |
| 2 | −0.161 (−0.274, −0.048) | −0.035 (−0.160, +0.091) | 0.13 |

**Exit criterion met:** PPO-MLP beats VWAP on train in 3/3 seeds, with every CI excluding 0. *(M3 update: after re-training,
the val gains did not replicate; see the M3 section.)* The val gap is
0 to 0.27 bps. Seed 0 being the best of three is itself a selection, so M4 reports the mean over seeds.

### What did not work (kept for the report)

| variant (3M steps) | best val vs VWAP | note |
|---|---|---|
| arrival reward, 64 envs, lr 3e-4 | ≈ 0 ± 0.1 | edge lost in drift noise |
| mark-to-market reward, 64 envs | +0.2 to +0.4 (worse) | explained variance ~0.3 |
| vs-VWAP reward, 64 or 256 envs, lr 3e-4 | −0.1 ± 0.09 | learns slowly |
| vs-VWAP reward, 256 envs, lr 1e-3 | −0.23 ± 0.07 | adopted |
| + linear LR annealing (6M, 3 seeds) | −0.21 / −0.04 / −0.19 | no clear gain; off by default |

### Reading

- On default impact the classical schedules are within 0.05 bps of each other. AC ≈ TWAP because
  λ = 1e-4 makes κ small. Fees (~11-12 bps, mostly STT) are larger than every IS difference, so always
  report IS and fees separately.
- PPO-MLP's edge (~0.2 bps) is small but significant on paired episodes. It is the bar that the eFNN (M3) must beat.

## M3: static fuzzy (ANFIS) and evolving fuzzy (eFNN) vs MLP (2026-10-23)

Same PPO config as M2. The fuzzy policies use `configs/efnn.yaml`: R0 = 4 rules from k-means, antecedent LR
×0.3, `antecedent_grad: all`. ANFIS is the same network with evolution off. All three were re-trained on seeds
0-2 after M3's one shared change: the observation normaliser now passes the 3 regime inputs through unchanged.
Evaluation uses fresh paired episodes (seed 1, 4,000 per split) on `best.pt` (selected by val).

```bash
uv run hftf train ppo --run m3_efnn_s0 --policy fuzzy --seed 0      # --no-evolve for ANFIS, --policy mlp
uv run hftf eval policy runs/m3_efnn_s0/best.pt --split val
uv run hftf eval rules runs/m3_efnn_s0/best.pt
```

### Objective vs VWAP (bps; 95% CI)

| policy | seed | train | val | val vs MLP (same seed) | rules (best.pt) |
|---|---|---|---|---|---|
| MLP | 0 | −0.051 (−0.164, +0.061) | −0.022 (−0.127, +0.084) | | |
| MLP | 1 | −0.210 (−0.365, −0.056) | +0.059 (−0.110, +0.228) | | |
| MLP | 2 | −0.248 (−0.411, −0.085) | +0.127 (−0.060, +0.313) | | |
| ANFIS | 0 | −0.162 (−0.250, −0.074) | −0.017 (−0.110, +0.076) | +0.004 | 4 |
| ANFIS | 1 | −0.297 (−0.453, −0.140) | −0.047 (−0.242, +0.148) | −0.106 | 4 |
| ANFIS | 2 | −0.154 (−0.215, −0.093) | −0.075 (−0.152, +0.003) | −0.201 (−0.363, −0.040) | 4 |
| eFNN | 0 | −0.259 (−0.385, −0.134) | +0.057 (−0.072, +0.185) | +0.078 | 7 |
| eFNN | 1 | −0.005 (−0.044, +0.034) | +0.020 (−0.018, +0.058) | −0.039 | 5 |
| eFNN | 2 | −0.368 (−0.570, −0.166) | +0.067 (−0.153, +0.287) | −0.060 | 3 |
| **mean** MLP / ANFIS / eFNN | | −0.170 / −0.204 / −0.211 | +0.055 / −0.046 / +0.048 | ANFIS −0.101, eFNN −0.007 | |

eFNN seed 1's `best.pt` is iteration 3, i.e. almost VWAP: its val curve never beat the start.

### Structure evolution (M3 exit criterion)

| eFNN seed | add / merge / prune (all accepted) | rejected | max edit KL | PPO KL in the update after an edit (mean / max) | PPO KL max, whole run | R range, final |
|---|---|---|---|---|---|---|
| 0 | 13 / 5 / 6 | 0 | 0.0048 | 0.0029 / 0.0044 | 0.0065 | 5-9, 6 |
| 1 | 16 / 12 / 5 | 0 | 0.0162 | 0.0029 / 0.0042 | 0.0057 | 3-8, 3 |
| 2 | 8 / 6 / 3 | 0 | 0.0134 | 0.0029 / 0.0044 | 0.0054 | 3-8, 3 |

**Exit criterion met:** every add, merge and prune is logged (`runs/*/events.csv`). Every edit's KL is below
`target_kl` = 0.02, and PPO's own KL after an edit is the same as elsewhere (~0.003), so edits do not
destabilise PPO.

### Learned rules (eFNN seed 0, `best.pt`, firing on val)

| rule | IF imbalance | spread-var | RV | THEN pace | share |
|---|---|---|---|---|---|
| 4 | NEUTRAL | LOW | MED | 0.96 | 77.8% |
| 5 | NEUTRAL | LOW | LOW | 1.01 | 5.7% |
| 3 | NEUTRAL | **HIGH** | **HIGH** | **0.90** | 4.5% |
| 6 | AGAINST | LOW | LOW | 1.02 | 3.9% |
| 2 | AGAINST | MED | LOW | 1.05 | 3.3% |
| 0 | WITH | MED | LOW | 0.90 | 3.2% |
| 1 | NEUTRAL | MED | HIGH | 0.90 | 1.7% |

Seed 2 ends with 3 rules: a dominant calm regime (92%, pace 0.93) and a HIGH spread-var / HIGH RV rule (pace 0.83).
The direction matches the theory (`project_context.md` §4.4: slow down when spreads and volatility are high).
Each run, though, collapses onto one dominant regime.

### Reading (honest)

- On **train**, both fuzzy variants are at least as good as the MLP on average (−0.20 / −0.21 vs −0.17 bps).
- On **val**, no policy beats VWAP reliably. Seed means are within ±0.06 bps of VWAP, and only one seed's CI
  (ANFIS s2 vs MLP) excludes 0. The **M2 val result for MLP seed 0 (−0.24) did not replicate**: re-trained after
  the normaliser change, MLP seeds give −0.02 / +0.06 / +0.13 on val. Seed-to-seed variance is larger than the
  M2 three-seed table suggested.
- The train→val gap (~0.2 bps) is the main obstacle, not the architecture. M4 should go after it before
  comparing architectures: more seeds, seed-averaged or ensembled policies, a less noisy selection set, and
  regularisation.

## M4: experiments, ablations, robustness and the final test (2026-10-24)

### What changed, and why

M3 left every policy within ±0.06 bps of VWAP on val. Two diagnostics explained why:

1. **The train edge was timing against the price paths, not impact.** Re-evaluating the M3 policies at zero
   impact left their edge over VWAP unchanged (e.g. eFNN s0 on train −0.33 bps total, −0.36 at zero impact).
   With 180 train stock-sessions, the policy memorised when those particular paths moved, and that did not
   transfer.
2. **The CIs were too narrow.** Episodes on one session share a price path. Resampling whole sessions
   (60 on val) roughly doubles the val CI width compared with the per-episode normal CI used in M2/M3.

So M4 makes three changes (decision log 2026-10-24):

- **Training reward = expected cost (`reward.price_drift: expected`).** The inventory × mid-move term of the
  mark-to-market split is replaced by its martingale expectation, 0. The return is then the Almgren-Chriss
  objective: expected execution cost plus the risk penalty. **Evaluation still charges the realised IS** on
  the real paths.
- **Statistics:** session-cluster bootstrap 95% CIs and Wilcoxon signed-rank tests on per-session means
  (`hftf_eval.stats`).
- **Seed ensembles:** a group's headline policy is the mean pace of its seeds' `best.pt`; every seed is listed too.

Everything below is on **fresh** paired episodes (seed 1, 4,000 per split) at default impact unless stated.
Raw tables: `runs/reports/*.csv` (git-ignored; regenerate with the commands shown).

```bash
uv run hftf train ppo --run m4_efnn_s0 --policy fuzzy --seed 0       # --policy mlp; --no-evolve for ANFIS
uv run hftf eval compare mlp='runs/m4_mlp_s*/best.pt' anfis='runs/m4_anfis_s*/best.pt' \
    efnn='runs/m4_efnn_s*/best.pt' --split val --vs mlp --by-regime
uv run hftf eval robustness mlp='runs/m4_mlp_s*/best.pt' efnn='runs/m4_efnn_s*/best.pt' --param tau_res_min \
    --values 1,2,5,15,30 --vs mlp
```

### Main result: 5 seeds per architecture (objective − VWAP, bps; 95% session-cluster CI)

| policy | train (180 sessions) | val (60 sessions) | val, each seed | val − MLP |
|---|---|---|---|---|
| PPO-MLP | −0.159 (−0.250, −0.067) | **−0.239** (−0.350, −0.136) | −0.22 / −0.21 / −0.24 / −0.23 / −0.21 | |
| PPO-ANFIS | −0.165 (−0.262, −0.064) | **−0.249** (−0.353, −0.155) | −0.23 / −0.30 / −0.25 / −0.19 / −0.17 | −0.011 (−0.059, +0.035) |
| PPO-eFNN | −0.181 (−0.279, −0.085) | **−0.247** (−0.363, −0.149) | −0.27 / −0.25 / −0.15 / −0.25 / −0.24 | −0.009 (−0.049, +0.028) |

- **All 15 policies beat VWAP on val**, each with a CI that excludes 0 (Wilcoxon p ≤ 0.005). With the
  realised-drift reward of M2/M3, the same architectures were at +0.006 (MLP) and −0.004 (eFNN) on val.
- Val is not worse than train (it is slightly better), so the edge now transfers to unseen days.
- **The gain is impact, not timing.** At zero impact, the new MLP seeds are within ±0.06 bps of VWAP. The
  impact part is −0.18 to −0.22 bps on both train and val. IS falls from 4.92 to 4.70 bps; fees and child
  counts equal VWAP's.
- **At default impact the architectures are indistinguishable:** fuzzy − MLP is −0.01 ± 0.05 bps on val.
  On train, eFNN − MLP is −0.023 (−0.047, +0.001).

### By eval-only regime (previous-day RV tercile; val, ensembles)

| prev-day RV | episodes | MLP | ANFIS | eFNN |
|---|---|---|---|---|
| low | 875 | −0.54 (−0.82, −0.30) | −0.51 (−0.75, −0.30) | −0.57 (−0.90, −0.29) |
| mid | 1,649 | −0.09 (−0.21, +0.05) | −0.14 (−0.26, −0.03) | −0.11 (−0.22, −0.01) |
| high | 1,476 | −0.22 (−0.40, −0.05) | −0.22 (−0.38, −0.05) | −0.21 (−0.38, −0.04) |

### Ablations (val, 3-seed ensembles, seeds 0-2)

| variant | − VWAP | − eFNN |
|---|---|---|
| eFNN (`antecedent_grad: all`) | −0.235 (−0.345, −0.140) | |
| eFNN, `value_only` (only the critic shapes the MFs) | −0.263 (−0.396, −0.145) | −0.028 (−0.093, +0.038) |
| eFNN, `none` (MFs move only by evolution) | −0.285 (−0.414, −0.176) | −0.051 (−0.115, +0.005) |
| hard k-means gate (one-hot φ, no evolution) | −0.268 (−0.372, −0.170) | −0.033 (−0.087, +0.030) |
| ANFIS (evolution off) | −0.270 (−0.375, −0.175) | −0.035 (−0.072, +0.003) |
| MLP | −0.237 (−0.340, −0.140) | −0.002 (−0.049, +0.049) |
| eFNN, **realised-drift reward** (M3 runs) | −0.004 (−0.197, +0.183) | **+0.231** (+0.056, +0.411) |
| MLP, **realised-drift reward** (M3 runs) | +0.006 (−0.213, +0.223) | **+0.241** (+0.041, +0.455) |

No architecture ablation is distinguishable from the full eFNN (every CI covers 0). The reward is the only
factor whose effect is clear.

### Structure evolution (eFNN, new reward)

| run | add / merge / prune (all accepted) | max edit KL | final R |
|---|---|---|---|
| eFNN s0 / s1 / s2 | 6/3/3, 15/5/10, 8/3/2 | 0.006 | 4 / 4 / 7 |
| `value_only` s0 / s1 / s2 | 13/7/5, 10/7/2, 8/3/2 | 0.005 | 5 / 5 / 7 |
| `none` s0 / s1 / s2 | 13/7/6, 10/4/3, 16/9/8 | 0.003 | 4 / 7 / 3 |

Zero rejections. Every edit's KL is below 0.006, far under `target_kl` = 0.02.

### Robustness: impact parameters off the training range (val, 5-seed ensembles)

Training randomises η ∈ [0.25, 1], γ ∈ [0.1, 0.5], τ_res ∈ [2, 15] min; the default is η 0.5, γ 0.3, τ 5.

| setting | VWAP cost | MLP − VWAP | eFNN − VWAP | eFNN − MLP |
|---|---|---|---|---|
| η = 0.1 | 16.42 | −0.270 | −0.292 | −0.022 (−0.062, +0.014) |
| η = 1 | 18.10 | −0.200 | −0.191 | +0.009 (−0.032, +0.046) |
| η = 2 | 19.98 | −0.122 | −0.078 | **+0.045** (+0.001, +0.085) |
| η = 4 | 23.72 | +0.029 | +0.148 | **+0.119** (+0.066, +0.170) |
| γ = 0 (no transient impact) | 15.04 | +0.014 | +0.032 | +0.019 (−0.022, +0.056) |
| γ = 0.5 | 18.58 | −0.407 | −0.434 | −0.027 (−0.067, +0.009) |
| γ = 1 | 22.13 | −0.829 | −0.907 | **−0.077** (−0.117, −0.041) |
| τ_res = 1 min | 15.34 | −0.012 | +0.006 | +0.017 (−0.024, +0.054) |
| τ_res = 15 min | 20.57 | −0.619 | −0.681 | **−0.062** (−0.101, −0.027) |
| τ_res = 30 min | 23.69 | −0.849 | −0.944 | **−0.095** (−0.134, −0.059) |

- **The policies have learned to manage transient (Obizhaeva-Wang) impact.** The edge is 0 without it
  (γ = 0 or τ = 1 min) and grows to −0.85 to −0.94 bps as it gets stronger or more persistent.
- **Where transient impact dominates, eFNN beats MLP significantly**, by −0.06 to −0.10 bps (Wilcoxon
  p < 0.005), including outside the training range.
- **Under much larger *temporary* impact (η ≥ 2), eFNN degrades faster than MLP.** At η = 4, 4× the training
  maximum, both lose to VWAP (eFNN +0.15 bps).

### Learned rules (eFNN, all 5 seeds, firing on val)

| seed | R | dominant rule (share, pace) | calm rule (LOW RV) | stressed rule (HIGH spread-var / RV) | AMI vs prev-day label |
|---|---|---|---|---|---|
| 0 | 4 | LOW sv / MED RV (83%, 0.96) | LOW/LOW 1.05 | HIGH/HIGH 0.91 | 0.002 |
| 1 | 3 | MED / MED (88%, 0.92) | LOW/LOW 1.02 (WITH imbalance) | | 0.001 |
| 2 | 7 | LOW / MED (65%, 0.90) | LOW/LOW 1.01 | HIGH/HIGH 0.83, HIGH/MED 0.90 | 0.009 |
| 3 | 3 | LOW / MED (93%, 0.90) | LOW/LOW 1.00 | | 0.001 |
| 4 | 4 | LOW / MED (84%, 0.94) | MED/LOW 0.97 | | 0.000 |

- **The direction matches the theory in every seed:** calm (LOW RV) rules trade at or ahead of the schedule
  (pace 1.00 to 1.05), and stressed rules slow down (0.83 to 0.91). The dominant rule trades slightly behind
  the schedule (0.90 to 0.96). With transient impact, that spreads the order more evenly against the
  displacement than VWAP does.
- **AMI ≈ 0:** the learned partition has nothing to do with the previous-day RV tercile. The rules key on
  intraday 30-min spread variance and RV, which a daily label does not capture.

### Leave-one-stock-out (2 seeds per fold; MLP and eFNN)

Each fold trains (and fits the observation scaler) on two names and is evaluated on the third, which was never
seen. Its train-period and val-period days are both out of sample, so both are reported.

```bash
uv run hftf train ppo --run m4loso_ADE_efnn_s0 --policy fuzzy --seed 0 --hold-out "ADE IS Equity"
uv run hftf eval compare mlp='runs/m4loso_ADE_mlp_s*/best.pt' efnn='runs/m4loso_ADE_efnn_s*/best.pt' \
    --security "ADE IS Equity" --split val --vs mlp --no-members
```

| held out | days | MLP − VWAP | eFNN − VWAP | eFNN − MLP |
|---|---|---|---|---|
| ADE | Jun-Aug (60) | −0.303 (−0.473, −0.138) | −0.202 (−0.396, −0.025) | +0.101 (−0.010, +0.219) |
| ADE | Sep (20) | −0.365 (−0.533, −0.201) | −0.483 (−0.731, −0.253) | −0.119 (−0.315, +0.040) |
| ADSEZ | Jun-Aug | −0.211 (−0.296, −0.125) | −0.172 (−0.242, −0.103) | +0.039 (−0.046, +0.116) |
| ADSEZ | Sep | −0.245 (−0.369, −0.128) | −0.217 (−0.342, −0.111) | +0.028 (−0.048, +0.094) |
| APHS | Jun-Aug | −0.182 (−0.261, −0.108) | −0.214 (−0.311, −0.125) | −0.032 (−0.071, +0.006) |
| APHS | Sep | −0.145 (−0.235, −0.065) | −0.191 (−0.315, −0.079) | −0.046 (−0.157, +0.047) |

On a stock it has never seen, each policy beats VWAP in all 6 cells (every CI excludes 0). eFNN vs MLP goes
both ways and is never significant.

### Final test split (Oct 1-19, 36 sessions; run once, after every choice above was fixed)

`hftf eval compare ... --split test --vs mlp --by-regime` (5-seed ensembles; the M3 realised-reward runs for
comparison). Objective − reference, bps, 95% session-cluster CI, Wilcoxon p on session means:

| policy | − VWAP | − TWAP | − Almgren-Chriss | − MLP |
|---|---|---|---|---|
| PPO-MLP | **−0.160** (−0.263, −0.060), p 0.005 | −0.215 (−0.321, −0.115) | −0.241 (−0.365, −0.123) | |
| PPO-ANFIS | **−0.244** (−0.375, −0.129), p 0.001 | −0.299 (−0.445, −0.169) | −0.324 (−0.446, −0.206) | −0.083 (−0.174, −0.006), p 0.12 |
| PPO-eFNN | **−0.222** (−0.334, −0.121), p 0.001 | −0.277 (−0.391, −0.171) | −0.302 (−0.425, −0.188) | −0.062 (−0.129, −0.001), p 0.23 |
| MLP, realised reward (M3) | +0.256 (−0.166, +0.736) | | | +0.417 (−0.012, +0.901) |
| eFNN, realised reward (M3) | +0.170 (−0.242, +0.631) | | | +0.330 (−0.086, +0.798) |

Test VWAP costs 16.43 bps (IS 4.2 + fees 12.0 + risk). Single seeds range from −0.09 to −0.29 (MLP / eFNN)
and −0.14 to −0.43 (ANFIS); 10 of 15 have CIs that exclude 0 (MLP 2/5, ANFIS 3/5, eFNN 5/5). By eval label (test has no "low" days):
after high-RV days, MLP −0.12, ANFIS −0.18, eFNN −0.18; after mid-RV days, −0.43 / −0.65 / −0.48.

### Reading (honest)

- **The result that holds everywhere is the reward design.** Trained on the expected-cost reward, PPO beats
  VWAP, TWAP and Almgren-Chriss on unseen days (val and test) and on unseen stocks (LOSO), by about 0.15 to
  0.25 bps. Every ensemble CI excludes 0. Trained on the realised-drift reward of M2/M3, the same networks
  fit the train paths and are no better than VWAP out of sample.
- **The gain is small in absolute terms.** It is 1-2% of the total cost (~16-17 bps, of which ~12 bps is
  fees, mostly STT) and ~5% of the IS. It comes from transient-impact management in the simulator. It is
  therefore conditional on the impact model, and the robustness sweep shows how it scales with the impact
  parameters. It is not a claim about live savings.
- **Fuzzy vs MLP:** at default impact the architectures are indistinguishable on val (±0.05). On test, both
  fuzzy ensembles are ahead of MLP by 0.06-0.08 bps. The cluster CIs just exclude 0 but Wilcoxon does not
  (p 0.12 / 0.23), so this is suggestive, not established. The clearest architecture effect is the
  robustness sweep: eFNN is significantly better than MLP when transient impact is stronger or more
  persistent than in training (−0.06 to −0.10 bps), and significantly worse under much larger temporary impact.
- **Evolution vs static fuzzy:** eFNN and ANFIS cannot be told apart anywhere (val −0.035 ± 0.04 in ANFIS's
  favour; test within 0.03). Evolution adapts the structure stably (0 rejected edits, max edit KL 0.006), but
  on 3 stocks and 3 regime inputs, 3-4 k-means rules are already enough.
- **Interpretability holds up:** in every seed, calm-regime rules pace at or above the schedule and
  stressed-regime rules slow down, as the theory predicts. One dominant rule covers 65-93% of decisions, and
  the partition is unrelated to the previous-day volatility label (AMI ≈ 0).
- **Limitations:** 3 stocks and 60 training sessions; a parametric impact model with no depth data; and a
  test split of 36 stock-sessions, which can resolve differences of ~0.1 bps between policies but not
  smaller ones.

## M5 (part 1): TickSim re-evaluation of the M4 policies (2026-10-29)

TickSim (ARCHITECTURE §4.8) replays each child against the real 1-second L1 state: it fills at the touch
1 s after the bar boundary, shares within the displayed depth pay the half spread only, and the excess
walks the book with BarSim's square-root term. Ticks exist only on the **val tick sessions** (Sep 21-30,
8 per stock = 24) and the test sessions. **The test split is not used here.** Policies are the unchanged
M4 checkpoints (trained in BarSim). BarSim and TickSim run on **identical** episodes (seed 1, 4,000), so
every difference below is paired.

```bash
uv run hftf data build-l1                                   # data/processed/l1_1s_v1.parquet (13 s)
uv run hftf eval compare mlp='runs/m4_mlp_s*/best.pt' anfis='runs/m4_anfis_s*/best.pt' \
    efnn='runs/m4_efnn_s*/best.pt' --split val --vs mlp --sim tick           # TickSim
#   ... --tick-sessions (instead of --sim tick)                               # BarSim, same episodes
```

### What TickSim changes (VWAP, val tick sessions, seed 0; session-cluster 95% CI)

| switch | VWAP cost − BarSim |
|---|---|
| fill at the replayed tick mid/spread at bar start + 1 s, no depth | −0.004 (−0.047, +0.035) |
| + half spread only within displayed depth (**TickSim default**) | **−0.322** (−0.367, −0.278) |
| latency 0 s / 5 s instead of 1 s | −0.317 / −0.321 |
| 4 orders per decision (each pays the ₹20 per-order fee) | +2.807 (+2.507, +3.120); fees +3.17 |

BarSim's end-of-bar fill is unbiased against the replayed book (timing −0.004, n.s.), and the whole gap
comes from displayed depth. Splitting decisions into sub-minute orders does not pay under the per-order fee.

### Policies: objective − VWAP (bps), 5-seed ensembles, 24 val tick sessions

| policy | BarSim | TickSim | own TickSim − BarSim | − MLP (TickSim) |
|---|---|---|---|---|
| VWAP (cost) | 16.60 | 16.28 | −0.325 (−0.378, −0.274) | |
| PPO-MLP | −0.107 (−0.227, +0.014), p 0.065 | **−0.166** (−0.292, −0.041), p 0.012 | −0.384 | |
| PPO-ANFIS | −0.149 (−0.269, −0.029), p 0.023 | **−0.195** (−0.323, −0.069), p 0.005 | −0.370 | −0.029 (−0.071, +0.013) |
| PPO-eFNN | −0.123 (−0.261, +0.007), p 0.046 | **−0.180** (−0.324, −0.048), p 0.004 | −0.382 | −0.014 (−0.061, +0.028) |

Single seeds in TickSim: MLP −0.04 to −0.22, ANFIS −0.12 to −0.28, eFNN −0.12 to −0.28 (every one < 0).

### Reading

- **The M4 edge survives the higher-fidelity fill model and grows by about 0.05 bps.** Every policy gains
  more from real depth than VWAP does (−0.37 to −0.38 vs −0.33). The policies pace slightly behind the
  schedule (M4 rules: 0.90-0.96), so their children are smaller and more of each one fits inside the touch.
  In TickSim every ensemble's CI excludes 0.
- On these 24 sessions (the last third of val), the BarSim edge is smaller than on the full val split
  (−0.11 to −0.15 vs −0.24) and the MLP CI covers 0. This is a smaller, later sample, not a TickSim effect:
  both columns use the same episodes.
- Fuzzy vs MLP is still indistinguishable (±0.05 bps).
- Limits: depth beyond L1 is still parametric, and the replay is counterfactual (the book does not react
  to us except through the transient displacement). The 1-second grid comes from ~1 s conflated snapshots.

## M5 (part 2): limit orders (2026-10-29)

Design: ARCHITECTURE §4.9. Each decision posts a passive fraction `f` of the child at the near touch; the
rest goes aggressively, and unfilled passive shares return to the inventory at the end of the bar.
Policies train in **BarSim** with a bar-range fill rule and are evaluated in **TickSim** with a queue model.
Both fill models cap a fill at the volume that traded at or through our price. BarSim's two caps were
fitted to the queue model on the val tick sessions: θ_touch = 0.0656, θ_through = 0.2942. Fill rates are
48.7% (bar) vs 47.1% (queue) of posted shares over 68k synthetic orders. Runs: 3 seeds × {MLP, eFNN}, same
PPO config as M4, `init_passive` 0.2. Evaluation: val tick sessions (24), seed 1, 4,000 episodes,
identical in both simulators; pace-only references are the M4 seeds 0-2 ensembles.

```bash
uv run hftf eval calibrate-limit --split val
uv run hftf train ppo --run m5_lim_efnn_s0 --policy fuzzy --seed 0 --limit     # --policy mlp
```

### What did not work first (kept for the report)

The first calibration let any print **through** our price fill the whole order, however large. Under
that model a naive "always post the whole child passively" VWAP beat VWAP by −3.0 bps in TickSim, which is
an artefact: a 5-share print through the bid filled 400 shares. Those runs (`m5_limit_*`) are discarded.
With the volume cap, the queue model fills 47% of posted shares instead of 72%.

### Results (objective − reference, bps; 95% session-cluster CI; Wilcoxon p)

| policy (val tick sessions) | cost BarSim | cost TickSim | − VWAP, TickSim | passive share | CVaR95 IS (TickSim) |
|---|---|---|---|---|---|
| VWAP | 16.60 | 16.28 | | 0 | 68.6 |
| VWAP, 100% passive (naive) | 13.93 | 14.39 | **−1.895** (−2.294, −1.473), p 0.000 | 82% | 75.0 |
| PPO-MLP (pace only) | 16.46 | 16.07 | −0.207 (−0.357, −0.069), p 0.003 | 0 | 67.4 |
| PPO-eFNN (pace only) | 16.47 | 16.08 | −0.200 (−0.354, −0.065), p 0.005 | 0 | 66.8 |
| PPO-MLP + limit | 15.69 | 15.89 | −0.391 (−0.669, −0.112), p 0.013 | 58% | 77.8 |
| PPO-eFNN + limit | 15.69 | 15.89 | −0.391 (−0.684, −0.085), p 0.018 | 56% | 79.6 |

| paired comparison, TickSim | diff |
|---|---|
| limit-MLP − MLP | −0.184 (−0.500, +0.154), p 0.121 |
| limit-eFNN − eFNN | −0.191 (−0.520, +0.173), p 0.095 |
| limit-eFNN − limit-MLP | −0.000 (−0.105, +0.098) |
| limit policies − naive 100% passive | +1.504 (+1.19, +1.80) |

In BarSim (the training simulator) the limit policies are −0.91 vs VWAP and −0.77 vs their pace-only versions.

### Reading (honest)

- **The learned limit policies beat VWAP in TickSim (−0.39 bps) but are not significantly better than the
  pace-only policies, and they lose to the naive always-passive rule by 1.5 bps.**
- **Cause: the action parameterisation, not the network.** Both architectures settle at a passive fraction
  of about 0.56-0.6, where every child is split into a passive and an aggressive order. Each order pays the
  ₹20 fixed fee, so fees rise by +0.6 to +0.7 bps over VWAP, while always-passive *lowers* fees (fewer
  orders). A Beta-distributed fraction never reaches exactly 0 or 1, so the policy cannot learn "all
  passive" without paying the second fee: the reward has a cliff at f = 1 that the continuous head cannot
  reach. **Next step:** a dead-zone (f > 0.9 → all passive, f < 0.1 → none) or a discrete market/limit head.
- **Passive edges are fragile to the fill model:** the limit policies lose 0.5 bps of their edge from
  BarSim to TickSim (the pace edge did not lose any; part 1). Passive fills also widen the tail: CVaR95
  of IS is 75-80 vs 67-69, because unfilled shares wait while the price runs away.
- **Model caveats:** our resting order does not change the historical prints (no queue displacement of
  other traders), and conflated ~1 s snapshots make the queue position approximate. The always-passive
  advantage (−1.9 bps) is therefore an upper bound on what passive execution is worth here, not an estimate.
