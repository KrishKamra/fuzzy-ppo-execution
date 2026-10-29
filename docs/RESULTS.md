# Results log

> Publication-ready figures of these results (SVG/PNG, with captions and sources) are in [figures/](figures/README.md);
> regenerate them with `uv run hftf report figures`.

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

## M4 replication and stricter statistics (2026-10-29)

An independent replication of M4 by a team member (Dhanya Hariya): the same code (commit `799cbcd`)
retrained from scratch, 3 seeds per architecture, evaluated on the test split with a different episode seed
(42, 4,000 episodes). It is a **second use of the test split**, as a replication: nothing was selected on it.
Their per-episode results were re-analysed here with the same code as ours. Their analysis also brought
three checks that are now part of `hftf_eval.stats` and `hftf eval compare`:
- **Holm-Bonferroni** adjustment over the family of claims in a table (the seed ensembles);
- **date-cluster** CIs and Wilcoxon tests. The three stocks share market-wide moves on a date, so resampling
  whole dates is more conservative than resampling stock-sessions. With 12 test dates it is a sensitivity
  check, not the headline;
- the **unweighted session-mean t-interval**, the estimand the session Wilcoxon test is about.

```bash
uv run hftf eval compare mlp='runs/m4_mlp_s*/best.pt' anfis='runs/m4_anfis_s*/best.pt' \
    efnn='runs/m4_efnn_s*/best.pt' --split test --vs mlp --by security,horizon
```

### Test split: ensemble − VWAP (bps); session-cluster CI, Holm-adjusted session Wilcoxon p, date-cluster CI

| policy | ours: 5 seeds, episode seed 1 | Holm p | date CI | replication: 3 seeds, episode seed 42 | Holm p | date CI |
|---|---|---|---|---|---|---|
| PPO-MLP | −0.160 (−0.263, −0.060) | 0.005 | (−0.230, −0.091) | −0.136 (−0.271, −0.007) | 0.074 | (−0.222, −0.053) |
| PPO-ANFIS | −0.244 (−0.375, −0.129) | 0.002 | (−0.381, −0.113) | −0.195 (−0.354, −0.049) | 0.059 | (−0.316, −0.082) |
| PPO-eFNN | −0.222 (−0.334, −0.121) | 0.002 | (−0.314, −0.138) | −0.237 (−0.436, −0.067) | **0.018** | (−0.371, −0.119) |

### Architectures (Holm over the three pairs)

| pair | ours | Holm p | replication | Holm p |
|---|---|---|---|---|
| eFNN − MLP | −0.062 (−0.129, −0.001) | 0.453 | −0.101 (−0.212, −0.009) | 0.324 |
| ANFIS − MLP | −0.083 (−0.174, −0.006) | 0.357 | −0.060 (−0.154, +0.024) | 1.000 |
| eFNN − ANFIS | +0.022 (−0.050, +0.094) | 0.834 | −0.041 (−0.154, +0.050) | 1.000 |

### By security (ensemble − VWAP, bps)

| security | ours MLP / ANFIS / eFNN | replication MLP / ANFIS / eFNN |
|---|---|---|
| ADE | −0.19 / −0.30 / −0.32 | −0.34 / −0.50 / −0.58 |
| ADSEZ | −0.22 / −0.33 / −0.23 | −0.05 / −0.10 / −0.06 |
| APHS | −0.08 / −0.11 / −0.12 | −0.01 / +0.01 / −0.07 |

### Reading

- **The M4 result replicates.** Retrained independently with fewer seeds and evaluated on different
  episodes, every ensemble beats VWAP by 0.14-0.24 bps, and every date-cluster CI excludes 0 in both runs.
  With 5 seeds, all three survive Holm; with 3, eFNN does (0.018) and ANFIS/MLP are borderline (0.059/0.074).
  More seeds per ensemble buys significance, as M4 argued.
- **Architecture differences are not established.** After Holm, every pairwise p is ≥ 0.32 in both runs.
  This replaces the M4 wording that the fuzzy ensembles were "suggestively" ahead of MLP on test. The
  replication also notes that eFNN's seed-to-seed spread (0.116 bps) exceeds the eFNN − MLP gap (0.101), and
  that non-significance is not equivalence (that would need an equivalence test with a prespecified margin).
- **No single stock drives the result.** The replication's report attributed the edge to ADE; in our run
  ADSEZ contributes as much. What holds in both is that **APHS gains least**: it has the highest price, the
  lowest ADV and the smallest touch (74 shares). A causal story about which stocks benefit needs more than
  12 test dates.
- Corrections to the replication's written report (its numbers are right): horizons are {30, 60, 120} min,
  not {15, 30, 60}; `Q` is 1-20% of expected horizon volume, not of ADV; η is the *temporary* impact; VWAP
  crosses the spread every minute, so it is not a passive benchmark; and its Holm step-down lacked the
  running maximum (eFNN − ANFIS is 1.000, not 0.681).

## M5 (part 3): the passive-fraction dead zone (2026-10-29)

Fix for the fee cliff of part 2: fractions ≤ 0.1 snap to 0 and ≥ 0.9 to 1 (`limit.dead_zone`), so a child can
be one order of either kind. Retrained 3 seeds × {MLP, eFNN} (`m5_ldz_*`), otherwise unchanged. Same val tick
episodes as part 2; the part-2 runs are re-evaluated without the dead zone, as they were trained.

```bash
uv run hftf train ppo --run m5_ldz_mlp_s0 --policy mlp --seed 0 --limit     # dead zone from env.yaml
```

| TickSim, val tick sessions | cost | − VWAP | fees | passive share | CVaR95 IS |
|---|---|---|---|---|---|
| VWAP | 16.28 | | 12.18 | 0 | 68.6 |
| VWAP, 100% passive (naive) | 14.39 | −1.895 (−2.294, −1.473) | 11.92 | 82% | 75.0 |
| PPO-MLP / PPO-eFNN (pace only) | 16.07 / 16.08 | −0.207 / −0.200 | 12.18 | 0 | 67.4 / 66.8 |
| limit-MLP, no dead zone (part 2) | 15.89 | −0.391 (−0.669, −0.112) | 12.85 | 58% | 77.8 |
| **limit-MLP, dead zone** | **15.57** | **−0.715** (−1.032, −0.390) | 12.51 | 53% | 81.4 |
| limit-eFNN, no dead zone (part 2) | 15.89 | −0.391 (−0.684, −0.085) | 12.78 | 56% | 79.6 |
| limit-eFNN, dead zone | 15.80 | −0.476 (−0.772, −0.188) | 12.55 | 43% | 78.0 |

| paired, TickSim | diff (session CI) | date CI |
|---|---|---|
| limit-MLP: dead zone − none | **−0.324** (−0.484, −0.149) | (−0.517, −0.133) |
| limit-eFNN: dead zone − none | −0.085 (−0.215, +0.036) | (−0.202, +0.011) |
| limit-MLP (dead zone) − pace-only MLP | **−0.508** (−0.830, −0.161), p 0.005 | (−0.911, −0.125) |
| limit-eFNN (dead zone) − pace-only eFNN | −0.277 (−0.584, +0.067), p 0.042 | (−0.645, +0.088) |
| limit-eFNN − limit-MLP | +0.238 (+0.082, +0.389) | (+0.092, +0.384) |
| limit-MLP − naive 100% passive | +1.180 (+0.845, +1.488) | (+0.750, +1.601) |

In BarSim (the training simulator) the dead-zone limit-MLP is −1.22 vs VWAP, and the naive rule −2.67.

### Reading

- **The dead zone works for the MLP:** −0.32 bps from fewer double-fee children (fees 12.85 → 12.51). Its
  limit policy is now **significantly better than its pace-only policy** (−0.51 bps, both CIs exclude 0),
  so learned limit orders now add value over pacing alone.
- **It barely helps the eFNN** (−0.09, n.s.), which now trails the limit-MLP by 0.24 bps. With the larger
  action head, the fuzzy policy's rules move the passive fraction less (43% passive vs 53%).
- **Both still lose to the naive always-passive rule by 1.2-1.4 bps, and the gap is as large in BarSim.**
  This is an optimisation problem, not a simulator mismatch: the policies keep mixing order types, so
  fees stay above VWAP's while always-passive lowers them. **Next step:** start the passive head at the naive
  rule (`init_passive` ≈ 0.95), as the pace head starts at VWAP, so PPO improves on it instead of having
  to find it. Learned limit orders also widen the tail (CVaR95 78-81 vs 67-69) because unfilled shares wait
  while the price moves.

## M5 (part 4): starting the passive head at the naive rule (2026-10-29)

Part 3 left an optimisation gap: the learned limit policies lost to the naive always-passive VWAP by 1.2-1.4
bps in both simulators. As the pace head starts at VWAP (a = 1), the passive head now starts at the naive rule:
mean 0.95 with its own concentration of 30 (`--init-passive 0.95 --passive-concentration 30`), so almost every
initial child is a single passive order (dead zone ≥ 0.9). PPO then improves on the rule instead of having to
find it. 3 seeds × {MLP, eFNN} (`m5_lip_*`); same val tick episodes as parts 2-3.

```bash
uv run hftf train ppo --run m5_lip_efnn_s0 --policy fuzzy --seed 0 --limit --init-passive 0.95 \
    --passive-concentration 30
```

| val tick sessions | BarSim cost | TickSim cost | − VWAP, TickSim | fees | passive share | CVaR95 IS (TickSim) |
|---|---|---|---|---|---|---|
| VWAP | 16.60 | 16.28 | | 12.18 | 0 | 68.6 |
| naive always-passive VWAP | 13.93 | 14.39 | −1.895 (−2.294, −1.473) | 11.92 | 82% | 75.0 |
| pace-only MLP / eFNN | 16.46 / 16.47 | 16.07 / 16.08 | −0.207 / −0.200 | 12.18 | 0 | 67.4 / 66.8 |
| limit-MLP / eFNN, dead zone (part 3) | 15.39 / 15.71 | 15.57 / 15.80 | −0.715 / −0.476 | 12.51 / 12.55 | 53% / 43% | 81.4 / 78.0 |
| **limit-MLP, init at naive rule** | **13.51** | **13.95** | **−2.326** (−2.706, −1.952) | 11.93 | 91% | **69.1** |
| **limit-eFNN, init at naive rule** | **13.48** | **13.92** | **−2.358** (−2.734, −1.997) | 11.93 | 91% | **69.0** |

| paired, TickSim | diff (session CI) | date CI |
|---|---|---|
| limit-MLP − naive always-passive | **−0.431** (−0.605, −0.278) | (−0.614, −0.254) |
| limit-eFNN − naive always-passive | **−0.463** (−0.635, −0.307) | (−0.642, −0.289) |
| limit-eFNN − limit-MLP | −0.032 (−0.093, +0.028) | (−0.092, +0.032) |
| limit-MLP − pace-only MLP | −2.120 (−2.554, −1.672) | (−2.666, −1.630) |

In BarSim the gains over the naive rule are −0.419 (MLP) and −0.451 (eFNN): the same as in TickSim.

### Reading

- **The learned limit policies now beat the naive rule** by 0.43-0.46 bps, with both CIs excluding 0 and
  the same gain in BarSim and TickSim. The fill-model change does not create the improvement.
- **They also fix the naive rule's tail.** CVaR95 of IS is 69 against 75 for always-passive, close to VWAP's
  68.6. The policies post nearly everything passively (91%) but pace at about 1.6× the VWAP schedule, so
  shares that do not fill are not left waiting while the price runs away. That is the cost-risk trade-off
  the agent was meant to learn.
- **eFNN = MLP here too** (−0.03, n.s.), as for pacing alone.
- **Initialisation mattered more than architecture.** From a passive mean of 0.2, PPO settled in a local
  optimum of mixed orders (parts 2-3); from the naive rule it improves on it in every seed (final mean
  passive 0.95-0.96, pace 1.56-1.67).
- **Caveat (as part 2):** passive fills are counterfactual. Our resting order is filled from historical flow
  that did not know about it, and does not displace other traders in the queue. The absolute passive gains
  (−2.3 bps) are an upper bound. The comparison with the naive rule shares that bias, so the −0.4 bps the
  agent adds on top of it is the more defensible number.
