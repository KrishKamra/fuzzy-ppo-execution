# Results log

Results as they are produced, with the commands that regenerate them. Costs are in **bps of arrival
notional** (higher is worse). All comparisons are **paired**: every policy runs on identical episodes with
identical impact. The test split stays untouched until M4.

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
