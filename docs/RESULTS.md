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

**Exit criterion met:** PPO-MLP beats VWAP on train in 3/3 seeds, with every CI excluding 0. The val gap is
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
