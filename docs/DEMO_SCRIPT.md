# Demo script (3 minutes)

A 3-minute walkthrough of the web app. Every number on screen comes from
[RESULTS.md](RESULTS.md); report figures are in [figures/](figures/README.md).

## Before you present

1. Start the API and the app (the data is licensed, so everything runs on localhost):
   ```bash
   uv run hftf serve                  # terminal 1, repo root: http://127.0.0.1:8000
   cd web && npm run build && npm run preview   # terminal 2: http://127.0.0.1:4173
   ```
2. Open the order below once so it is cached, then go back to the Overview:
   `http://127.0.0.1:4173/execute?sec=APHS+IS+Equity&date=2026-09-29&start=12%3A42&h=60&side=sell&p=0.1021&sim=tick&pol=efnn_s0%2Cmlp%2Cefnn_limit_s0%2Calmgren_chriss%2Cvwap_passive100%2Cvwap`
3. On a projector, press `P` (presentation mode: light theme, larger type). Keys: `1`-`5` screens,
   `Space` play/pause, `←` `→` step, `Ctrl K` command palette.

## The walkthrough

| Time | Screen | Do | Say |
|---|---|---|---|
| 0:00 | **Overview** (`1`) | Point at the three numbers | "An execution agent you can read. It slices a large order through the day, beats VWAP out of sample, and every decision traces back to a fuzzy rule." |
| 0:20 | **Execute** (`2`), the cached APHS order | `Space` to play at 4× | "Sell 4,288 shares of APHS over an hour, on a day we replay against the real order book. Six policies, same market, same order." |
| 0:40 | Execute, mid-replay | Point at the cost race, then the right panel | "Below zero is cheaper than VWAP. On the right, the eFNN's reasoning: the rule firing now, as a sentence, and where the market sits on each fuzzy set." |
| 1:05 | Execute, end of replay | Point at the dashed orange line spiking | "The naive rule posts everything passively. It looks cheap until the end, when its unfilled shares are forced through: +7.7 bps. The learned limit-order policy paces faster and avoids that." |
| 1:25 | **Brain** (`3`) | Drag to orbit, hover a rule | "The eFNN sees three inputs, so its rules are literally regions of a 3D space. Blue rules slow down, orange speed up. The line is the market state along an order." |
| 1:50 | **Evidence** (`4`) | Toggle *Both* | "Out of sample, every ensemble beats VWAP, and an independent retraining by a teammate reproduces it. We correct for multiple comparisons and cluster by session and by date." |
| 2:15 | Evidence, *Architectures* | Point at the label "Not established" | "What we do not claim: the fuzzy network does not beat the plain network. Its value is interpretability at equal performance." |
| 2:30 | Evidence, *Robustness* | Point at γ = 0 | "The edge is transient-impact management: zero without it, growing with it." |
| 2:45 | **Training** (`5`) | Point at the rule events | "Rules are added for new regimes, then merged and pruned, and every structural edit stays far below the KL guard." |
| 3:00 | | | "All of this is in a simulator calibrated to Bloomberg data, so it is a research result, not a claim about live trading." |

## If something goes wrong

- **"API offline" in the top bar:** terminal 1 is not running, or not on port 8000.
- **No WebGL (the Brain screen shows 2D projections):** that is the built-in fallback; the story is the same.
- **A slow first order:** the first request loads the checkpoints; the cached order above avoids it.
- **No projector time for the app:** use the figures in [figures/](figures/README.md); each one has a caption and its source.
