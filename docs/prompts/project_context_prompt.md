# Prompt: generate `project_context.md`

Paste everything below the line into Claude Code at the repo root. Re-run it after major changes
(new data pull, architecture revision). It regenerates the file from the repo's current state.

---

You are writing `project_context.md` at the repository root. It is the **single onboarding document**
for anyone, human or AI agent, who works on this repo. Every later Claude Code session will load it
before touching code, so it must hold the domain knowledge, intuition and metadata needed to make
correct decisions without re-deriving them.

## Project vision (ground truth: do not change it, only explain it)
**Market regime-aware optimal execution with Fuzzy PPO.** A Proximal Policy Optimization agent embeds an
evolving Fuzzy Neural Network (eFNN). Order-book imbalance, bid-ask spread variance and realized
volatility feed an evolving fuzzy layer, which generates, merges or prunes fuzzy rules as market regimes
shift (e.g. "High Volatility / Low Liquidity", "Trend Bullish"). The agent learns the child-order slice
size (a pace relative to a TWAP/VWAP schedule) that minimises slippage, market impact and transaction
fees. The Bellman (TD/GAE value) loss and the PPO policy loss also adjust the fuzzy membership
functions (centres/widths), together with the actor-critic heads. Data: NSE equities from Bloomberg
(blpapi).

## Read these first. Your claims must agree with them
1. `docs/ARCHITECTURE.md`: the binding design (env, action, reward, eFNN, PPO, evaluation, layout). Copy
   numbers and decisions from it; don't invent alternatives. If you see a flaw, record it under "Open
   questions". Never silently diverge.
2. `docs/DATA_AUDIT.md` and `docs/data/data_audit.json`: what data exists, its quirks, and the fixes.
3. `packages/hftf-data/src/hftf_data/*.py`: the actual column names and semantics. Read
   `data/processed/features_1m_v3.parquet` with pandas (via `uv run python`) to list the real columns,
   dtypes and coverage. **Never document a column you have not seen in the file.**
4. `docs/DATA_TRANSFORMS.md`: every cleaning, imputation and transformation step (the collector that pulled the data is in git history at `de800d8`).
5. `pyproject.toml` and each `packages/*/pyproject.toml`: the workspace layout and dependencies.

## Required sections (in this order)
1. **TL;DR**: 5-8 lines: what we are building, why, and the current state.
2. **Glossary**: IS, arrival price, VWAP/TWAP/POV, participation rate, child/parent order, slippage,
   temporary/permanent/transient impact, L1, OFI, microprice, realized vol, spread variance, regime,
   membership function, firing strength, TSK, eFNN, PPO, GAE, Bellman error. One or two lines each,
   precise.
3. **Market microstructure: NSE specifics**: session times (pre-open auction, continuous, closing),
   tick-size price bands, lot sizes, price bands/circuit filters, the fee stack (STT, exchange charge,
   SEBI, stamp duty, GST, brokerage) and which components are proportional and which are per-order.
   Explain why per-child fixed fees are what penalise over-slicing. Include the intraday U-shape and the
   measured spreads/depth/ADV from DATA_AUDIT. Mark any regulatory number as "verify against current
   NSE/SEBI circulars".
4. **Optimal execution theory with intuition**: the IS decomposition (delay, impact, timing/opportunity
   risk, fees); Almgren-Chriss (the trade-off between impact cost and risk, λ, the closed-form trajectory
   and why it front-loads when risk-averse); the square-root impact law and what β ≈ 0.5 implies;
   Obizhaeva-Wang transient impact and resilience; VWAP as a benchmark versus a strategy. State what each
   regime input should do to the optimal pace (e.g. high spread variance → patience; high RV →
   urgency under risk aversion; buy-side imbalance → front-load buys before the price moves). These
   priors are hypotheses the agent should discover. Call them hypotheses.
5. **Fuzzy systems and the eFNN**: Gaussian MFs, product t-norm, normalised firing, first-order TSK; the
   evolving-systems lineage (eTS, ePL, DENFIS, PANFIS, GENEFIS) and which ideas we borrow (novelty-based
   rule addition, overlap-based merging, utility-based pruning). Explain exactly how `ARCHITECTURE.md §5`
   implements gradient flow from the Bellman/value loss into the MFs. Explain why structural edits happen
   only between PPO iterations, and what "function-preserving rule insertion" means and why it matters.
6. **RL formulation**: the MDP tuple as implemented: observation groups, the Beta pace action, the
   reward (and why it telescopes to the AC objective), episode sampling, γ = 1 with finite horizon, PPO
   hyperparameters, normalisation. Note the CPU-only constraint.
7. **Data**: (a) inventory table (dataset, instruments, date range, granularity); (b) **column
   dictionary for `features_1m_v3`**, grouped (identity/time, prices, spread, volatility, volume/profile,
   L1 microstructure, context, daily-lagged, data-quality, eval-only), with units and causality notes;
   (c) every raw-data quirk with its fix (condition codes, conflation, zero pre-open quotes, sentinel,
   calendar gaps, off-book days); (d) the imbalance proxy and its measured correlation; (e) splits by date
   with session counts computed from the data.
8. **Invariants and conventions**: causality (`decision_time = time + 1 min`, no same-day daily data,
   train-only normalisation stats); UTC storage / IST reasoning; prices in ₹ and costs in bps of arrival
   notional; side-signed formulas; seeds; where code goes in the monorepo; "notebooks never hold logic";
   licensed Bloomberg data is never committed.
9. **Evaluation protocol**: baselines, ablations, metrics, paired statistics, robustness sweeps, and what
   counts as a win.
10. **Known limitations and pitfalls**: small universe; conflated ticks; no L2 depth; counterfactual
    impact; rolling Bloomberg intraday windows; the classic look-ahead traps for this specific dataset;
    PPO + changing architecture.
11. **Current status and next steps**: milestone table from ARCHITECTURE §9 with real status, plus the
    data actions from DATA_AUDIT.
12. **Open questions**: decisions not yet settled, each with a recommended default.
13. **References**: Almgren & Chriss (2000); Obizhaeva & Wang (2013); Cont, Kukanov & Stoikov (2014);
    Bouchaud et al. on the square-root law; Schulman et al. PPO (2017) and GAE (2016); Angelov & Filev
    eTS (2004); Lughofer (Evolving Fuzzy Systems); Pratama et al. PANFIS (2014); Takagi & Sugeno (1985);
    Jang ANFIS (1993); Nevmyvaka, Feng & Kearns (2006) RL for execution; Ning, Lin & Jaimungal (2021)
    double deep Q-learning for execution. Cite only works you are confident exist. Don't fabricate
    titles, venues or years. If unsure, omit.

## Style rules
- Write for a strong ML engineer who is new to market microstructure. Give intuition first, then the
  formula. Use LaTeX-free plain-text math (`σ`, `√`, `Σ`) so it reads well in a terminal.
- Prefer tables for metadata. Every number must come from the repo/data or be marked as an assumption.
- Be concise but complete: about 600-900 lines. No marketing tone. No emojis.
- End with a metadata block: generated date, git commit (if any), data snapshot date, and the source
  files you read.

## Verification before you finish
- Re-open the parquet and confirm that every column name you documented exists.
- Confirm that each number you quote from DATA_AUDIT matches `docs/data/data_audit.json` or a quick
  recomputation.
- Confirm that the session counts per split come from the data, not from guesswork.
- List any discrepancy you found between ARCHITECTURE.md and the code under "Open questions".
