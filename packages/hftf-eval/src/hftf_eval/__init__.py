"""hftf_eval: baselines, backtests and statistics (M1, M4).

Built (M1): baselines (immediate, TWAP, VWAP, POV, Almgren-Chriss), backtest (paired runner), metrics
(IS, CVaR, VWAP slippage). Built (M4, docs/ARCHITECTURE.md §7): stats (session-cluster bootstrap CI, Wilcoxon
on session means, AMI) and report (shared-episode backtests, regime breakdowns, impact-robustness sweeps).
"""
