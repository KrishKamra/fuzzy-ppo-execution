"""hftf_sim: execution environment (M1).

Planned modules (docs/ARCHITECTURE.md §4): episodes (EpisodeSampler), impact (square-root + transient
ImpactModel), fees (NSE FeeModel from configs/fees_nse.yaml), schedules (TWAP/VWAP profiles),
env (BatchExecutionEnv + gymnasium wrapper).
"""
