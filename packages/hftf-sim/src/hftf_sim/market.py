"""Session arrays the simulator reads, built once from the final feature table.

Row `j` of a session is the bar starting at 09:15 + j min, observable at its `decision_time` (j + 1).
The env observes row `j` when deciding at minute j + 1 and fills the child in bar j + 1, so
observation columns and fill columns are kept apart here (ARCHITECTURE §4.2, §4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hftf_data.calendar import SESSION_MINUTES
from hftf_data.paths import REPO_ROOT

SPLITS = ("train", "val", "test")
FIRST_DECISION_MSO = 4  # row observed at 09:20 (configs/data_readiness.yaml first_decision_mso)
OBS_CLIP = 5.0

# Observation columns, standardised with train statistics; SIGNED ones are multiplied by side in the env.
REGIME_COLUMNS = ("imb_bar_proxy_ema5", "log_spread_var_30m", "log_rv_30m")
MARKET_COLUMNS = ("spread_bps", "dev_vwap_bps", "ret_5m", "ctx_nifty_ret_30m", "ctx_vix")
SIGNED_COLUMNS = ("imb_bar_proxy_ema5", "dev_vwap_bps", "ret_5m", "ctx_nifty_ret_30m")
# Fill-side columns: realised values the simulator may use but the agent never observes.
FILL_COLUMNS = ("mid", "spread_bps", "volume", "value", "rv_30m", "prof_avg_vol")
RV_WINDOW = 30


@dataclass
class MarketData:
    """Per-session arrays, shape (n_sessions, 360); `sessions` maps axis 0 to (security, date, split)."""

    sessions: pd.DataFrame
    mid: np.ndarray
    spread: np.ndarray  # fraction of mid
    volume: np.ndarray
    value: np.ndarray
    sigma: np.ndarray  # 1-min return vol, rv_30m / sqrt(30)
    prof: np.ndarray  # expected minute volume from the previous sessions' profile
    cum_prof: np.ndarray  # (n_sessions, 361): cum_prof[:, j] = sum of prof over rows < j
    obs: np.ndarray  # (n_sessions, 360, n_obs) standardised observation columns
    obs_columns: tuple[str, ...]
    stats: dict[str, tuple[float, float]]

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        securities: list[str] | None = None,
        stats: dict[str, tuple[float, float]] | None = None,
    ) -> MarketData:
        """`securities` keeps only those names (leave-one-stock-out); `stats` reuses a training scaler, else
        it is fitted on the train split of the names kept."""
        path = REPO_ROOT / "data/processed/features_1m_v3.parquet" if path is None else Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        cols = ["security", "date", "mso", "split", *REGIME_COLUMNS, *MARKET_COLUMNS, *FILL_COLUMNS]
        df = pd.read_parquet(path, columns=list(dict.fromkeys(cols)))
        if securities is not None:
            df = df[df["security"].isin(securities)]
            if df.empty:
                raise ValueError(f"no rows for securities {securities}")
        return cls.from_frame(df, stats=stats)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, stats: dict[str, tuple[float, float]] | None = None) -> MarketData:
        df = df[df["split"].isin(SPLITS)].sort_values(["security", "date", "mso"])
        n = df.groupby(["security", "date"], sort=False)["mso"].transform("size")
        df = df[n == SESSION_MINUTES]
        obs_columns = (*REGIME_COLUMNS, *MARKET_COLUMNS)
        if stats is None:
            train = df[(df["split"] == "train") & (df["mso"] >= FIRST_DECISION_MSO)]
            stats = {c: (float(train[c].mean()), float(train[c].std())) for c in obs_columns}
        sessions = df.drop_duplicates(["security", "date"])[["security", "date", "split"]]
        sessions = sessions.reset_index(drop=True)

        def grid(col: str) -> np.ndarray:
            return df[col].to_numpy(dtype=np.float64).reshape(len(sessions), SESSION_MINUTES)

        prof = grid("prof_avg_vol")
        # Signed columns are scaled but not centred, so multiplying by side is an exact mirror.
        z = np.stack(
            [(grid(c) - (0.0 if c in SIGNED_COLUMNS else stats[c][0])) / stats[c][1] for c in obs_columns],
            axis=-1,
        )
        return cls(
            sessions=sessions,
            mid=grid("mid"),
            spread=grid("spread_bps") / 1e4,
            volume=grid("volume"),
            value=grid("value"),
            sigma=grid("rv_30m") / np.sqrt(RV_WINDOW),
            prof=prof,
            cum_prof=np.concatenate([np.zeros((len(sessions), 1)), np.cumsum(prof, axis=1)], axis=1),
            obs=np.clip(z, -OBS_CLIP, OBS_CLIP),
            obs_columns=obs_columns,
            stats=stats,
        )

    def session_ids(self, split: str | None = None) -> np.ndarray:
        if split is None:
            return np.arange(len(self.sessions))
        return np.flatnonzero(self.sessions["split"].to_numpy() == split)

    @property
    def signed_mask(self) -> np.ndarray:
        return np.array([c in SIGNED_COLUMNS for c in self.obs_columns])
