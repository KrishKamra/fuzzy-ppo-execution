"""Validated env and fee configs (`configs/env.yaml`, `configs/fees_nse.yaml`)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from hftf_data.paths import REPO_ROOT

ENV_YAML = REPO_ROOT / "configs" / "env.yaml"
FEES_YAML = REPO_ROOT / "configs" / "fees_nse.yaml"


class EpisodeConfig(BaseModel):
    horizons_min: list[int] = [30, 60, 120]
    earliest_start: str = "09:20"
    latest_end: str = "15:15"
    participation_loguniform: tuple[float, float] = (0.01, 0.20)
    sides: list[Literal["buy", "sell"]] = ["buy", "sell"]


class ActionConfig(BaseModel):
    a_max: float = 4.0
    participation_cap: float = 0.30


class ImpactConfig(BaseModel):
    beta: float = 0.5
    eta: float = 0.5
    gamma: float = 0.3
    tau_res_min: float = 5.0
    volume_floor_frac: float = 0.01  # V_eff = max(V, frac * expected volume): guards (near-)empty minutes
    randomise: dict[str, tuple[float, float]] = Field(default_factory=dict)


class RewardConfig(BaseModel):
    risk_aversion: float = 1.0e-4
    # arrival: (p − m0) per child. mark_to_market: the same IS split into each step's execution cost and
    # inventory P&L (same episode return). vs_vwap: mark_to_market minus a shadow adaptive VWAP on the same
    # path (return shifted by an action-independent amount). See ARCHITECTURE §4.6.
    mode: Literal["arrival", "mark_to_market", "vs_vwap"] = "vs_vwap"
    # realised: charge the inventory × mid-move term of the mark-to-market split. expected: replace it by its
    # martingale expectation, 0, so the reward is expected cost + risk penalty (Almgren-Chriss). Needs
    # mark_to_market or vs_vwap. Evaluation always reports the realised IS. See ARCHITECTURE §4.6.
    price_drift: Literal["realised", "expected"] = "realised"


class TickConfig(BaseModel):
    """TickSim (ARCHITECTURE §4.8): fills against the replayed 1-second L1 state on tick sessions."""

    path: str = "data/processed/l1_1s_v1.parquet"
    latency_s: int = Field(1, ge=0, le=14)  # send at the bar boundary, fill against the state L seconds later
    # Orders per decision, sent evenly over the minute. Each is a child order and pays the per-order fee.
    sub_slices: int = Field(1, ge=1, le=4)
    depth_fills: bool = True  # shares within the displayed touch pay the half spread only (no walk)


class LimitConfig(BaseModel):
    """Passive child orders at the near touch (ARCHITECTURE §4.9)."""

    enabled: bool = False  # policies get a second action: the passive fraction of each child
    # BarSim: a bar that trades at our price but not through it fills min(posted, θ_touch·V), one that trades
    # through it min(posted, θ_through·V); both fitted to the TickSim queue model on the val tick sessions
    # (`hftf eval calibrate-limit`).
    touch_fill_frac: float = 0.05
    through_fill_frac: float = 0.2
    # Passive fractions within `dead_zone` of 0 or 1 snap to it, so a child can be a single order of either
    # kind. Without it a Beta never reaches 0 or 1 exactly and every child pays the per-order fee twice.
    dead_zone: float = Field(0.1, ge=0.0, lt=0.5)


class EnvConfig(BaseModel):
    features_path: str = "data/processed/features_1m_v3.parquet"
    episode: EpisodeConfig = EpisodeConfig()
    action: ActionConfig = ActionConfig()
    regime_inputs: list[str] = ["imb_bar_proxy_ema5", "log_spread_var_30m", "log_rv_30m"]
    impact: ImpactConfig = ImpactConfig()
    reward: RewardConfig = RewardConfig()
    tick: TickConfig = TickConfig()
    limit: LimitConfig = LimitConfig()

    @classmethod
    def load(cls, path: Path = ENV_YAML) -> EnvConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class SideRates(BaseModel):
    buy: float
    sell: float


class Brokerage(BaseModel):
    bps: float = 0.0
    fixed_per_order_inr: float = 0.0


class FeeConfig(BaseModel):
    product: Literal["delivery", "intraday"] = "delivery"
    stt: dict[str, SideRates]
    exchange_txn: float
    sebi_fee: float
    stamp_duty: dict[str, SideRates]
    gst_rate: float
    brokerage: Brokerage = Brokerage()

    @classmethod
    def load(cls, path: Path = FEES_YAML) -> FeeConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
