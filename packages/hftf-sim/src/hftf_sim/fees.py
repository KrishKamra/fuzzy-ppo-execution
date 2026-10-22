"""NSE cash-equity transaction costs per child order (ARCHITECTURE §4.5)."""

from __future__ import annotations

import numpy as np

from hftf_sim.config import FeeConfig


class FeeModel:
    """Fees in ₹ for child orders: side-specific proportional rates plus a fixed fee per non-zero child."""

    def __init__(self, cfg: FeeConfig) -> None:
        stt, stamp = cfg.stt[cfg.product], cfg.stamp_duty[cfg.product]
        self.rate_buy = stt.buy + stamp.buy
        self.rate_sell = stt.sell + stamp.sell
        self.exchange = cfg.exchange_txn + cfg.sebi_fee  # GST applies to these and to brokerage
        self.brokerage = cfg.brokerage.bps / 1e4
        self.fixed = cfg.brokerage.fixed_per_order_inr
        self.gst = cfg.gst_rate

    @classmethod
    def zero(cls) -> FeeModel:
        side = {"buy": 0.0, "sell": 0.0}
        cfg = FeeConfig(
            stt={"delivery": side},
            exchange_txn=0.0,
            sebi_fee=0.0,
            stamp_duty={"delivery": side},
            gst_rate=0.0,
        )
        return cls(cfg)

    def __call__(self, notional: np.ndarray, side: np.ndarray, traded: np.ndarray) -> np.ndarray:
        """Total fees (₹). `notional` = q·p_exec ≥ 0, `side` = ±1, `traded` = q > 0."""
        rate = np.where(side > 0, self.rate_buy, self.rate_sell)
        taxable = notional * (self.exchange + self.brokerage) + self.fixed * traded
        return notional * rate + taxable * (1.0 + self.gst)
