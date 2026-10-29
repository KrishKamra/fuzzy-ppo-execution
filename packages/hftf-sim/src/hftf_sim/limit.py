"""Passive (limit) child orders at the near touch (ARCHITECTURE §4.9).

At each decision the agent posts `⌊f·q⌋` of its child at the near touch (the bid for a buy, the ask for a
sell) and sends the rest as a marketable order. The passive order rests for the minute; whatever is not
filled by the end of the bar is cancelled and returns to the inventory. A fill pays the limit price (no
spread, no temporary impact, no transient kick) plus fees; adverse selection comes from the replayed prices.

Two fill models share this interface:
  * BarSim (`bar_passive_fill`, every day): the bar's trade range against our price. Through it:
    `min(posted, θ_through·V)`. At it: `min(posted, θ_touch·V)`. Never reached: nothing. Both θ are
    calibrated against the queue model on the val tick sessions (`calibrate_touch_fill`).
  * TickSim (`queue_passive_fill`, tick sessions): we join the back of the displayed queue. Volume that
    trades at our price (sell-initiated for a buy) consumes the queue ahead first, then us. A print through
    our price means the queue ahead has cleared, so we fill up to the volume traded at or through our price.
    A fill never exceeds the volume that actually traded there: a 5-share print through the bid does not
    fill a 400-share order.
"""

from __future__ import annotations

import numpy as np

from hftf_sim.market import MarketData

PRICE_TOL = 1e-6  # relative; far below one tick at every price band


def bar_passive_fill(
    post: np.ndarray,
    side: np.ndarray,
    price: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    volume: np.ndarray,
    theta_touch: float,
    theta_through: float,
) -> np.ndarray:
    """Shares filled out of `post` resting at `price` for one bar with trade range [low, high], volume V."""
    tol = PRICE_TOL * price
    through = np.where(side > 0, low < price - tol, high > price + tol)
    touched = np.where(side > 0, low <= price + tol, high >= price - tol)
    theta = np.where(through, theta_through, np.where(touched, theta_touch, 0.0))
    return np.minimum(post, np.floor(theta * volume))


def queue_passive_fill(
    post: np.ndarray,
    side: np.ndarray,
    price: np.ndarray,
    queue: np.ndarray,
    px_min: np.ndarray,
    px_max: np.ndarray,
    buy_vol: np.ndarray,
    sell_vol: np.ndarray,
) -> np.ndarray:
    """Shares filled out of `post` joining a queue of `queue` shares at `price`. The trade arrays are
    (n, seconds) windows after the order arrives; seconds without prints hold NaN prices and 0 volume."""
    p, tol, buy = price[:, None], (PRICE_TOL * price)[:, None], (side > 0)[:, None]
    with np.errstate(invalid="ignore"):
        through = np.where(buy, px_min < p - tol, px_max > p + tol).any(axis=1)
        at = np.where(buy, px_min <= p + tol, px_max >= p - tol)
    hit = np.where(at, np.where(buy, sell_vol, buy_vol), 0.0).sum(axis=1)  # traded at or through our price
    filled = np.where(through, np.minimum(hit, post), np.clip(hit - queue, 0.0, post))
    return np.floor(filled)


def calibrate_touch_fill(
    market: MarketData,
    ticks,
    sessions: np.ndarray,
    latency_s: int = 1,
    participation: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3),
    first_bar: int = 5,
) -> dict:
    """Fit BarSim's θ_touch and θ_through to the TickSim queue model: passive orders of `participation ×
    expected minute volume` on every bar of `sessions`, both sides. Each θ minimises the squared fill error
    on its own bars (traded at our price but not through it, or through it)."""
    from hftf_data.calendar import SESSION_MINUTES

    s = np.repeat(sessions, SESSION_MINUTES - first_bar)
    j = np.tile(np.arange(first_bar, SESSION_MINUTES), len(sessions))
    rows = []
    for side in (1, -1):
        for rho in participation:
            sd = np.full(len(s), side)
            post = np.maximum(np.floor(rho * market.prof[s, j]), 1.0)
            tr, g0 = ticks.row[s], 60 * j + latency_s
            quote = np.where(side > 0, market.bid[s, j - 1], market.ask[s, j - 1])
            low, high, tol = market.low[s, j], market.high[s, j], PRICE_TOL * quote
            through = low < quote - tol if side > 0 else high > quote + tol
            reached = low <= quote + tol if side > 0 else high >= quote - tol
            tick = ticks.passive_fill(post, sd, tr, g0, 60 * j + 60)
            rows.append((post, tick, through, reached & ~through, market.volume[s, j]))
    post, tick, through, touched, vol = (np.concatenate(x) for x in zip(*rows, strict=True))
    grid = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 400)])

    def fit(mask: np.ndarray) -> float:
        err = [((np.minimum(post, np.floor(t * vol)) - tick)[mask] ** 2).mean() for t in grid]
        return float(grid[int(np.argmin(err))])

    theta_touch, theta_through = fit(touched), fit(through)
    theta = np.where(through, theta_through, np.where(touched, theta_touch, 0.0))
    bar_fill = np.minimum(post, np.floor(theta * vol))
    untouched = ~through & ~touched

    def rate(fill: np.ndarray, mask: np.ndarray) -> float:
        return float(fill[mask].sum() / max(post[mask].sum(), 1.0))

    every = np.ones(len(post), bool)
    return {
        "theta_touch": theta_touch,
        "theta_through": theta_through,
        "orders": int(len(post)),
        "share_through": float(through.mean()),
        "share_touched": float(touched.mean()),
        "fill_rate_tick": rate(tick, every),
        "fill_rate_bar": rate(bar_fill, every),
        "fill_rate_tick_when_through": rate(tick, through),
        "fill_rate_bar_when_through": rate(bar_fill, through),
        "fill_rate_tick_when_touched": rate(tick, touched),
        "fill_rate_bar_when_touched": rate(bar_fill, touched),
        "fill_rate_tick_when_untouched": rate(tick, untouched),
    }
