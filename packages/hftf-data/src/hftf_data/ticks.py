"""Tick cleaning and per-minute L1 microstructure features.

Bloomberg's NSE tick history is a ~1-second conflated snapshot stream, not the raw exchange feed.
Each snapshot can carry several rows sharing one timestamp: an untagged TRADE (last print), a 'BC'
TRADE (the remaining volume batched since the previous snapshot), then BID and ASK with sizes.

Condition codes on TRADE rows (verified: untagged + BC volume == 1-min bar volume, ratio 1.000 every day):
  <none>, BC     continuous-session volume                       -> keep
  T              cumulative indicative volume (pre-open & close)  -> drop (cumulative, not a print)
  ACB            day-total summary print near 15:29               -> drop
  AOB, AUO, AUC  auction open/close markers                       -> drop
BID/ASK rows tagged MQ carry price 0.0 during the pre-open auction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hftf_data.calendar import continuous_mask

CONTINUOUS_TRADE_CODES = {"", "BC"}


def clean_quotes(ticks: pd.DataFrame) -> pd.DataFrame:
    """L1 quote state per distinct timestamp: time, bid, ask, bid_sz, ask_sz (continuous session only).

    Collapsing to the last BID and last ASK per timestamp before forward-filling removes the ~3%
    transiently crossed states that a row-by-row forward fill produces.
    """
    q = ticks[ticks["type"].isin(["BID", "ASK"])]
    if q.empty:
        return pd.DataFrame(columns=["time", "bid", "ask", "bid_sz", "ask_sz"])
    last = q.groupby(["time", "type"], sort=True)[["value", "size"]].last().unstack("type")
    out = pd.DataFrame(
        {
            "bid": last.get(("value", "BID"), np.nan),
            "ask": last.get(("value", "ASK"), np.nan),
            "bid_sz": last.get(("size", "BID"), np.nan),
            "ask_sz": last.get(("size", "ASK"), np.nan),
        }
    ).ffill()
    out = out.reset_index()
    ok = (out["bid"] > 0) & (out["ask"] > out["bid"]) & (out["bid_sz"] > 0) & (out["ask_sz"] > 0)
    out = out[ok & continuous_mask(out["time"])]
    return out.reset_index(drop=True)


def clean_trades(ticks: pd.DataFrame, quotes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Continuous-session trade prints with a Lee-Ready aggressor sign (+1 buy, -1 sell, 0 unknown).

    The quote used for signing is the one prevailing strictly before the trade timestamp, because
    quotes stamped at the same instant already reflect the post-trade book.
    """
    tr = ticks[ticks["type"] == "TRADE"]
    codes = tr["conditionCodes"].fillna("").astype(str)
    tr = tr[codes.isin(CONTINUOUS_TRADE_CODES) & continuous_mask(tr["time"]) & (tr["value"] > 0)]
    tr = tr[["time", "value", "size"]].rename(columns={"value": "price"}).reset_index(drop=True)
    if tr.empty:
        return tr.assign(sign=pd.Series(dtype=float))
    quotes = clean_quotes(ticks) if quotes is None else quotes
    tr = pd.merge_asof(
        tr, quotes[["time", "bid", "ask"]], on="time", direction="backward", allow_exact_matches=False
    )
    mid = (tr["bid"] + tr["ask"]) / 2
    quote_sign = np.sign(tr["price"] - mid)
    tick_rule = np.sign(tr["price"].diff()).replace(0, np.nan).ffill()
    tr["sign"] = quote_sign.where(quote_sign != 0, tick_rule).fillna(0.0)
    return tr.drop(columns=["bid", "ask"])


def order_flow_imbalance(q: pd.DataFrame) -> pd.Series:
    """Cont, Kukanov & Stoikov (2014) OFI contribution of each L1 quote update (in shares)."""
    pb, pa, qb, qa = q["bid"], q["ask"], q["bid_sz"], q["ask_sz"]
    pb0, pa0, qb0, qa0 = pb.shift(), pa.shift(), qb.shift(), qa.shift()
    e = (pb >= pb0) * qb - (pb <= pb0) * qb0 - (pa <= pa0) * qa + (pa >= pa0) * qa0
    return e.where(pb0.notna(), 0.0).astype(float)


def minute_microstructure(ticks: pd.DataFrame) -> pd.DataFrame:
    """Per-minute L1 features from one security-day of raw ticks. Index: minute start (UTC)."""
    q = clean_quotes(ticks)
    if q.empty:
        return pd.DataFrame()
    depth = q["bid_sz"] + q["ask_sz"]
    q = q.assign(
        mid=(q["bid"] + q["ask"]) / 2,
        imb=(q["bid_sz"] - q["ask_sz"]) / depth,
        micro=(q["ask"] * q["bid_sz"] + q["bid"] * q["ask_sz"]) / depth,
        depth=depth,
        ofi=order_flow_imbalance(q),
        minute=q["time"].dt.floor("min"),
    )
    q["spread_bps"] = (q["ask"] - q["bid"]) / q["mid"] * 1e4
    q["micro_dev_bps"] = (q["micro"] - q["mid"]) / q["mid"] * 1e4
    agg = q.groupby("minute").agg(
        l1_imb_last=("imb", "last"),
        l1_imb_mean=("imb", "mean"),
        microprice_dev_bps=("micro_dev_bps", "last"),
        tick_spread_bps_mean=("spread_bps", "mean"),
        tick_spread_bps_var=("spread_bps", "var"),
        ofi=("ofi", "sum"),
        l1_depth_mean=("depth", "mean"),
        n_quote_updates=("bid", "size"),
        bid_sz_last=("bid_sz", "last"),
        ask_sz_last=("ask_sz", "last"),
    )
    agg["ofi_norm"] = agg["ofi"] / agg["l1_depth_mean"].replace(0, np.nan)

    tr = clean_trades(ticks, q)
    if len(tr):
        tr = tr.assign(
            minute=tr["time"].dt.floor("min"),
            buy=np.where(tr["sign"] > 0, tr["size"], 0.0),
            sell=np.where(tr["sign"] < 0, tr["size"], 0.0),
        )
        ta = tr.groupby("minute").agg(
            n_trades=("price", "size"),
            tick_volume=("size", "sum"),
            buy_vol=("buy", "sum"),
            sell_vol=("sell", "sum"),
        )
        ta["signed_vol"] = ta["buy_vol"] - ta["sell_vol"]
        ta["trade_imb"] = ta["signed_vol"] / (ta["buy_vol"] + ta["sell_vol"]).replace(0, np.nan)
        agg = agg.join(ta, how="outer")
    agg.index.name = "time"
    return agg
