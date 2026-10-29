import numpy as np
import pandas as pd

from hftf_data.l1 import SESSION_SECONDS, second_grid

DAY = pd.Timestamp("2026-10-05")
OPEN = pd.Timestamp("2026-10-05 09:15", tz="Asia/Kolkata").tz_convert("UTC")


def _ticks(rows):
    df = pd.DataFrame(rows, columns=["dt", "type", "value", "size", "conditionCodes"])
    df["time"] = OPEN + pd.to_timedelta(df.pop("dt"), unit="s")
    return df


def test_second_grid_state_trades_and_carry():
    g = second_grid(
        _ticks(
            [
                (10.2, "BID", 100.0, 10, None),
                (10.2, "ASK", 100.2, 20, None),
                (10.7, "BID", 100.1, 5, None),  # same second: the last update wins
                (12.5, "TRADE", 100.2, 7, None),
                (12.6, "TRADE", 100.2, 3, "BC"),
                (12.9, "TRADE", 100.0, 999, "T"),  # cumulative indicative print: dropped
                (14.0, "ASK", 100.3, 8, None),
                (14.5, "TRADE", 100.1, 4, None),
            ]
        ),
        DAY,
    )
    assert len(g) == SESSION_SECONDS
    assert g.loc[:9, "bid"].isna().all()  # never back-filled before the first quote
    r = g.set_index("sec")
    assert tuple(r.loc[10, ["bid", "ask", "bid_sz", "ask_sz"]]) == (100.1, 100.2, 5, 20)
    assert tuple(r.loc[13, ["bid", "ask", "quote_age_s"]]) == (100.1, 100.2, 3)
    assert tuple(r.loc[14, ["ask", "ask_sz", "quote_age_s"]]) == (100.3, 8, 0)
    assert r.loc[12, "volume"] == 10 and r.loc[12, "buy_vol"] == 10  # at the ask: buyer-initiated
    assert r.loc[14, "sell_vol"] == 4  # inside the 100.1/100.3 quote, below mid: seller-initiated
    assert r["volume"].sum() == 14 and np.isnan(r.loc[13, "px_min"])
