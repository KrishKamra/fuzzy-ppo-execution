import numpy as np
import pandas as pd
import pytest

from hftf_data.derive import bar_imbalance_proxy, infer_tick_size, nse_tick_size
from hftf_data.ticks import clean_quotes, clean_trades, order_flow_imbalance

T0 = pd.Timestamp("2026-10-05 05:00:00", tz="UTC")  # 10:30 IST, inside the continuous session


def _ticks(rows):
    df = pd.DataFrame(rows, columns=["dt", "type", "value", "size", "conditionCodes"])
    df["time"] = T0 + pd.to_timedelta(df.pop("dt"), unit="s")
    return df


@pytest.mark.parametrize(
    ("price", "tick"), [(100, 0.01), (500, 0.05), (2600, 0.10), (7600, 0.50), (15000, 1.0), (30000, 5.0)]
)
def test_nse_tick_rule(price, tick):
    assert nse_tick_size(price) == tick


def test_infer_tick_size_picks_largest_consistent_grid():
    rng = np.random.default_rng(0)
    assert infer_tick_size(7600 + 0.5 * rng.integers(0, 200, 1000)) == 0.5
    assert infer_tick_size(2600 + 0.1 * rng.integers(0, 200, 1000)) == 0.1


def test_trade_filter_keeps_only_continuous_prints():
    t = _ticks(
        [
            (0, "BID", 100.0, 10, None),
            (0, "ASK", 100.1, 10, None),
            (1, "TRADE", 100.1, 5, None),
            (1, "TRADE", 100.1, 7, "BC"),
            (2, "TRADE", 100.0, 9999, "T"),
            (3, "TRADE", 100.0, 88888, "ACB"),
        ]
    )
    tr = clean_trades(t)
    assert tr["size"].sum() == 12
    assert (tr["sign"] == 1).all()  # printed at the ask -> buyer-initiated


def test_quote_collapse_removes_transient_crossing():
    # The book shifts up within one timestamp. A row-wise ffill would momentarily pair the new
    # bid (100.2) with the stale ask (100.1): a crossed state that never existed.
    t = _ticks(
        [
            (0, "BID", 100.0, 10, None),
            (0, "ASK", 100.1, 10, None),
            (1, "BID", 100.2, 4, None),
            (1, "ASK", 100.3, 6, None),
        ]
    )
    q = clean_quotes(t)
    assert len(q) == 2
    assert (q["ask"] > q["bid"]).all()


def test_ofi_signs():
    q = pd.DataFrame(
        {
            "bid": [100.0, 100.1, 100.1],
            "ask": [100.2, 100.2, 100.2],
            "bid_sz": [10.0, 5.0, 8.0],
            "ask_sz": [10.0, 10.0, 10.0],
        }
    )
    e = order_flow_imbalance(q)
    assert e.iloc[0] == 0
    assert e.iloc[1] == 5  # bid improved: +new bid size
    assert e.iloc[2] == 3  # bid size grew at same price


def test_bar_imbalance_proxy():
    idx = pd.date_range(T0, periods=2, freq="min")
    bid = pd.DataFrame({"time": idx, "volume": [300, 300], "numEvents": [3, 3]})  # mean size 100
    ask = pd.DataFrame({"time": idx, "volume": [100, 100], "numEvents": [1, 4]})  # mean size 100, 25
    np.testing.assert_allclose(bar_imbalance_proxy(bid, ask).to_numpy(), [0.0, 0.6])
