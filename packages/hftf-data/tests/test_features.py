import numpy as np
import pandas as pd

from hftf_data.calendar import CLOSING_AUCTION_FROM, SESSION_MINUTES, session_grid
from hftf_data.features import _diff, _rv, assign_split


def _day(values, date="2026-09-01"):
    idx = session_grid([date])[: len(values)]
    s = pd.Series(values, index=idx, dtype=float)
    return s, idx.tz_convert("Asia/Kolkata").date, pd.Series(np.arange(len(values)), index=idx)


def test_grid_stops_at_15_15_ist():
    g = session_grid(["2026-07-01", CLOSING_AUCTION_FROM])
    assert len(g) == 2 * SESSION_MINUTES == 720
    last = g[-1].tz_convert("Asia/Kolkata")
    assert (last.hour, last.minute) == (15, 14)


def test_partial_window_rv_is_rescaled():
    r2, day, _ = _day([np.nan, 1e-6, 1e-6, 1e-6, 1e-6])
    rv = _rv(r2, day, 30)
    assert rv.iloc[:3].isna().all()  # fewer than 3 returns
    assert np.isclose(rv.iloc[3], np.sqrt(3e-6 * 30 / 3))
    assert np.isclose(rv.iloc[4], np.sqrt(30e-6))  # constant per-minute variance -> same 30m estimate


def test_returns_are_measured_since_open_inside_the_window():
    lp, day, mso = _day(np.log([100.0, 101.0, 102.0, 103.0]))
    r = _diff(lp, day, mso, 2)
    assert np.isnan(r.iloc[0])
    assert np.isclose(r.iloc[1], np.log(1.01))  # mso 1 < w: since the open
    assert np.isclose(r.iloc[3], np.log(103 / 101))  # full window


def test_windows_never_cross_sessions():
    a, *_ = _day([1e-6] * 5, "2026-09-01")
    b, *_ = _day([1e-6] * 5, "2026-09-02")
    r2 = pd.concat([a, b])
    rv = _rv(r2, r2.index.tz_convert("Asia/Kolkata").date, 30)
    assert rv.iloc[5:7].isna().all()


def test_split_labels():
    dates = pd.Series(pd.to_datetime(["2026-06-05", "2026-07-01", "2026-09-02", "2026-10-05", "2026-10-21"]))
    excl = pd.Series([True, False, False, False, False])
    splits = {
        "train": ("2026-06-03", "2026-08-31"),
        "val": ("2026-09-01", "2026-09-30"),
        "test": ("2026-10-01", "2026-10-19"),
    }
    assert assign_split(dates, excl, splits).tolist() == ["excluded", "train", "val", "test", "excluded"]
