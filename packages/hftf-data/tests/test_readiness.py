import pandas as pd

from hftf_data.readiness import evaluate, load_thresholds

EPISODE_COLS = load_thresholds()["episode_columns"]

EQUITIES = [f"E{i:02d} IS Equity" for i in range(50)]
CTX = ["NIFTY Index", "NSEBANK Index", "INVIXN Index"]


def _report(n_eq=3, bar_days=96, missing=(), tick_eq=3, tick_days=20, ratio=1.0, crossed=0, pearson=0.8):
    eqs = EQUITIES[:n_eq]
    bars = [
        {
            "security": s.replace(" ", "_"),
            "event": ev,
            "rows": 1,
            "days": bar_days,
            "missing_sessions": list(missing),
        }
        for s in eqs
        for ev in ("TRADE", "BID", "ASK")
    ]
    bars += [
        {
            "security": s.replace(" ", "_"),
            "event": "TRADE",
            "rows": 1,
            "days": bar_days,
            "missing_sessions": [],
        }
        for s in CTX
    ]
    ticks = [
        {
            "security": s.replace(" ", "_"),
            "day": f"d{i}",
            "last_tick_ist": "15:29:00",
            "clean_over_bar": ratio,
            "crossed_after_clean": crossed,
        }
        for s in eqs[:tick_eq]
        for i in range(tick_days)
    ]
    return {
        "intraday_equities": eqs,
        "bars": bars,
        "ticks": ticks,
        "imbalance_proxy": [{"security": s.replace(" ", "_"), "pearson": pearson} for s in eqs[:tick_eq]],
        "tick_size": [{"security": s, "inferred_tick": 0.1, "nse_rule_tick_at_median_px": 0.1} for s in eqs],
        "daily": {"equity": {"sessions_per_security": {s: 3000 for s in eqs}}},
    }


def _features(eqs=EQUITIES[:3]):
    splits = ["train", "val", "test"]  # one row per split for every equity
    n = len(eqs) * len(splits)
    return pd.DataFrame(
        {
            "security": [s for s in eqs for _ in splits],
            "dq_partial_day": [False] * n,
            "imb_bar_proxy": [0.1] * n,
            "spread_var_30m": [1.0] * n,
            "rv_30m": [0.01] * n,
            "split": splits * len(eqs),
            "mso": [10] * n,
            **{c: [1.0] * n for c in EPISODE_COLS if c not in ("rv_30m",)},
        }
    )


def _failed(report, features="default"):
    feats = _features(report["intraday_equities"]) if isinstance(features, str) else features
    return {c.id for c in evaluate(report, load_thresholds(), feats) if not c.passed}


def test_current_dataset_shape_is_ready():
    assert _failed(_report()) == set()
    assert _failed(_report(missing=["2026-09-10"])) == set()  # acknowledged gap, dropped whole


def test_each_defect_fails_its_check():
    assert _failed(_report(n_eq=2, tick_eq=2)) == {"R01", "R05"}
    assert _failed(_report(bar_days=80)) == {"R02"}
    assert _failed(_report(missing=["2026-09-11"])) == {"R03"}
    assert _failed(_report(tick_days=15)) == {"R05"}
    assert _failed(_report(tick_eq=2)) == {"R05"}
    assert _failed(_report(ratio=1.01)) == {"R06"}
    assert _failed(_report(crossed=3)) == {"R07"}
    assert _failed(_report(pearson=0.5)) == {"R08"}


def test_partial_tick_days_do_not_count():
    r = _report()
    r["ticks"][0]["last_tick_ist"] = "12:14:44"  # one equity drops to 19 full days -> 2 covered
    assert _failed(r) == {"R05"}


def test_missing_features_fail():
    assert _failed(_report(), features=None) == {"R11", "R12"}


def test_features_must_cover_every_equity():
    assert _failed(_report(), features=_features(EQUITIES[:2])) == {"R11"}


def test_episode_rows_must_be_complete():
    f = _features()
    f.loc[0, "prof_share"] = None
    assert _failed(_report(), features=f) == {"R12"}
    f = _features()
    f.loc[0, "mso"] = 2  # before the first decision: may be NaN
    f.loc[0, "prof_share"] = None
    assert _failed(_report(), features=f) == set()
    assert _failed(_report(), features=_features().assign(split="train")) == {"R12"}  # val/test empty
