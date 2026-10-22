"""Causal 1-minute feature table (v3) for the execution environment.

Supersedes bbg_data/features/features_1m.parquet, which has two defects:
  * tick volume / signed volume included cumulative 'T' and day-total 'ACB' prints (2-76x inflated);
  * L1 imbalance/OFI used a row-wise forward fill that produced ~3% crossed quotes.
It also had L1 imbalance for only 22% of rows; v2 fills the remaining days with a bar-derived proxy.

v3 (final dataset) changes v2 as follows. Nothing is synthesised: a value is either observed, carried
forward inside its session (flagged), or estimated over a shorter window during warm-up (flagged).
  * The grid is 09:15-15:15 IST (360 min) on every day; from 2026-08-03 15:15-15:30 is a closing
    auction (calendar.py), and v2 filled it with fake zero-volume minutes.
  * A session is kept only if TRADE, BID, ASK and the context bars all exist for it. Whole missing
    sessions (2026-09-10) are dropped, never imputed.
  * Rolling statistics use the available minutes during the first `w` minutes of a session; realised
    variance is rescaled by w/n, returns are measured since the open.
  * Context bars are put on the grid and carried forward inside the session before differencing.
  * `split` labels come from configs/env.yaml; partial and profile burn-in days are `excluded`.

Row semantics: `time` is the bar START (UTC). Everything in a row is observable at
`decision_time = time + 1 min`; rolling windows never cross a session boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from hftf_data.calendar import IST, OPEN_MINUTE_OF_DAY, SESSION_MINUTES, session_grid
from hftf_data.derive import bar_imbalance_proxy, infer_tick_size
from hftf_data.loaders import load_bars, load_daily, load_static, load_ticks, tick_days
from hftf_data.paths import REPO_ROOT, DataPaths
from hftf_data.ticks import minute_microstructure

FEATURES_FILE = "features_1m_v3.parquet"
ENV_CONFIG = REPO_ROOT / "configs" / "env.yaml"
RV_WINDOWS = (5, 15, 30, 60)
SPREAD_VAR_WINDOWS = (15, 30, 60)
PROFILE_DAYS = 20
PROFILE_MIN_DAYS = 3
WARM_MIN_OBS = 3  # minimum observations for a partial-window estimate
WARMUP_MINUTES = 30  # longest eFNN input window; rows before it carry dq_warmup
LOG_EPS = 1e-8
CONTEXT = {"nifty": "NIFTY Index", "banknifty": "NSEBANK Index", "vix": "INVIXN Index"}
TICK_FILL_COLS = ("l1_imb_last", "l1_imb_mean", "microprice_dev_bps", "bid_sz_last", "ask_sz_last")
TICK_COUNT_COLS = ("ofi", "n_quote_updates", "n_trades", "tick_volume", "buy_vol", "sell_vol", "signed_vol")


def _roll(df: pd.DataFrame, col: str, w: int, fn: str, min_periods: int | None = None) -> pd.Series:
    mp = max(2, w // 3) if min_periods is None else min_periods
    return df.groupby("date", sort=False)[col].transform(
        lambda s: getattr(s.rolling(w, min_periods=mp), fn)()
    )


def _rv(r2: pd.Series, day, w: int) -> pd.Series:
    """sqrt of w-minute realised variance; a partial window of n < w returns is rescaled by w/n."""
    g = r2.groupby(day, sort=False)
    s = g.transform(lambda x: x.rolling(w, min_periods=WARM_MIN_OBS).sum())
    n = g.transform(lambda x: x.rolling(w, min_periods=1).count())
    return np.sqrt(s * w / n)


def _diff(x: pd.Series, day, mso: pd.Series, w: int) -> pd.Series:
    """x_t - x_{t-w} inside a session; in the first w minutes, the change since the session's first value."""
    g = x.groupby(day, sort=False)
    return g.diff(w).where(mso >= w, x - g.transform("first")).where(mso > 0)


def _mso(index: pd.DatetimeIndex) -> pd.Series:
    ist = index.tz_convert(IST)
    return pd.Series(ist.hour * 60 + ist.minute - OPEN_MINUTE_OF_DAY, index=index)


def _last_close(bars: pd.DataFrame) -> pd.Series:
    return bars.set_index("time")["close"] if len(bars) else pd.Series(dtype=float)


def tick_features(security: str, paths: DataPaths) -> pd.DataFrame:
    parts = [minute_microstructure(load_ticks(security, d, paths)) for d in tick_days(security, paths)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


def context_features(paths: DataPaths) -> pd.DataFrame:
    """Same-minute index features on the session grid; a minute without a print keeps the last value."""
    cols: dict[str, pd.Series] = {}
    for key, sec in CONTEXT.items():
        b = load_bars(sec, "TRADE", paths)
        if b.empty:
            continue
        c = b.set_index("time")["close"].astype(float)
        c = c.reindex(session_grid(sorted({str(d) for d in b["date"]})))
        day = c.index.tz_convert(IST).date
        c = c.groupby(day).ffill()
        mso = _mso(c.index)
        if key == "vix":
            cols["ctx_vix"] = c
            cols["ctx_vix_chg_30m"] = _diff(c, day, mso, 30)
            continue
        logc = np.log(c)
        lr = logc.groupby(day).diff()
        cols[f"ctx_{key}_ret_1m"] = lr
        cols[f"ctx_{key}_ret_30m"] = _diff(logc, day, mso, 30)
        if key == "nifty":
            cols["ctx_nifty_rv_30m"] = _rv(lr**2, day, 30)
    out = pd.DataFrame(cols)
    return out.dropna(how="all")


def daily_lagged(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-day features from daily history, shifted one session so they are known before the open."""
    d = daily.sort_values("date")
    px, vol = d["PX_LAST"].astype(float), d["PX_VOLUME"].astype(float)
    ret = px.pct_change()
    ivol = d.get("30DAY_IMPVOL_100.0%MNY_DF")
    f = pd.DataFrame(
        {
            "date": d["date"].dt.date.to_numpy(),
            "d_ret_1d": ret,
            "d_vol_20d": ret.rolling(20, min_periods=10).std() * np.sqrt(252),
            "d_adv_20d": vol.rolling(20, min_periods=10).mean(),
            "d_range_pct": (d["PX_HIGH"] - d["PX_LOW"]) / px,
            "d_amihud_20d": (ret.abs() / (px * vol).replace(0, np.nan)).rolling(20, min_periods=10).mean(),
            "d_ivol_30d": ivol.ffill(limit=1) if ivol is not None else np.nan,  # 1-day IV gap: carry
            "d_volume": vol,
        }
    )
    cols = [c for c in f.columns if c != "date"]
    f[cols] = f[cols].shift(1)
    f["daily_volume_same_day"] = vol.to_numpy()  # NOT causal: used only for data-quality flags
    return f


def load_splits(path=ENV_CONFIG) -> dict[str, tuple[str, str]]:
    return {k: tuple(v) for k, v in yaml.safe_load(path.read_text())["splits"].items()}


def assign_split(dates: pd.Series, excluded: pd.Series, splits: dict[str, tuple[str, str]]) -> pd.Series:
    """Walk-forward split label per row: train/val/test by date, `excluded` for flagged or unlisted days."""
    d = pd.to_datetime(dates)
    out = pd.Series("excluded", index=dates.index, dtype=object)
    for name, (lo, hi) in splits.items():
        out[(d >= lo) & (d <= hi)] = name
    out[excluded.astype(bool)] = "excluded"
    return out


def build_security(
    security: str,
    ctx: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    paths: DataPaths | None = None,
) -> pd.DataFrame:
    paths = paths or DataPaths()
    trade = load_bars(security, "TRADE", paths)
    if trade.empty:
        return pd.DataFrame()
    bid, ask = load_bars(security, "BID", paths), load_bars(security, "ASK", paths)
    t = trade.set_index("time")
    df = pd.DataFrame(
        {
            "o": t["open"],
            "h": t["high"],
            "l": t["low"],
            "c": t["close"],
            "volume": t["volume"],
            "n_events": t["numEvents"],
            "value": t["value"],
        }
    ).astype(float)
    df["bid"], df["ask"] = _last_close(bid), _last_close(ask)
    imb_proxy = bar_imbalance_proxy(bid, ask) if len(bid) and len(ask) else None

    # A session needs all three bar series and the market context; incomplete ones are dropped whole.
    days = set(trade["date"]) & set(bid.get("date", [])) & set(ask.get("date", []))
    if ctx is not None and len(ctx):
        days &= set(ctx.index.tz_convert(IST).date)
    df = df.reindex(session_grid(sorted(str(d) for d in days)))
    df.index.name = "time"
    ist = df.index.tz_convert(IST)
    df["date"] = ist.date
    df["mso"] = ist.hour * 60 + ist.minute - OPEN_MINUTE_OF_DAY
    df["min_to_close"] = SESSION_MINUTES - 1 - df["mso"]
    df["decision_time"] = df.index + pd.Timedelta(minutes=1)
    df["fill_trade"] = df["c"].isna()  # no TRADE bar: volume 0, OHLC = last close
    df["fill_quote"] = df["bid"].isna() | df["ask"].isna()  # no BID/ASK bar: last quote carried

    g = df.groupby("date", sort=False)
    for c in ("volume", "n_events", "value"):
        df[c] = df[c].fillna(0.0)
    for c in ("c", "bid", "ask"):
        df[c] = g[c].ffill()
    for c in ("o", "h", "l"):
        df[c] = df[c].fillna(df["c"])

    df["tick_size"] = infer_tick_size(trade["close"])
    valid_q = (df["bid"] > 0) & (df["ask"] > df["bid"])
    df["mid"] = ((df["bid"] + df["ask"]) / 2).where(valid_q)
    df["px"] = df["mid"].fillna(df["c"])
    df["spread_bps"] = ((df["ask"] - df["bid"]) / df["mid"] * 1e4).where(valid_q)
    df["spread_ticks"] = ((df["ask"] - df["bid"]) / df["tick_size"]).where(valid_q).round()

    g = df.groupby("date", sort=False)
    mso = df["mso"]
    logpx = np.log(df["px"])
    df["ret_1m"] = g["px"].transform(lambda s: np.log(s).diff())
    df["_r2"] = df["ret_1m"] ** 2
    df["_hl2"] = np.log(df["h"] / df["l"]).replace([np.inf, -np.inf], np.nan) ** 2
    for w in RV_WINDOWS:
        df[f"rv_{w}m"] = _rv(df["_r2"], df["date"], w)
        df[f"ret_{w}m"] = _diff(logpx, df["date"], mso, w)
    df["parkinson_30m"] = np.sqrt(_roll(df, "_hl2", 30, "mean", WARM_MIN_OBS) / (4 * np.log(2)))
    for w in SPREAD_VAR_WINDOWS:
        df[f"spread_var_{w}m"] = _roll(df, "spread_bps", w, "var", WARM_MIN_OBS)
    df["spread_mean_30m"] = _roll(df, "spread_bps", 30, "mean", WARM_MIN_OBS)
    df["dq_warmup"] = mso < WARMUP_MINUTES

    df["turnover"] = np.where(df["value"] > 0, df["value"], df["c"] * df["volume"])
    g = df.groupby("date", sort=False)
    df["cumvol"] = g["volume"].cumsum()
    df["vwap_run"] = g["turnover"].cumsum() / df["cumvol"].replace(0, np.nan)
    df["dev_vwap_bps"] = (df["px"] / df["vwap_run"] - 1) * 1e4

    # Intraday volume profile from the previous PROFILE_DAYS sessions only (causal).
    pv = df.pivot_table(index="date", columns="mso", values="volume", aggfunc="sum")
    share = pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0)
    prof = share.shift(1).rolling(PROFILE_DAYS, min_periods=PROFILE_MIN_DAYS).mean()
    avg = pv.shift(1).rolling(PROFILE_DAYS, min_periods=PROFILE_MIN_DAYS).mean()
    key = pd.MultiIndex.from_arrays([df["date"], df["mso"]])
    df["prof_share"] = prof.stack().reindex(key).to_numpy()
    df["prof_cum_share"] = prof.cumsum(axis=1).stack().reindex(key).to_numpy()
    df["prof_avg_vol"] = avg.stack().reindex(key).to_numpy()
    df["rel_volume"] = df["volume"] / df["prof_avg_vol"].replace(0, np.nan)
    df["dq_burnin"] = df["date"].map(prof.isna().all(axis=1))  # fewer than PROFILE_MIN_DAYS prior sessions

    # L1 microstructure: tick-derived where ticks exist, bar proxy everywhere.
    if imb_proxy is not None:
        df["imb_bar_proxy"] = imb_proxy.reindex(df.index)
        df["imb_bar_proxy"] = df.groupby("date", sort=False)["imb_bar_proxy"].ffill()
        df["imb_bar_proxy_ema5"] = df.groupby("date", sort=False)["imb_bar_proxy"].transform(
            lambda s: s.ewm(span=5, adjust=False).mean()
        )
    df["log_spread_var_30m"] = np.log(df["spread_var_30m"] + LOG_EPS)
    df["log_rv_30m"] = np.log(df["rv_30m"] + LOG_EPS)
    tf = tick_features(security, paths)
    if len(tf):
        df = df.join(tf, how="left")
        has_ticks = df.groupby("date", sort=False)["n_quote_updates"].transform(lambda s: s.notna().any())
        for c in TICK_FILL_COLS:
            df[c] = df.groupby("date", sort=False)[c].ffill()
        for c in TICK_COUNT_COLS:
            if c in df.columns:
                df[c] = df[c].where(~has_ticks, df[c].fillna(0.0))  # zero-fill only on days that HAVE ticks
        df["ofi_5m"] = _roll(df, "ofi", 5, "sum")
        df["ofi_norm_5m"] = _roll(df, "ofi_norm", 5, "sum")
        df["signed_vol_5m"] = _roll(df, "signed_vol", 5, "sum")
    else:
        has_ticks = pd.Series(False, index=df.index)
    df["has_ticks"] = has_ticks.astype(bool)
    tick_imb = df["l1_imb_mean"] if "l1_imb_mean" in df.columns else pd.Series(np.nan, index=df.index)
    df["l1_imb"] = tick_imb.where(df["has_ticks"], df.get("imb_bar_proxy"))
    df["l1_imb_source"] = np.where(
        df["has_ticks"], "tick", np.where(df["l1_imb"].notna(), "bar_proxy", "none")
    )

    if ctx is not None and len(ctx):
        df = df.join(ctx, how="left")

    if daily is not None and len(daily):
        dl = daily_lagged(daily)
        day_vol = df.groupby("date", sort=False)["volume"].sum()
        df = df.reset_index().merge(dl, on="date", how="left").set_index("time")
        off = 1 - df["date"].map(day_vol) / df["daily_volume_same_day"].replace(0, np.nan)
        df["dq_offbook_frac"] = off.clip(lower=0)  # volume printed outside the 09:15-15:15 bars
        df = df.drop(columns=["daily_volume_same_day"])

    active = df.groupby("date", sort=False)["n_events"].agg(lambda s: int((s > 0).sum()))
    df["dq_partial_day"] = df["date"].map(active < 0.8 * active.median())

    # Evaluation-only regime label from previous-day realised vol terciles (never a model input).
    day_rv = df.groupby("date", sort=False)["rv_30m"].mean().shift(1)
    q33 = day_rv.rolling(60, min_periods=10).quantile(0.33)
    q67 = day_rv.rolling(60, min_periods=10).quantile(0.67)
    lab = pd.Series(
        np.where(
            day_rv.isna() | q33.isna(),
            None,
            np.where(day_rv <= q33, "low", np.where(day_rv >= q67, "high", "mid")),
        ),
        index=day_rv.index,
    )
    df["regime_eval_prevday"] = df["date"].map(lab)

    df = df.drop(columns=["_r2", "_hl2"])
    df.insert(0, "security", security)
    return df.reset_index()


def build_feature_table(
    securities: list[str] | None = None,
    paths: DataPaths | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    paths = paths or DataPaths()
    daily_all = load_daily("equity", paths)
    securities = securities or sorted(load_static("equity", paths)["security"])
    ctx = context_features(paths)
    frames = []
    for sec in securities:
        f = build_security(sec, ctx, daily_all[daily_all["security"] == sec], paths)
        if len(f):
            frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["split"] = assign_split(
        out["date"], out["dq_partial_day"] | out["dq_burnin"], splits or load_splits()
    )
    return out


def manifest(df: pd.DataFrame, paths: DataPaths, episode_columns: list[str], first_mso: int) -> dict:
    """What the final table contains, what was dropped or filled, and train-only statistics."""
    nifty = load_daily("index", paths)
    nifty = nifty.loc[nifty["security"] == "NIFTY Index", "date"]
    expected = set(nifty[(nifty >= df["date"].min()) & (nifty <= df["date"].max())].dt.date)
    sessions = {s: set(g["date"].dt.date) for s, g in df.groupby("security")}
    day = df.drop_duplicates(["security", "date"])
    eligible = (df["split"] != "excluded") & (df["mso"] >= first_mso)
    train = df[eligible & (df["split"] == "train")]
    return {
        "file": FEATURES_FILE,
        "rows": len(df),
        "columns": len(df.columns),
        "securities": sorted(sessions),
        "grid": f"09:15-15:15 IST, {SESSION_MINUTES} min/session",
        "first": str(df["date"].min().date()),
        "last": str(df["date"].max().date()),
        "sessions_by_split": {
            s: g.groupby("security")["date"].nunique().to_dict() for s, g in day.groupby("split")
        },
        "excluded_sessions": {
            s: sorted(str(d.date()) for d in g.loc[g["split"] == "excluded", "date"].unique())
            for s, g in day.groupby("security")
        },
        "dropped_sessions": {s: sorted(str(d) for d in expected - v) for s, v in sessions.items()},
        "filled_share": {
            "fill_trade": round(float(df["fill_trade"].mean()), 5),
            "fill_quote": round(float(df["fill_quote"].mean()), 5),
        },
        "episode_rows": int(eligible.sum()),
        "episode_nan": {c: int(df.loc[eligible, c].isna().sum()) for c in episode_columns},
        "train_stats": {
            c: {
                "mean": float(train[c].mean()),
                "std": float(train[c].std()),
                "p01": float(train[c].quantile(0.01)),
                "p99": float(train[c].quantile(0.99)),
            }
            for c in episode_columns
        },
    }
