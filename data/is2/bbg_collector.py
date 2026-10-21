#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bbg_collector.py  -  Bloomberg BLPAPI data collector for the
"Market-regime-aware execution (Fuzzy PPO / eFNN)" project (NSE equities).

WHAT IT DOES
  probe        -> checks timezone handling, services, event types, real history limits
  validate     -> validates every candidate field mnemonic via //blp/apiflds
  universe     -> validates tickers, pulls Nifty constituents (current + 15y point-in-time)
  static       -> instrument metadata (sector, ISIN, lot size, free float, ...)
  daily        -> 15y daily history (adjusted AND raw) for equities/indices/FX/rates/commodities
  events       -> dividends/corporate actions, earnings dates
  derivs       -> futures + options chain discovery, daily history, greeks snapshot
  bars         -> 1-minute TRADE / BID / ASK bars (as far back as your entitlement allows)
  ticks        -> tick TRADE / BID / ASK with sizes + condition codes (level-1 book)
  live         -> forward recorder: level-1 subscription stream (//blp/mktdata)
  depth        -> forward recorder: market depth (//blp/mktdepthdata), raw capture (needs entitlement)
  consolidate  -> merges chunk files into one parquet per dataset
  features     -> causal feature table (1-min) for the RL environment
  manifest     -> coverage manifest (CSV)      excel -> Excel summary workbook
  all          -> probe, validate, universe, static, daily, events, derivs, bars, ticks,
                  consolidate, features, manifest, excel   (stops cleanly if a data limit is hit)
  bundle       -> zip bbg_data/ for copying to another machine

All timestamps are stored in UTC. NSE session 09:15-15:30 IST = 03:45-10:00 UTC.
Everything is read-only (no order entry).  Nothing here is fabricated: if Bloomberg
does not return a field/period, it is logged in reports/errors.jsonl and left empty.

REQUIREMENTS: Bloomberg Terminal running & logged in on this machine (localhost:8194),
  pip install pandas numpy pyarrow openpyxl tzdata
  pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import blpapi  # type: ignore
    BLPAPI_OK, _BLPAPI_ERR = True, ""
except Exception as _e:  # pragma: no cover
    blpapi, BLPAPI_OK, _BLPAPI_ERR = None, False, repr(_e)

# =============================================================================
# 1. CONFIGURATION  (edit here, or override from a notebook: bbg.OUT = Path(...))
# =============================================================================
HOST, PORT = "localhost", 8194
OUT = Path("bbg_data")

TODAY = date.today()
DAILY_START = date(TODAY.year - 15, TODAY.month, min(TODAY.day, 28))
INTRADAY_LOOKBACK_DAYS = 140     # overwritten by probe results if reports/capabilities.json exists
TICK_LOOKBACK_DAYS = None        # None = as far back as the probe says your terminal serves (~140 calendar days)
BAR_CHUNK_DAYS = 5               # calendar days per intraday bar request
BAR_INTERVAL_MIN = 1             # collect 1-min bars; resample locally to 5/15/30/60
TICK_UNIVERSE_N = 10             # ticks are huge (100k-1M rows per stock-day): first N equities by weight
TICK_SESSION_UTC = ("03:30", "10:05")   # 09:00-15:35 IST: pre-open auction + session + closing, nothing else
USE_TOPY = True                  # fast C-level message conversion (blpapi >= 3.20), falls back automatically
MAX_FIELDS_PER_HIST_REQ = 25     # Bloomberg limit for HistoricalDataRequest
REQUEST_SLEEP = 0.05
MAX_RETRIES = 3
INDEX_FOR_MEMBERS = ["NIFTY Index"]          # constituents become the equity universe
MEMBERSHIP_HISTORY = True                    # point-in-time constituents (survivorship control)
INCLUDE_HIST_MEMBERS_DAILY = True            # also pull daily data for past members
DERIV_UNDERLYING_INDICES = ["NIFTY Index", "NSEBANK Index"]
# Generic (continuous, auto-rolling) futures. FUT_CHAIN is asked on these too, incl. expired contracts.
FUTURES_GENERICS = ["NZ1 Index", "NZ2 Index", "AF1 Index", "AF2 Index"]
DERIV_N_EQUITIES = 5                         # stock futures/options for first N equities
OPT_MAX_DAYS = 45                            # only options expiring within N days
OPT_STRIKE_BAND = 0.10                       # +/-10% around spot
OPT_MAX_PER_UNDERLYING = 300
SESSION_OPEN_IST, SESSION_MINUTES = "09:15", 375
PROFILE_DAYS = 20                            # lookback for intraday volume profile
LIVE_FIELDS = ["BID", "ASK", "BID_SIZE", "ASK_SIZE", "LAST_PRICE", "SIZE_LAST_TRADE", "VOLUME",
               "MKTDATA_EVENT_TYPE", "MKTDATA_EVENT_SUBTYPE", "EVENT_TIME",
               "BID_UPDATE_STAMP_RT", "ASK_UPDATE_STAMP_RT", "TRADE_UPDATE_STAMP_RT"]
# Market-depth topic format is entitlement/version dependent -> check Bloomberg docs / WAPI <GO>.
DEPTH_TOPIC_TEMPLATE = "//blp/mktdepthdata/ticker/{sec}"
DEPTH_FIELDS: list = []
DEPTH_OPTIONS: list = ["type=MBL"]

REFDATA, APIFLDS, MKTDATA, MKTDEPTH = "//blp/refdata", "//blp/apiflds", "//blp/mktdata", "//blp/mktdepthdata"

# ---- Universe seeds (Bloomberg tickers; `universe` step validates them and reports bad ones) ----
STARTER_EQUITIES = [
    "RIL IN Equity", "HDFCB IN Equity", "ICICIBC IN Equity", "AXSB IN Equity", "INFO IN Equity",
    "TCS IN Equity", "HCLT IN Equity", "LT IN Equity", "ITC IN Equity", "HUVR IN Equity",
    "MSIL IN Equity", "MM IN Equity", "TATA IN Equity", "JSTL IN Equity", "ONGC IN Equity",
    "SBIN IN Equity", "BHARTI IN Equity", "SUNP IN Equity", "TTMT IN Equity", "BAF IN Equity",
    "KMB IN Equity", "APNT IN Equity", "NEST IN Equity", "TTAN IN Equity", "UTCEM IN Equity",
    "PWGR IN Equity", "NTPC IN Equity", "COAL IN Equity", "WPRO IN Equity", "TECHM IN Equity",
]
SEED_CLASSES = {
    "index": ["NIFTY Index", "NSEBANK Index", "INVIXN Index", "SENSEX Index", "NSEIT Index",
              "SPX Index", "NDX Index", "VIX Index", "SX5E Index", "NKY Index", "HSI Index"],
    "fx": ["USDINR Curncy", "EURINR Curncy", "DXY Curncy"],
    "rates": ["GIND10YR Index", "USGG10YR Index", "USGG2YR Index"],
    "commodity": ["CO1 Comdty", "CL1 Comdty", "GC1 Comdty"],
}
# More seeds. Anything that does not resolve on your terminal is reported in universe_validation and skipped.
SEED_CLASSES["index"] += ["NSEAUTO Index", "NSEFMCG Index", "NSEPHRM Index", "NSEMET Index", "NSEFIN Index",
                          "NSEMCAP Index", "NSESMCP Index", "NSENEXT50 Index", "BSE500 Index"]
SEED_CLASSES["fx"] += ["USDINR1M Curncy", "USDINRV1M Curncy"]
SEED_CLASSES["rates"] += ["GIND2YR Index", "GIND5YR Index", "MIBOR Index", "RBIRREPO Index"]
CONTEXT_FOR_FEATURES = {"nifty": "NIFTY Index", "vix": "INVIXN Index", "banknifty": "NSEBANK Index"}
INTRADAY_EVENTS = {"equity": ["TRADE", "BID", "ASK"], "futures": ["TRADE", "BID", "ASK"],
                   "index": ["TRADE"], "fx": ["BID", "ASK"]}
TICK_EVENTS = ["TRADE", "BID", "ASK", "AT_TRADE"]

# ---- Candidate fields (validated at runtime; invalid ones are dropped & reported) ----
EQUITY_DAILY = [
    "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_BID", "PX_ASK", "PX_VOLUME", "TURNOVER",
    "EQY_WEIGHTED_AVG_PX", "NUM_TRADES", "EQY_SH_OUT", "CUR_MKT_CAP", "EQY_FREE_FLOAT_PCT",
    "EQY_DVD_YLD_IND", "TOT_RETURN_INDEX_GROSS_DVDS", "TOT_RETURN_INDEX_NET_DVDS",
    "DAY_TO_DAY_TOT_RETURN_GROSS_DVDS", "VOLATILITY_10D", "VOLATILITY_30D", "VOLATILITY_60D",
    "VOLATILITY_90D", "VOLATILITY_260D", "30DAY_IMPVOL_100.0%MNY_DF", "30DAY_IMPVOL_90.0%MNY_DF",
    "30DAY_IMPVOL_110.0%MNY_DF", "60DAY_IMPVOL_100.0%MNY_DF", "PUT_CALL_VOLUME_RATIO_CUR_DAY",
    "PUT_CALL_OPEN_INTEREST_RATIO", "PE_RATIO", "PX_TO_BOOK_RATIO",
    "VOLUME_AVG_20D", "VOLUME_AVG_30D", "EQY_TURNOVER_REALTIME", "BID_ASK_SPREAD",
    "PX_CLOSE_1D", "VOLUME_TDY", "HIGH_52WEEK", "LOW_52WEEK", "BETA_RAW_OVERRIDABLE",
]
RAW_PRICE_FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]   # unadjusted copy
DAILY_FIELDS = {
    "equity": EQUITY_DAILY,
    "index": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME", "TURNOVER", "CUR_MKT_CAP",
              "TOT_RETURN_INDEX_GROSS_DVDS", "VOLATILITY_30D", "VOLATILITY_90D", "PE_RATIO",
              "PX_TO_BOOK_RATIO", "EQY_DVD_YLD_IND", "30DAY_IMPVOL_100.0%MNY_DF"],
    "fx": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_BID", "PX_ASK", "VOLATILITY_30D"],
    "rates": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"],
    "commodity": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME", "OPEN_INT"],
    "futures": ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_SETTLE", "PX_BID", "PX_ASK",
                "PX_VOLUME", "OPEN_INT", "FUT_AGGTE_VOL", "FUT_AGGTE_OPEN_INT"],
    "options": ["PX_LAST", "PX_BID", "PX_ASK", "PX_VOLUME", "OPEN_INT", "IVOL_MID", "DELTA_MID",
                "GAMMA_MID", "VEGA_MID", "THETA_MID"],
}
STATIC_FIELDS = {
    "equity": ["NAME", "SHORT_NAME", "SECURITY_TYP", "ID_BB_GLOBAL", "ID_ISIN", "ID_SEDOL1", "TICKER",
               "EXCH_CODE", "EQY_PRIM_EXCH", "CRNCY", "COUNTRY_ISO", "GICS_SECTOR_NAME",
               "GICS_INDUSTRY_GROUP_NAME", "GICS_INDUSTRY_NAME", "GICS_SUB_INDUSTRY_NAME",
               "INDUSTRY_SECTOR", "INDUSTRY_GROUP", "INDUSTRY_SUBGROUP", "EQY_SH_OUT", "EQY_FLOAT",
               "EQY_FREE_FLOAT_PCT", "CUR_MKT_CAP", "TRADING_LOT_SIZE", "TICK_SIZE", "EQY_INIT_PO_DT",
               "ANNOUNCEMENT_DT", "EXPECTED_REPORT_DT", "LATEST_ANNOUNCEMENT_DT", "MARKET_STATUS",
               "PX_LAST", "PX_BID", "PX_ASK", "BID_SIZE", "ASK_SIZE", "VOLUME"],
    "index": ["NAME", "ID_BB_GLOBAL", "COUNTRY_ISO", "CRNCY", "EXCH_CODE", "PX_LAST"],
    "fx": ["NAME", "ID_BB_GLOBAL", "PX_LAST"], "rates": ["NAME", "ID_BB_GLOBAL", "PX_LAST"],
    "commodity": ["NAME", "ID_BB_GLOBAL", "PX_LAST"],
    "futures": ["NAME", "ID_BB_GLOBAL", "FUT_CONT_SIZE", "FUT_TICK_SIZE", "LAST_TRADEABLE_DT", "FUT_FIRST_TRADE_DT",
                "FUT_DLV_DT_FIRST", "UNDL_SPOT_TICKER", "OPEN_INT", "PX_LAST"],
    "options": ["NAME", "ID_BB_GLOBAL", "OPT_STRIKE_PX", "OPT_PUT_CALL", "OPT_EXPIRE_DT",
                "OPT_UNDL_TICKER", "OPT_CONT_SIZE", "OPT_EXER_TYP"],
}
EVENT_BULK_FIELDS = ["DVD_HIST_ALL", "EARN_ANN_DT_TIME_HIST_WITH_EPS"]
OPTION_SNAPSHOT_FIELDS = ["PX_LAST", "PX_BID", "PX_ASK", "BID_SIZE", "ASK_SIZE", "VOLUME", "OPEN_INT",
                          "IVOL_MID", "IVOL_BID", "IVOL_ASK", "DELTA_MID", "GAMMA_MID", "VEGA_MID",
                          "THETA_MID", "RHO_MID", "OPT_UNDL_PX"]

# =============================================================================
# 2. SMALL UTILITIES
# =============================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def date_chunks(start: date, end: date, n: int):
    d = start
    while d <= end:
        e = min(d + timedelta(days=n - 1), end)
        yield d, e
        d = e + timedelta(days=1)


def has_weekday(a: date, b: date) -> bool:
    d = a
    while d <= b:
        if d.weekday() < 5:
            return True
        d += timedelta(days=1)
    return False


def utc_dt(d: date, end_of_day: bool = False) -> datetime:
    """Naive datetime that Bloomberg interprets as UTC."""
    return datetime(d.year, d.month, d.day, 23, 59, 59) if end_of_day else datetime(d.year, d.month, d.day)


def prev_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def write_df(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_many(paths) -> pd.DataFrame:
    frames = [read_df(p) for p in paths]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def mark_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".empty").write_text("")


def chunk_done(path: Path, end_day: date) -> bool:
    """A finished chunk is skipped on re-run; chunks touching the last 2 days are always refetched."""
    if end_day >= TODAY - timedelta(days=2):
        return False
    return path.exists() or path.with_suffix(".empty").exists()


def _stringify_nested(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            vals = df[c].dropna()
            if vals.map(lambda v: isinstance(v, (list, dict))).any():
                df[c] = df[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
            vals = df[c].dropna()
            if len(vals) and not vals.map(lambda v: isinstance(v, (int, float, np.number, bool))).all():
                if not vals.map(lambda v: isinstance(v, (pd.Timestamp, datetime, date))).all():
                    df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    return df


def coerce_numeric(df: pd.DataFrame, keep=("security", "date", "time", "class", "event", "type")) -> pd.DataFrame:
    """Object columns that are really numbers become float; genuinely textual columns become str."""
    for c in df.columns:
        if c in keep or df[c].dtype != object:
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() == df[c].notna().sum():
            df[c] = num
        else:
            df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    return df


ERRORS_PATH = lambda: OUT / "reports" / "errors.jsonl"   # noqa: E731


def log_error(kind: str, **kw) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **kw}
    p = ERRORS_PATH()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


# =============================================================================
# 3. BLPAPI SESSION + REQUEST LAYER
# =============================================================================
SESSION = None
_OPEN_SERVICES: set = set()
_CID = [0]


class BBGRequestFailure(Exception):
    """Bloomberg answered with RequestFailure (entitlement, bad request, ...). Not retried."""


class BBGLimitReached(Exception):
    """Bloomberg reported a daily/monthly data limit. Everything stops so the shared terminal is not hammered."""


def connect():
    global SESSION
    if SESSION is not None:
        return SESSION
    if not BLPAPI_OK:
        raise RuntimeError(f"blpapi not importable: {_BLPAPI_ERR}. Install Bloomberg's blpapi in this Python.")
    o = blpapi.SessionOptions()
    o.setServerHost(HOST)
    o.setServerPort(PORT)
    s = blpapi.Session(o)
    if not s.start():
        raise RuntimeError(f"Could not start Bloomberg session at {HOST}:{PORT}. Is the Terminal logged in?")
    SESSION = s
    open_service(REFDATA)
    return s


def close():
    global SESSION
    if SESSION is not None:
        try:
            SESSION.stop()
        except Exception:
            pass
    SESSION = None
    _OPEN_SERVICES.clear()


def open_service(name: str):
    if SESSION is None:
        connect()
    if name not in _OPEN_SERVICES:
        if not SESSION.openService(name):
            raise RuntimeError(f"Could not open service {name} (not enabled for this account?)")
        _OPEN_SERVICES.add(name)
    return SESSION.getService(name)


def el_to_py(el):
    """Recursively convert a blpapi Element into plain Python (dict / list / scalar)."""
    try:
        if el.isNull():
            return None
    except Exception:
        pass
    try:
        if el.isArray():
            out = []
            for i in range(el.numValues()):
                try:
                    out.append(el_to_py(el.getValueAsElement(i)))
                except Exception:
                    out.append(el.getValue(i))
            return out
        if el.isComplexType():
            return {str(el.getElement(i).name()): el_to_py(el.getElement(i)) for i in range(el.numElements())}
        return el.getValue()
    except Exception:
        return str(el)


_TOP_KEYS = {"securityData", "barData", "tickData", "fieldData", "responseError", "reason", "categories"}


def msg_to_dict(msg) -> dict:
    """Message -> dict. Uses the fast C-level Message.toPy() (blpapi >= 3.20) when it gives the expected shape,
    otherwise walks the element tree in Python."""
    mtype = str(msg.messageType())
    d = None
    if USE_TOPY and hasattr(msg, "toPy"):
        try:
            d = msg.toPy()
            if isinstance(d, dict) and len(d) == 1 and mtype in d and isinstance(d[mtype], dict):
                d = d[mtype]                                   # some versions wrap the body in its type name
            if not (isinstance(d, dict) and (set(d) & _TOP_KEYS or mtype == "RequestFailure")):
                d = None
        except Exception:
            d = None
    if d is None:
        d = el_to_py(msg.asElement())
    if not isinstance(d, dict):
        d = {"value": d}
    d = dict(d)
    d["_type"] = mtype
    return d


def _cid_value(c):
    try:
        return c.value()
    except Exception:
        return c


def _is_mine(msg, cid) -> bool:
    try:
        ids = msg.correlationIds()
    except Exception:
        try:
            ids = [msg.correlationId()]
        except Exception:
            return True
    want = _cid_value(cid)
    return any(_cid_value(c) == want for c in ids)


def _limit_hit(d: dict):
    """Returns the error text if a message says a Bloomberg data limit was reached, else None."""
    errs = []
    if isinstance(d.get("responseError"), dict):
        errs.append(d["responseError"])
    sds = d.get("securityData")
    for sd in (sds if isinstance(sds, list) else [sds] if isinstance(sds, dict) else []):
        if isinstance(sd, dict) and isinstance(sd.get("securityError"), dict):
            errs.append(sd["securityError"])
    for e in errs:
        cat = str(e.get("category", "")).upper()
        sub = str(e.get("subcategory", "")).upper()
        msg = str(e.get("message", "")).upper()
        if cat == "LIMIT" or "LIMIT_REACHED" in sub or "DAILY LIMIT" in msg or "MONTHLY LIMIT" in msg or "DAILY_CAPACITY" in sub:
            return json.dumps(e, default=str)[:400]
    return None


def _send_once(req, timeout: float) -> list:
    """Sends one request and collects only the messages that belong to it.
    `timeout` is an IDLE timeout: it resets every time a chunk of the response arrives, so a large tick
    download that keeps streaming is never killed halfway."""
    _CID[0] += 1
    cid = blpapi.CorrelationId(_CID[0])
    SESSION.sendRequest(req, correlationId=cid)
    out, last = [], time.time()
    req_status = getattr(blpapi.Event, "REQUEST_STATUS", None)
    while True:
        ev = SESSION.nextEvent(500)
        et = ev.eventType()
        if et in (blpapi.Event.PARTIAL_RESPONSE, blpapi.Event.RESPONSE) or (req_status is not None and et == req_status):
            mine = False
            for msg in ev:
                if not _is_mine(msg, cid):
                    continue                                  # stale reply to an earlier, timed-out request
                mine = True
                d = msg_to_dict(msg)
                if d.get("_type") == "RequestFailure" or (req_status is not None and et == req_status):
                    if d.get("_type") == "RequestFailure":
                        reason = json.dumps(d, default=str)[:600]
                        if "LIMIT" in reason.upper() and "REACHED" in reason.upper():
                            raise BBGLimitReached(reason)
                        raise BBGRequestFailure(reason)
                    continue
                lim = _limit_hit(d)
                if lim:
                    raise BBGLimitReached(lim)
                out.append(d)
            if mine:
                last = time.time()
                if et == blpapi.Event.RESPONSE:
                    return out
        elif et == blpapi.Event.SESSION_STATUS:
            for msg in ev:
                if str(msg.messageType()) in ("SessionTerminated", "SessionConnectionDown"):
                    raise ConnectionError(str(msg))
        if time.time() - last > timeout:
            try:
                SESSION.cancel(cid)
            except Exception:
                pass
            raise TimeoutError(f"no data for {timeout}s")


def send(make_request, timeout: float = 180) -> list:
    """Builds + sends a request (re-built on every retry); returns list of message dicts."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            res = _send_once(make_request(), timeout)
            time.sleep(REQUEST_SLEEP)
            return res
        except (BBGRequestFailure, BBGLimitReached):
            raise
        except Exception as e:  # timeouts / connection hiccups
            last = e
            log(f"  retry {attempt + 1}/{MAX_RETRIES} after {e!r}")
            if isinstance(e, ConnectionError):
                close()                                       # session died: reconnect on next call
                try:
                    connect()
                except Exception as ce:
                    last = ce
            time.sleep(2 ** attempt)
    raise last  # type: ignore


# =============================================================================
# 4. PURE PARSERS (dict -> rows).  Unit-testable without Bloomberg.
# =============================================================================
def _err_rows(sec, sd: dict, errs: list) -> None:
    if sd.get("securityError"):
        errs.append({"security": sec, "type": "securityError", "detail": json.dumps(sd["securityError"], default=str)})
    for fe in sd.get("fieldExceptions") or []:
        info = fe.get("errorInfo") or {}
        errs.append({"security": sec, "type": "fieldException", "field": fe.get("fieldId"),
                     "detail": info.get("message") or json.dumps(info, default=str)})


def parse_reference(msgs):
    """-> (rows[scalar fields], bulk[long rows for array fields], errors)"""
    rows, bulk, errs = [], [], []
    for m in msgs:
        sds = m.get("securityData") or []
        sds = sds if isinstance(sds, list) else [sds]
        for sd in sds:
            sec = sd.get("security")
            _err_rows(sec, sd, errs)
            row = {"security": sec}
            for k, v in (sd.get("fieldData") or {}).items():
                if isinstance(v, list):
                    for i, item in enumerate(v):
                        rec = {"security": sec, "field": k, "row": i}
                        rec.update(item if isinstance(item, dict) else {"value": item})
                        bulk.append(rec)
                else:
                    row[k] = v
            rows.append(row)
    return rows, bulk, errs


def parse_historical(msgs):
    rows, errs = [], []
    for m in msgs:
        sds = m.get("securityData") or []
        sds = sds if isinstance(sds, list) else [sds]
        for sd in sds:
            sec = sd.get("security")
            _err_rows(sec, sd, errs)
            for item in sd.get("fieldData") or []:
                rows.append({"security": sec, **item})
    return rows, errs


def parse_bars(msgs, security: str, event: str):
    rows, errs = [], []
    for m in msgs:
        if m.get("responseError"):
            errs.append({"security": security, "type": "responseError", "detail": json.dumps(m["responseError"], default=str)})
        bd = m.get("barData")
        if bd:
            for bar in bd.get("barTickData") or []:
                rows.append({**bar, "security": security, "event": event})
    return rows, errs


def parse_ticks(msgs, security: str):
    rows, errs = [], []
    for m in msgs:
        if m.get("responseError"):
            errs.append({"security": security, "type": "responseError", "detail": json.dumps(m["responseError"], default=str)})
        td = m.get("tickData")
        if td:
            for t in td.get("tickData") or []:
                rows.append({**t, "security": security})
    return rows, errs


def bulk_member(rec: dict):
    """Ticker + weight from an index-membership bulk row. INDX_MWEIGHT uses 'Member Ticker and Exchange Code' /
    'Percentage Weight', INDX_MWEIGHT_HIST uses 'Index Member' / 'Percent Weight' -> match by name pattern."""
    mem = wt = None
    for k, v in rec.items():
        kl = str(k).lower()
        if mem is None and ("member" in kl or "ticker" in kl) and isinstance(v, str) and v.strip():
            mem = v.strip()
        elif wt is None and "weight" in kl:
            wt = v
    return mem, wt


def with_yellow(s: str, default="Equity") -> str:
    s = str(s).strip()
    return s if s.endswith(("Equity", "Index", "Comdty", "Curncy", "Corp", "Govt")) else f"{s} {default}"


OPT_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})\s+([CP])\s?(\d+(?:\.\d+)?)")


def parse_option_desc(desc: str):
    m = OPT_RE.search(desc)
    if not m:
        return None
    exp = pd.to_datetime(m.group(1), format="%m/%d/%y", errors="coerce")
    if pd.isna(exp):
        exp = pd.to_datetime(m.group(1), errors="coerce")
    return {"expiry": exp, "cp": m.group(2), "strike": float(m.group(3))}


# =============================================================================
# 5. GENERIC REQUEST WRAPPERS
# =============================================================================
def ref_request(securities, fields, overrides=None):
    def mk():
        r = open_service(REFDATA).createRequest("ReferenceDataRequest")
        for s in securities:
            r.getElement("securities").appendValue(s)
        for f in fields:
            r.getElement("fields").appendValue(f)
        for k, v in (overrides or {}).items():
            o = r.getElement("overrides").appendElement()
            o.setElement("fieldId", k)
            o.setElement("value", str(v))
        return r

    return parse_reference(send(mk))


def hist_request(sec, fields, start: date, end: date, adjust=True, periodicity="DAILY") -> pd.DataFrame:
    """Daily/weekly/... history for ONE security; fields chunked to Bloomberg's limit and merged."""
    merged = None
    for fchunk in chunks(fields, MAX_FIELDS_PER_HIST_REQ):
        def mk():
            r = open_service(REFDATA).createRequest("HistoricalDataRequest")
            r.getElement("securities").appendValue(sec)
            for f in fchunk:
                r.getElement("fields").appendValue(f)
            r.set("startDate", start.strftime("%Y%m%d"))
            r.set("endDate", end.strftime("%Y%m%d"))
            r.set("periodicitySelection", periodicity)
            r.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")
            r.set("adjustmentFollowDPDF", False)
            r.set("adjustmentSplit", bool(adjust))
            r.set("adjustmentAbnormal", bool(adjust))
            r.set("adjustmentNormal", False)
            return r

        try:
            rows, errs = parse_historical(send(mk, timeout=240))
        except BBGRequestFailure as e:
            log_error("hist_request_failure", security=sec, detail=str(e))
            continue
        for e in errs:
            log_error("hist_error", **e)
        df = pd.DataFrame(rows)
        if df.empty or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = coerce_numeric(df.drop(columns=["security"]))
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged.insert(0, "security", sec)
    return merged.sort_values("date").reset_index(drop=True)


def bars_request(sec: str, event: str, start_dt: datetime, end_dt: datetime, interval: int = 1) -> pd.DataFrame:
    def mk():
        r = open_service(REFDATA).createRequest("IntradayBarRequest")
        r.set("security", sec)
        r.set("eventType", event)
        r.set("interval", int(interval))
        r.set("startDateTime", start_dt)      # naive datetime == UTC for Bloomberg
        r.set("endDateTime", end_dt)
        r.set("gapFillInitialBar", False)
        return r

    rows, errs = parse_bars(send(mk), sec, event)
    for e in errs:
        log_error("bar_error", event=event, **e)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = to_utc(df["time"])
    return df


def ticks_request(sec: str, events, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    def mk():
        r = open_service(REFDATA).createRequest("IntradayTickRequest")
        r.set("security", sec)
        for e in events:
            r.getElement("eventTypes").appendValue(e)
        r.set("startDateTime", start_dt)
        r.set("endDateTime", end_dt)
        r.set("includeConditionCodes", True)
        r.set("includeExchangeCodes", True)
        r.set("includeNonPlottableEvents", True)
        return r

    rows, errs = parse_ticks(send(mk, timeout=300), sec)
    for e in errs:
        log_error("tick_error", **e)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = to_utc(df["time"])
    return df


# =============================================================================
# 6. DATA CATALOG  (single source of truth for "what are we extracting")
# =============================================================================
DATA_CATALOG = [
    # dataset, source request/service, granularity, history, content, used for
    ("universe/index_membership", "ReferenceData INDX_MWEIGHT / INDX_MWEIGHT_HIST", "quarterly snapshots", "15y",
     "Nifty constituents + weights, point-in-time", "survivorship control, liquid universe"),
    ("reference/static_*", "ReferenceData", "static", "now",
     "name, ISIN, FIGI, exchange, GICS/industry sector, free float, shares out, lot/tick size", "stock buckets, tick size"),
    ("daily/daily_equity", "HistoricalData", "daily", "15y",
     "OHLC (adjusted+raw), volume, turnover, daily VWAP, closing bid/ask, mkt cap, free float, TR index, "
     "realised vol 10-260d, implied vol 30/60d (90/100/110% moneyness), put/call ratios, P/E, P/B", "regime context, ADV, vol-of-vol"),
    ("daily/daily_index|fx|rates|commodity", "HistoricalData", "daily", "15y",
     "Nifty/BankNifty/Sensex/sector/global indices, India VIX, USDINR, DXY, G-sec yields, Brent/WTI/Gold", "macro regime inputs"),
    ("daily/daily_futures|options", "HistoricalData", "daily", "contract life",
     "generic NZ1/NZ2/AF1/AF2 + live and expired contracts: OHLC, settle, bid/ask, volume, OI; option IV + greeks",
     "basis, hedging flow, IV regime"),
    ("intraday_bars/bars_1m_TRADE", "IntradayBar TRADE", "1-minute", "probe-limited (~140d typical)",
     "open, high, low, close, volume, numEvents, value", "realised vol, VWAP, volume profile, benchmarks"),
    ("intraday_bars/bars_1m_BID|ASK", "IntradayBar BID/ASK", "1-minute", "probe-limited",
     "last/high/low bid and ask per minute", "spread, spread variance, mid price"),
    ("ticks/<sec>/<date>", "IntradayTick TRADE/BID/ASK", "tick", "probe-limited (~140 calendar days)",
     "price, size, condition codes, exchange code (best bid/offer with sizes)", "L1 order-book imbalance, OFI, microprice, signed volume"),
    ("live/* (forward recorder)", "//blp/mktdata subscription", "event", "from the day you start",
     "bid/ask/sizes/last/volume + exchange update stamps, recv-time in ns", "latency-aware live replay, longer L1 history"),
    ("depth/* (forward recorder)", "//blp/mktdepthdata subscription", "event", "from the day you start",
     "raw market-by-level messages (schema discovered on first run)", "true multi-level LOB (if entitled)"),
    ("events/DVD_HIST_ALL", "ReferenceData bulk", "event", "full",
     "dividends / corporate actions with declared, ex, record, pay dates", "adjustment checks, event flags"),
    ("events/EARN_ANN_DT_TIME_HIST_WITH_EPS", "ReferenceData bulk", "event", "full",
     "earnings announcement dates/times and EPS", "event-risk regime"),
    ("snapshots/options_*", "ReferenceData", "snapshot (run daily to accumulate)", "from first run",
     "bid/ask, volume, OI, IV, greeks for filtered option chain", "implied-vol / gamma context"),
    ("features/features_1m", "derived locally", "1-minute, causal", "as bars",
     "returns, RV 5/15/30/60m, Parkinson vol, spread & spread variance, L1 imbalance, OFI, microprice, signed "
     "volume, VWAP deviation, volume-profile & relative volume, time-of-day, Nifty/VIX intraday context, "
     "lagged daily context, evaluation-only regime label", "state vector for PPO + eFNN"),
]
NOT_AVAILABLE = [
    "Historical multi-level order book (only forward-recorded via //blp/mktdepthdata, if entitled)",
    "Historical news text (not exposed through standard BLPAPI refdata)",
    "Your own order/fill history (must be simulated; labelled as simulated)",
]


# =============================================================================
# 7. PROBE / FIELD VALIDATION / UNIVERSE
# =============================================================================
def _caps_path():
    return OUT / "reports" / "capabilities.json"


def load_caps() -> dict:
    return load_json(_caps_path(), {}) or {}


def probe(sample="RIL IN Equity"):
    """Empirically measures timezone behaviour, accepted event types, services and history limits."""
    connect()
    caps: dict = {"sample_security": sample, "run_at": datetime.now(timezone.utc).isoformat()}
    for svc in (APIFLDS, MKTDATA, MKTDEPTH, "//blp/instruments"):
        try:
            open_service(svc)
            caps[f"service_{svc}"] = True
        except Exception as e:
            caps[f"service_{svc}"] = False
            caps[f"service_{svc}_error"] = str(e)
    log(f"services: { {k: v for k, v in caps.items() if k.startswith('service_') and not k.endswith('_error')} }")

    # --- timezone check: full-day 1m TRADE bars on last weekday that returns data
    d = prev_weekday(TODAY - timedelta(days=1))
    tz_ok, tries = None, 0
    while tries < 6:
        df = bars_request(sample, "TRADE", utc_dt(d), utc_dt(d, True), 1)
        if len(df):
            first = df["time"].min().tz_convert("Asia/Kolkata")
            last = df["time"].max().tz_convert("Asia/Kolkata")
            caps["tz_check"] = {"date": str(d), "first_bar_ist": str(first), "last_bar_ist": str(last), "n_bars": len(df)}
            tz_ok = 8 * 60 + 45 <= first.hour * 60 + first.minute <= 9 * 60 + 30
            caps["tz_check"]["looks_correct"] = bool(tz_ok)
            log(f"timezone check {d}: first bar {first:%H:%M} IST, last {last:%H:%M} IST -> {'OK' if tz_ok else 'UNEXPECTED, inspect!'}")
            break
        d = prev_weekday(d - timedelta(days=1))
        tries += 1
    if tz_ok is None:
        log("timezone check: no intraday bars returned (entitlement or market closed days?)")

    # --- event type acceptance (bars and ticks)
    ref_day = date.fromisoformat(caps["tz_check"]["date"]) if "tz_check" in caps else prev_weekday(TODAY - timedelta(days=2))
    s, e = datetime(ref_day.year, ref_day.month, ref_day.day, 3, 45), datetime(ref_day.year, ref_day.month, ref_day.day, 4, 15)
    caps["bar_events"], caps["tick_events"] = {}, {}
    for ev in ["TRADE", "BID", "ASK", "BEST_BID", "BEST_ASK", "MIDPRICE"]:
        try:
            df = bars_request(sample, ev, s, e, 1)
            caps["bar_events"][ev] = {"accepted": True, "rows": int(len(df))}
        except Exception as ex:
            caps["bar_events"][ev] = {"accepted": False, "error": str(ex)[:200]}
    for ev in ["TRADE", "BID", "ASK", "BID_BEST", "ASK_BEST", "BEST_BID", "BEST_ASK", "MID_PRICE", "AT_TRADE"]:
        try:
            df = ticks_request(sample, [ev], s, e)
            caps["tick_events"][ev] = {"accepted": True, "rows": int(len(df))}
        except Exception as ex:
            caps["tick_events"][ev] = {"accepted": False, "error": str(ex)[:200]}
    log(f"bar events: { {k: v['accepted'] and v['rows'] > 0 for k, v in caps['bar_events'].items()} }")
    log(f"tick events: { {k: v['accepted'] and v['rows'] > 0 for k, v in caps['tick_events'].items()} }")

    # --- history depth
    def depth(offsets, fn):
        res = {}
        for off in offsets:
            day = prev_weekday(TODAY - timedelta(days=off))
            got = False
            for dd in (day, prev_weekday(day - timedelta(days=1))):
                try:
                    got = fn(dd)
                except Exception:
                    got = False
                if got:
                    break
            res[off] = bool(got)
        return res

    bar_depth = depth([14, 30, 60, 90, 120, 140, 160, 200, 365],
                      lambda dd: len(bars_request(sample, "TRADE", utc_dt(dd), utc_dt(dd, True), 1)) > 0)
    tick_depth = depth([1, 5, 10, 20, 30, 60, 90, 120, 140, 160],
                       lambda dd: len(ticks_request(sample, ["TRADE"], datetime(dd.year, dd.month, dd.day, 3, 45),
                                                    datetime(dd.year, dd.month, dd.day, 3, 50))) > 0)
    caps["bar_history_probe"], caps["tick_history_probe"] = bar_depth, tick_depth
    caps["bar_lookback_days"] = max([k for k, v in bar_depth.items() if v], default=0)
    caps["tick_lookback_days"] = max([k for k, v in tick_depth.items() if v], default=0)
    caps["good_bar_events"] = [k for k, v in caps["bar_events"].items() if v.get("accepted") and v.get("rows", 0) > 0]
    caps["good_tick_events"] = [k for k, v in caps["tick_events"].items() if v.get("accepted") and v.get("rows", 0) > 0]
    save_json(caps, _caps_path())
    log(f"history: 1m bars ~{caps['bar_lookback_days']}d, ticks ~{caps['tick_lookback_days']}d  (saved {_caps_path()})")
    return caps


def all_candidate_fields() -> list:
    s = set()
    for lst in list(DAILY_FIELDS.values()) + list(STATIC_FIELDS.values()) + [EVENT_BULK_FIELDS, OPTION_SNAPSHOT_FIELDS]:
        s.update(lst)
    return sorted(s)


def validate_fields(fields=None) -> pd.DataFrame:
    """Checks each mnemonic against //blp/apiflds. Result -> reference/field_validation.parquet"""
    connect()
    fields = fields or all_candidate_fields()
    open_service(APIFLDS)
    rows = []
    for fchunk in chunks(fields, 50):
        def mk():
            r = open_service(APIFLDS).createRequest("FieldInfoRequest")
            for f in fchunk:
                r.append("id", f)
            r.set("returnFieldDocumentation", False)
            return r
        try:
            msgs = send(mk)
        except Exception as e:
            log_error("field_validation_failure", detail=str(e))
            continue
        for m in msgs:
            for fd in m.get("fieldData") or []:
                fi = fd.get("fieldInfo")
                if fi:
                    rows.append({"field": fd.get("id"), "valid": True, "mnemonic": fi.get("mnemonic"),
                                 "description": fi.get("description"), "datatype": fi.get("datatype"),
                                 "ftype": fi.get("ftype"), "error": None})
                else:
                    rows.append({"field": fd.get("id"), "valid": False, "mnemonic": None, "description": None,
                                 "datatype": None, "ftype": None, "error": json.dumps(fd.get("fieldError"), default=str)})
    df = pd.DataFrame(rows)
    if len(df):
        write_df(_stringify_nested(df), OUT / "reference" / "field_validation.parquet")
        bad = df[~df["valid"]]["field"].tolist()
        log(f"field validation: {int(df['valid'].sum())} valid, {len(bad)} invalid {bad[:15]}")
    return df


def valid_fields(cands) -> list:
    p = OUT / "reference" / "field_validation.parquet"
    if not p.exists():
        return list(cands)
    fv = read_df(p)
    bad = set(fv.loc[~fv["valid"].astype(bool), "field"])
    kept = [f for f in cands if f not in bad]
    return kept or list(cands)


def build_universe():
    connect()
    ref = OUT / "reference"
    current: list = []
    # ---- constituents of configured indices
    memb_rows = []
    for idx in INDEX_FOR_MEMBERS:
        for fld in ("INDX_MWEIGHT", "INDX_MEMBERS"):          # INDX_MEMBERS = fallback without weights
            try:
                _, bulk, errs = ref_request([idx], [fld])
            except BBGRequestFailure as e:
                log_error("universe_failure", index=idx, field=fld, detail=str(e))
                continue
            for e in errs:
                log_error("universe_error", **e)
            got = 0
            for rec in bulk:
                mem, wt = bulk_member(rec)
                if mem:
                    got += 1
                    memb_rows.append({"index": idx, "asof": str(TODAY), "member": with_yellow(mem), "weight": wt})
            if got:
                break
    # ---- point-in-time constituents (survivorship control)
    if MEMBERSHIP_HISTORY:
        qdates, d = [], date(DAILY_START.year, 3, 31)
        while d < TODAY:
            qdates.append(d)
            m = d.month + 3
            d = date(d.year + (m - 1) // 12, (m - 1) % 12 + 1, 28)
            d = (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()
        for idx in INDEX_FOR_MEMBERS:
            for qd in qdates:
                try:
                    _, bulk, errs = ref_request([idx], ["INDX_MWEIGHT_HIST"], {"END_DATE_OVERRIDE": qd.strftime("%Y%m%d")})
                except BBGRequestFailure as e:
                    log_error("membership_failure", detail=str(e))
                    continue
                for rec in bulk:
                    mem, wt = bulk_member(rec)
                    if mem:
                        memb_rows.append({"index": idx, "asof": str(qd), "member": with_yellow(mem), "weight": wt})
    memb = pd.DataFrame(memb_rows)
    if len(memb):
        write_df(memb, ref / "index_membership_history.parquet")
    cur = memb[memb["asof"] == str(TODAY)].copy() if len(memb) else pd.DataFrame(columns=["member", "weight"])
    if len(cur):
        cur["w"] = pd.to_numeric(cur["weight"], errors="coerce").fillna(0)
        cur = cur.sort_values("w", ascending=False)
    cur_members = cur["member"].drop_duplicates().tolist()
    hist_members = sorted(set(memb["member"]) - set(cur_members)) if len(memb) else []

    cand = {"equity": list(dict.fromkeys(cur_members + STARTER_EQUITIES)),
            "equity_hist_only": [m for m in hist_members if m not in STARTER_EQUITIES] if INCLUDE_HIST_MEMBERS_DAILY else [],
            **{k: list(v) for k, v in SEED_CLASSES.items()},
            "futures": list(FUTURES_GENERICS)}
    uni, unresolved, checks = {}, [], []
    for cls, secs in cand.items():
        ok = []
        for batch in chunks(secs, 50):
            try:
                rows, _, errs = ref_request(batch, ["ID_BB_GLOBAL", "NAME", "SECURITY_TYP"])
            except BBGRequestFailure as e:
                log_error("universe_failure", detail=str(e))
                rows = []
            good = {r["security"]: r for r in rows if r.get("ID_BB_GLOBAL")}
            for s in batch:
                (ok if s in good else unresolved).append(s)
                checks.append({"class": cls, "security": s, "resolved": s in good,
                               "name": good.get(s, {}).get("NAME"), "figi": good.get(s, {}).get("ID_BB_GLOBAL")})
        uni[cls] = ok
    uni["futures"] = uni.pop("futures", [])
    uni["unresolved"] = unresolved
    save_json(uni, ref / "universe.json")
    write_df(pd.DataFrame(checks), ref / "universe_validation.parquet")
    log("universe: " + ", ".join(f"{k}={len(v)}" for k, v in uni.items()))
    if unresolved:
        log(f"  {len(unresolved)} identifiers did NOT resolve (fix tickers in Terminal): {unresolved[:12]}")
    return uni


def load_universe(limit=None) -> dict:
    uni = load_json(OUT / "reference" / "universe.json")
    if uni is None:
        raise RuntimeError("Run the `universe` step first.")
    if limit:
        uni = {k: (v[:limit] if k != "unresolved" else v) for k, v in uni.items()}
    return uni


def _field_key(cls: str) -> str:
    return "equity" if cls in ("equity", "equity_hist_only") else cls


# =============================================================================
# 8. STATIC / DAILY / EVENTS
# =============================================================================
def collect_static(limit=None, classes=None):
    connect()
    uni = load_universe(limit)
    for cls, secs in uni.items():
        if cls == "unresolved" or not secs or _field_key(cls) not in STATIC_FIELDS:
            continue
        if classes and cls not in classes:
            continue
        fields = valid_fields(STATIC_FIELDS[_field_key(cls)])
        rows = []
        for batch in chunks(secs, 25):
            try:
                r, _, errs = ref_request(batch, fields)
            except BBGRequestFailure as e:
                log_error("static_failure", cls=cls, detail=str(e))
                continue
            rows += r
            for e in errs:
                log_error("static_error", **e)
        df = pd.DataFrame(rows)
        if len(df):
            df["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            write_df(_stringify_nested(df), OUT / "reference" / f"static_{cls}.parquet")
            log(f"static_{cls}: {len(df)} rows x {len(df.columns)} cols")


def collect_daily(limit=None, classes=None):
    connect()
    uni = load_universe(limit)
    for cls, secs in uni.items():
        if cls == "unresolved" or not secs or _field_key(cls) not in DAILY_FIELDS:
            continue
        if classes and cls not in classes:
            continue
        key = _field_key(cls)
        fields = valid_fields(DAILY_FIELDS[key])
        log(f"daily[{cls}] {len(secs)} securities x {len(fields)} fields")
        for i, sec in enumerate(secs, 1):
            part = OUT / "daily" / "_parts" / f"{cls}__{safe(sec)}.parquet"
            if part.exists() or part.with_suffix(".empty").exists():
                continue
            df = hist_request(sec, fields, DAILY_START, TODAY, adjust=True)
            if df.empty:
                mark_empty(part)
                continue
            if key == "equity":
                raw = hist_request(sec, valid_fields(RAW_PRICE_FIELDS), DAILY_START, TODAY, adjust=False)
                if not raw.empty:
                    raw = raw.drop(columns=["security"]).rename(columns={c: f"{c}_RAW" for c in raw.columns if c != "date"})
                    df = df.merge(raw, on="date", how="left")
            df["class"] = cls
            write_df(df, part)
            if i % 10 == 0:
                log(f"  daily[{cls}] {i}/{len(secs)}")
    consolidate_daily()


def collect_events(limit=None):
    connect()
    uni = load_universe(limit)
    secs = uni.get("equity", []) + uni.get("equity_hist_only", [])
    fields = valid_fields(EVENT_BULK_FIELDS)
    acc: dict = {}
    for sec in secs:
        try:
            _, bulk, errs = ref_request([sec], fields)
        except BBGRequestFailure as e:
            log_error("events_failure", security=sec, detail=str(e))
            continue
        for e in errs:
            log_error("events_error", **e)
        for rec in bulk:
            acc.setdefault(rec["field"], []).append(rec)
    for field, rows in acc.items():
        df = pd.DataFrame(rows).dropna(axis=1, how="all")
        write_df(_stringify_nested(df), OUT / "events" / f"{field}.parquet")
        log(f"events/{field}: {len(df)} rows")


# =============================================================================
# 9. DERIVATIVES
# =============================================================================
def discover_derivatives(limit=None):
    connect()
    uni = load_universe(limit)
    unders = DERIV_UNDERLYING_INDICES + uni.get("equity", [])[:DERIV_N_EQUITIES]
    generics = [g for g in uni.get("futures", []) if g in FUTURES_GENERICS] or \
        [g for g in FUTURES_GENERICS if g not in uni.get("unresolved", [])]
    spot = {}
    rows, _, _ = ref_request(unders, ["PX_LAST"])
    for r in rows:
        spot[r["security"]] = r.get("PX_LAST")
    fut_rows, opt_rows = [], []
    # (security asked, field, sink, overrides). Expired futures give a continuous history of past contracts.
    asks = [(u, "FUT_CHAIN", fut_rows, {"INCLUDE_EXPIRED_CONTRACTS": "Y"}) for u in unders + [g for g in generics if g.endswith("1 Index")]] + \
           [(u, "OPT_CHAIN", opt_rows, None) for u in unders]
    for u, field, sink, ov in asks:
        try:
            _, bulk, errs = ref_request([u], [field], ov)
        except BBGRequestFailure as e:
            log_error("chain_failure", underlying=u, field=field, detail=str(e))
            continue
        for e in errs:
            log_error("chain_error", **e)
        for rec in bulk:
            desc = rec.get("Security Description")
            if desc:
                sink.append({"underlying": u, "security": with_yellow(desc, "Index" if "Index" in u else "Equity"), "desc": desc})
    fut = pd.DataFrame(fut_rows)
    opt = pd.DataFrame(opt_rows)
    selected_opts = []
    if len(opt):
        parsed = opt["desc"].map(parse_option_desc)
        opt = opt[parsed.notna()].copy()
        info = pd.DataFrame(list(parsed[parsed.notna()]), index=opt.index)
        opt = pd.concat([opt, info], axis=1)
        opt["spot"] = opt["underlying"].map(spot)
        horizon = pd.Timestamp(TODAY) + pd.Timedelta(days=OPT_MAX_DAYS)
        sel = opt[(opt["expiry"] >= pd.Timestamp(TODAY)) & (opt["expiry"] <= horizon)]
        sel = sel[(sel["strike"] / sel["spot"] - 1).abs() <= OPT_STRIKE_BAND]
        sel = sel.assign(dist=(sel["strike"] / sel["spot"] - 1).abs()).sort_values("dist")
        sel = sel.groupby("underlying", group_keys=False).head(OPT_MAX_PER_UNDERLYING)
        selected_opts = sel["security"].tolist()
        write_df(opt.drop(columns=["dist"], errors="ignore"), OUT / "reference" / "options_chain_all.parquet")
    if len(fut):
        write_df(fut, OUT / "reference" / "futures_chain.parquet")
    if len(fut):
        fut = fut.drop_duplicates("security")
    chain = fut["security"].tolist() if len(fut) else []
    uni["futures"] = list(dict.fromkeys(generics + chain))
    uni["options"] = selected_opts
    save_json(uni, OUT / "reference" / "universe.json")
    log(f"derivatives: {len(uni['futures'])} futures, {len(selected_opts)} filtered options")
    return uni


def collect_derivatives(limit=None):
    discover_derivatives(limit)
    collect_daily(limit, classes=["futures", "options"])
    uni = load_universe()
    if uni.get("options") or uni.get("futures"):
        snapshot_options()
    collect_static(limit, classes=["futures", "options"])


def snapshot_options():
    """One-off snapshot of quotes + IV + greeks for the filtered option chain. Run daily to build history."""
    connect()
    uni = load_universe()
    secs = uni.get("options", []) + uni.get("futures", [])
    if not secs:
        return
    fields = valid_fields(OPTION_SNAPSHOT_FIELDS)
    rows = []
    for batch in chunks(secs, 50):
        r, _, errs = ref_request(batch, fields)
        rows += r
        for e in errs:
            log_error("snapshot_error", **e)
    df = pd.DataFrame(rows)
    if len(df):
        ts = datetime.now(timezone.utc)
        df["snapshot_utc"] = ts.isoformat()
        write_df(_stringify_nested(df), OUT / "snapshots" / f"options_{ts:%Y%m%d_%H%M%S}.parquet")
        log(f"options snapshot: {len(df)} contracts")


# =============================================================================
# 10. INTRADAY BARS + TICKS
# =============================================================================
def _effective_lookbacks():
    caps = load_caps()
    bars = caps.get("bar_lookback_days") or INTRADAY_LOOKBACK_DAYS
    ticks = caps.get("tick_lookback_days") or TICK_LOOKBACK_DAYS or 30
    if TICK_LOOKBACK_DAYS:
        ticks = min(ticks, TICK_LOOKBACK_DAYS)
    return min(bars, 400), ticks


def _bar_jobs(uni, limit=None):
    caps = load_caps()
    good = set(caps.get("good_bar_events", [])) or {"TRADE", "BID", "ASK"}
    jobs = []
    for cls, evs in INTRADAY_EVENTS.items():
        secs = uni.get(cls, [])[:limit] if limit else uni.get(cls, [])
        if cls == "futures":                 # continuous generics only; dated contracts mostly expired
            secs = [x for x in secs if x in FUTURES_GENERICS]
        for sec in secs:
            for ev in evs:
                if ev in good:
                    jobs.append((cls, sec, ev))
    return jobs


def collect_bars(limit=None):
    connect()
    uni = load_universe()
    lb, _ = _effective_lookbacks()
    start, end = TODAY - timedelta(days=lb), TODAY
    jobs = _bar_jobs(uni, limit)
    log(f"bars: {len(jobs)} (security,event) jobs, window {start}..{end}, {BAR_INTERVAL_MIN}-min")
    for j, (cls, sec, ev) in enumerate(jobs, 1):
        for a, b in date_chunks(start, end, BAR_CHUNK_DAYS):
            if not has_weekday(a, b):
                continue
            part = OUT / "intraday_bars" / f"{BAR_INTERVAL_MIN}m" / ev / safe(sec) / f"{a:%Y%m%d}_{b:%Y%m%d}.parquet"
            if chunk_done(part, b):
                continue
            try:
                df = bars_request(sec, ev, utc_dt(a), utc_dt(b, True), BAR_INTERVAL_MIN)
            except BBGRequestFailure as e:
                log_error("bars_failure", security=sec, event=ev, chunk=f"{a}..{b}", detail=str(e))
                continue
            if df.empty:
                mark_empty(part)
            else:
                df["class"] = cls
                write_df(df, part)
        if j % 10 == 0:
            log(f"  bars {j}/{len(jobs)} jobs")
    consolidate_bars()


def collect_ticks(limit=None):
    connect()
    uni = load_universe()
    _, lb = _effective_lookbacks()
    caps = load_caps()
    events = [e for e in TICK_EVENTS if e in set(caps.get("good_tick_events", TICK_EVENTS))] or ["TRADE", "BID", "ASK"]
    extra = [e for e in ("BID_BEST", "ASK_BEST") if e in set(caps.get("good_tick_events", []))]
    secs = uni.get("equity", [])
    n = limit or TICK_UNIVERSE_N
    secs = secs[:n] if n else secs
    log(f"ticks: {len(secs)} securities x {lb} days, events {events + extra}")
    d0 = TODAY - timedelta(days=lb)
    for sec in secs:
        for k in range(lb + 1):
            d = d0 + timedelta(days=k)
            if d.weekday() >= 5:
                continue
            part = OUT / "ticks" / safe(sec) / f"{d:%Y%m%d}.parquet"
            if chunk_done(part, d):
                continue
            try:
                (h0, m0), (h1, m1) = (map(int, x.split(":")) for x in TICK_SESSION_UTC)
                df = ticks_request(sec, events + extra, datetime(d.year, d.month, d.day, h0, m0),
                                   datetime(d.year, d.month, d.day, h1, m1))
            except BBGRequestFailure as e:
                log_error("ticks_failure", security=sec, date=str(d), detail=str(e))
                continue
            if df.empty:
                mark_empty(part)
            else:
                write_df(_stringify_nested(df), part)
        log(f"  ticks done: {sec}")


# =============================================================================
# 11. LIVE / DEPTH FORWARD RECORDERS
# =============================================================================
def _flush(buf: list, folder: Path, tag: str):
    if not buf:
        return
    df = _stringify_nested(pd.DataFrame(buf))
    p = folder / f"{tag}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}.parquet"
    write_df(df, p)
    log(f"  flushed {len(df)} rows -> {p.name}")
    buf.clear()


def record_stream(securities, service, topic_fn, fields, options, folder: Path, minutes=None, flush_every=5000):
    connect()
    open_service(service)
    subs = blpapi.SubscriptionList()
    for s in securities:
        subs.add(topic_fn(s), fields=fields, options=options, correlationId=blpapi.CorrelationId(s))
    SESSION.subscribe(subs)
    until = time.time() + minutes * 60 if minutes else None
    buf, status = [], []
    log(f"recording {len(securities)} securities from {service} (Ctrl+C to stop) -> {folder}")
    try:
        while until is None or time.time() < until:
            ev = SESSION.nextEvent(500)
            recv = time.time_ns()
            et = ev.eventType()
            if et == blpapi.Event.SUBSCRIPTION_DATA:
                for msg in ev:
                    d = el_to_py(msg.asElement())
                    d = d if isinstance(d, dict) else {"value": d}
                    d.update({"_sec": msg.correlationIds()[0].value(), "_recv_ns": recv, "_type": str(msg.messageType())})
                    buf.append(d)
            elif et in (blpapi.Event.SUBSCRIPTION_STATUS, blpapi.Event.SESSION_STATUS, blpapi.Event.SERVICE_STATUS):
                for msg in ev:
                    status.append({"recv_ns": recv, "type": str(msg.messageType()), "msg": str(msg)[:800]})
            if len(buf) >= flush_every:
                _flush(buf, folder, "stream")
    except KeyboardInterrupt:
        log("stopped by user")
    finally:
        _flush(buf, folder, "stream")
        if status:
            write_df(pd.DataFrame(status), folder / f"status_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.parquet")
        try:
            SESSION.unsubscribe(subs)
        except Exception:
            pass


def record_live(limit=None, minutes=None):
    uni = load_universe()
    secs = (uni.get("equity", [])[: (limit or TICK_UNIVERSE_N or 20)] + ["NIFTY Index", "INVIXN Index"])
    record_stream(secs, MKTDATA, lambda s: s, LIVE_FIELDS, [], OUT / "live", minutes)


def record_depth(limit=None, minutes=None):
    uni = load_universe()
    secs = uni.get("equity", [])[: (limit or 10)]
    record_stream(secs, MKTDEPTH, lambda s: DEPTH_TOPIC_TEMPLATE.format(sec=s), DEPTH_FIELDS, DEPTH_OPTIONS,
                  OUT / "depth", minutes, flush_every=2000)


# =============================================================================
# 12. CONSOLIDATION
# =============================================================================
def consolidate_daily():
    parts = sorted((OUT / "daily" / "_parts").glob("*.parquet"))
    by_cls: dict = {}
    for p in parts:   # current + past index members share one file; the `class` column tells them apart
        by_cls.setdefault(_field_key(p.name.split("__")[0]), []).append(p)
    for cls, ps in by_cls.items():
        df = read_many(ps)
        if df.empty:
            continue
        df = coerce_numeric(df.sort_values(["security", "date"]).drop_duplicates(["security", "date"], keep="last"))
        write_df(df, OUT / "daily" / f"daily_{cls}.parquet")
        log(f"daily_{cls}.parquet: {len(df):,} rows, {df['security'].nunique()} securities")


def consolidate_bars():
    base = OUT / "intraday_bars" / f"{BAR_INTERVAL_MIN}m"
    if not base.exists():
        return
    for evdir in sorted(p for p in base.iterdir() if p.is_dir()):
        df = read_many(sorted(evdir.rglob("*.parquet")))
        if df.empty:
            continue
        df["time"] = to_utc(df["time"])
        df = df.sort_values(["security", "time"]).drop_duplicates(["security", "time"], keep="last")
        write_df(df, OUT / "intraday_bars" / f"bars_{BAR_INTERVAL_MIN}m_{evdir.name}.parquet")
        log(f"bars_{BAR_INTERVAL_MIN}m_{evdir.name}.parquet: {len(df):,} rows, {df['security'].nunique()} securities")


def consolidate_all():
    consolidate_daily()
    consolidate_bars()


# =============================================================================
# 13. FEATURES  (all causal: a row is valid at decision_time = bar_start + 1 min)
# =============================================================================
def _grid_for_days(days) -> pd.DatetimeIndex:
    parts = []
    for d in days:
        o = pd.Timestamp(f"{d} {SESSION_OPEN_IST}", tz="Asia/Kolkata").tz_convert("UTC")
        parts.append(pd.date_range(o, periods=SESSION_MINUTES, freq="min"))
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def tick_features_day(ticks: pd.DataFrame) -> pd.DataFrame:
    """Level-1 microstructure features per minute from one day of tick data. Index = minute start (UTC)."""
    if ticks is None or ticks.empty:
        return pd.DataFrame()
    t = ticks.copy()
    t["time"] = to_utc(t["time"])
    t = t.sort_values("time", kind="mergesort")
    q = t[t["type"].isin(["BID", "ASK"])]
    out = []
    if len(q):
        isb = (q["type"] == "BID").to_numpy()
        qq = pd.DataFrame({"time": q["time"].to_numpy(),
                           "bid": np.where(isb, q["value"], np.nan), "bid_sz": np.where(isb, q["size"], np.nan),
                           "ask": np.where(~isb, q["value"], np.nan), "ask_sz": np.where(~isb, q["size"], np.nan)})
        qq[["bid", "bid_sz", "ask", "ask_sz"]] = qq[["bid", "bid_sz", "ask", "ask_sz"]].ffill()
        qq = qq.dropna()
        qq = qq[(qq["bid"] > 0) & (qq["ask"] >= qq["bid"])].reset_index(drop=True)
        if len(qq):
            tot = (qq["bid_sz"] + qq["ask_sz"]).replace(0, np.nan)
            qq["mid"] = (qq["bid"] + qq["ask"]) / 2
            qq["imb"] = (qq["bid_sz"] - qq["ask_sz"]) / tot
            qq["micro"] = (qq["ask"] * qq["bid_sz"] + qq["bid"] * qq["ask_sz"]) / tot
            qq["micro_dev_bps"] = (qq["micro"] - qq["mid"]) / qq["mid"] * 1e4
            db, da = qq["bid"].diff(), qq["ask"].diff()
            bq, aq = qq["bid_sz"], qq["ask_sz"]
            e_b = np.where(db > 0, bq, np.where(db == 0, bq - bq.shift(), -bq.shift()))
            e_a = np.where(da < 0, -aq, np.where(da == 0, -(aq - aq.shift()), aq.shift()))
            qq["ofi"] = np.nan_to_num(e_b + e_a)            # Cont-Kukanov-Stoikov order-flow imbalance
            qq["minute"] = qq["time"].dt.floor("min")
            agg = qq.groupby("minute").agg(l1_imb_last=("imb", "last"), l1_imb_mean=("imb", "mean"),
                                           microprice_dev_bps=("micro_dev_bps", "last"), ofi=("ofi", "sum"),
                                           n_quote_updates=("bid", "size"), bid_sz_last=("bid_sz", "last"),
                                           ask_sz_last=("ask_sz", "last"))
            out.append(agg)
            tr = t[t["type"] == "TRADE"][["time", "value", "size"]].copy()
            if len(tr):
                tr = pd.merge_asof(tr, qq[["time", "bid", "ask"]], on="time", direction="backward")
                mid = (tr["bid"] + tr["ask"]) / 2
                sign = np.sign(tr["value"] - mid)
                tick_rule = np.sign(tr["value"].diff()).replace(0, np.nan).ffill()
                tr["sign"] = sign.where(sign != 0, tick_rule).fillna(0)
                tr["minute"] = tr["time"].dt.floor("min")
                tr["buy"] = np.where(tr["sign"] > 0, tr["size"], 0.0)
                tr["sell"] = np.where(tr["sign"] < 0, tr["size"], 0.0)
                ta = tr.groupby("minute").agg(n_trades=("value", "size"), tick_volume=("size", "sum"),
                                              buy_vol=("buy", "sum"), sell_vol=("sell", "sum"))
                ta["signed_vol"] = ta["buy_vol"] - ta["sell_vol"]
                ta["trade_imb"] = ta["signed_vol"] / (ta["buy_vol"] + ta["sell_vol"]).replace(0, np.nan)
                out.append(ta)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, axis=1)
    res.index.name = "time"
    return res


def _rolling_by_day(df: pd.DataFrame, col: str, w: int, fn: str) -> pd.Series:
    return df.groupby("date", sort=False)[col].transform(
        lambda s: getattr(s.rolling(w, min_periods=max(2, w // 3)), fn)())


def context_frame(trade_bars: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Intraday market context (Nifty return/vol, VIX level, BankNifty return) indexed by bar start (UTC)."""
    cols = {}
    for key, sec in names.items():
        b = trade_bars[trade_bars["security"] == sec]
        if b.empty:
            continue
        b = b.drop_duplicates("time").set_index(to_utc(b["time"]).rename("time")).sort_index()
        c = b["close"].astype(float)
        ist_day = c.index.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d")
        lr = np.log(c).groupby(ist_day).diff()
        if key == "vix":
            cols["ctx_vix"] = c
            cols["ctx_vix_chg_30m"] = c.groupby(ist_day).diff(30)
        else:
            cols[f"ctx_{key}_ret_1m"] = lr
            cols[f"ctx_{key}_ret_30m"] = np.log(c).groupby(ist_day).diff(30)
            if key == "nifty":
                cols["ctx_nifty_rv_30m"] = np.sqrt((lr ** 2).groupby(ist_day).transform(lambda s: s.rolling(30, min_periods=10).sum()))
    return pd.DataFrame(cols) if cols else pd.DataFrame()


def daily_lagged_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-day features from DAILY data, shifted by one trading day so they are known before the session opens."""
    d = daily.sort_values("date").copy()
    px = pd.to_numeric(d["PX_LAST"], errors="coerce")
    vol = pd.to_numeric(d.get("PX_VOLUME", pd.Series(np.nan, index=d.index)), errors="coerce")
    ret = px.pct_change()
    f = pd.DataFrame({"date": d["date"].dt.strftime("%Y-%m-%d"),
                      "d_ret_1d": ret,
                      "d_vol_20d": ret.rolling(20, min_periods=10).std() * np.sqrt(252),
                      "d_adv_20d": vol.rolling(20, min_periods=10).mean(),
                      "d_range_pct": (pd.to_numeric(d.get("PX_HIGH"), errors="coerce") - pd.to_numeric(d.get("PX_LOW"), errors="coerce")) / px,
                      "d_amihud_20d": (ret.abs() / (px * vol).replace(0, np.nan)).rolling(20, min_periods=10).mean()})
    for extra in ("30DAY_IMPVOL_100.0%MNY_DF", "VOLATILITY_30D", "EQY_FREE_FLOAT_PCT"):
        if extra in d.columns:
            f["d_" + safe(extra)] = pd.to_numeric(d[extra], errors="coerce").to_numpy()
    cols = [c for c in f.columns if c != "date"]
    f[cols] = f[cols].shift(1)
    return f


def features_one(sec, trade, bid=None, ask=None, tickf=None, ctx=None, daily=None) -> pd.DataFrame:
    if trade is None or trade.empty:
        return pd.DataFrame()
    t = trade.copy()
    t["time"] = to_utc(t["time"])
    t = t.drop_duplicates("time").set_index("time").sort_index()
    df = pd.DataFrame({"o": t["open"], "h": t["high"], "l": t["low"], "c": t["close"], "volume": t["volume"],
                       "n_events": t.get("numEvents", pd.Series(np.nan, index=t.index)),
                       "value": t.get("value", pd.Series(np.nan, index=t.index))}).astype(float)
    for name, b in (("bid", bid), ("ask", ask)):
        if b is not None and len(b):
            bb = b.copy()
            bb["time"] = to_utc(bb["time"])
            df[name] = bb.drop_duplicates("time").set_index("time").sort_index()["close"].astype(float)
        else:
            df[name] = np.nan
    ist = df.index.tz_convert("Asia/Kolkata")
    days = sorted(set(ist.strftime("%Y-%m-%d")))
    grid = _grid_for_days(days)
    df = df.reindex(grid)
    df.index.name = "time"
    ist = df.index.tz_convert("Asia/Kolkata")
    df["date"] = ist.strftime("%Y-%m-%d")
    df["mso"] = ist.hour * 60 + ist.minute - 555                # minutes since open
    g = df.groupby("date", sort=False)
    for c in ("volume", "n_events", "value"):
        df[c] = df[c].fillna(0.0)
    for c in ("c", "bid", "ask"):
        df[c] = g[c].ffill()
    for c in ("o", "h", "l"):
        df[c] = df[c].fillna(df["c"])
    valid_q = (df["bid"] > 0) & (df["ask"] >= df["bid"])
    df["mid"] = ((df["bid"] + df["ask"]) / 2).where(valid_q)
    df["px"] = df["mid"].fillna(df["c"])
    df["spread_bps"] = ((df["ask"] - df["bid"]) / df["mid"] * 1e4).where(valid_q)
    g = df.groupby("date", sort=False)
    df["ret_1m"] = g["px"].transform(lambda s: np.log(s).diff())
    df["ret2"] = df["ret_1m"] ** 2
    df["hl2"] = np.log(df["h"] / df["l"]).replace([np.inf, -np.inf], np.nan) ** 2
    for w in (5, 15, 30, 60):
        df[f"rv_{w}m"] = np.sqrt(_rolling_by_day(df, "ret2", w, "sum"))
        df[f"ret_{w}m"] = g["px"].transform(lambda s, w=w: np.log(s).diff(w))
    df["parkinson_30m"] = np.sqrt(_rolling_by_day(df, "hl2", 30, "mean") / (4 * np.log(2)))
    for w in (15, 30, 60):
        df[f"spread_var_{w}m"] = _rolling_by_day(df, "spread_bps", w, "var")
    df["spread_mean_30m"] = _rolling_by_day(df, "spread_bps", 30, "mean")
    # ---- volume / VWAP
    df["turnover_est"] = np.where(df["value"] > 0, df["value"], df["c"] * df["volume"])
    g = df.groupby("date", sort=False)
    df["cumvol"] = g["volume"].cumsum()
    cum_turn = g["turnover_est"].cumsum()
    df["vwap_run"] = (cum_turn / df["cumvol"].replace(0, np.nan))
    df["dev_vwap_bps"] = (df["px"] / df["vwap_run"] - 1) * 1e4
    pv = df.pivot(index="date", columns="mso", values="volume")
    share = pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0)
    prof = share.shift(1).rolling(PROFILE_DAYS, min_periods=3).mean()          # previous days only
    volavg = pv.shift(1).rolling(PROFILE_DAYS, min_periods=3).mean()
    idx = pd.MultiIndex.from_arrays([df["date"], df["mso"]])
    df["prof_share"] = prof.stack().reindex(idx).to_numpy()
    df["prof_avg_vol"] = volavg.stack().reindex(idx).to_numpy()
    df["prof_cum_share"] = prof.cumsum(axis=1).stack().reindex(idx).to_numpy()
    df["rel_volume"] = df["volume"] / df["prof_avg_vol"].replace(0, np.nan)
    # ---- time features
    df["min_to_close"] = SESSION_MINUTES - 1 - df["mso"]
    df["dow"] = ist.dayofweek
    df["is_month_end_week"] = (ist.day >= 24).astype(int)
    df["decision_time"] = df.index + pd.Timedelta(minutes=1)
    # ---- tick-derived microstructure
    if tickf is not None and len(tickf):
        tf = tickf.copy()
        if tf.index.tz is None:
            tf.index = tf.index.tz_localize("UTC")
        df = df.join(tf, how="left")
        for c in ("l1_imb_last", "l1_imb_mean", "microprice_dev_bps", "bid_sz_last", "ask_sz_last"):
            if c in df.columns:
                df[c] = df.groupby("date", sort=False)[c].ffill()
        cnt_cols = [c for c in ("ofi", "n_quote_updates", "n_trades", "tick_volume", "buy_vol", "sell_vol", "signed_vol")
                    if c in df.columns]
        # zero-fill only on days that HAVE tick coverage; days without ticks stay NaN (never fake "no flow")
        has_ticks = df.groupby("date", sort=False)["n_quote_updates"].transform(lambda s: s.notna().any()) \
            if "n_quote_updates" in df.columns else pd.Series(True, index=df.index)
        for c in cnt_cols:
            df[c] = df[c].where(~has_ticks.astype(bool), df[c].fillna(0.0))
        if "ofi" in df.columns:
            df["ofi_5m"] = _rolling_by_day(df, "ofi", 5, "sum")
            df["signed_vol_5m"] = _rolling_by_day(df, "signed_vol", 5, "sum")
    # ---- market context (same-bar index data are observable at decision_time)
    if ctx is not None and len(ctx):
        df = df.join(ctx, how="left")
    # ---- lagged daily context
    if daily is not None and len(daily):
        dl = daily_lagged_features(daily)
        df = df.reset_index().merge(dl, on="date", how="left").set_index("time")
    # ---- evaluation-only regime label from previous-day realised vol (never use as model input)
    day_rv = df.groupby("date", sort=False)["rv_30m"].mean()
    prev = day_rv.shift(1)
    q33 = prev.rolling(60, min_periods=10).quantile(0.33)
    q67 = prev.rolling(60, min_periods=10).quantile(0.67)
    lab = pd.Series(np.where(prev.isna() | q33.isna(), None, np.where(prev <= q33, "low", np.where(prev >= q67, "high", "mid"))), index=day_rv.index)
    df["regime_eval_prevday"] = df["date"].map(lab)
    df = df.drop(columns=["ret2", "hl2"])
    df.insert(0, "security", sec)
    return df.reset_index()


def build_features(limit=None):
    base = OUT / "intraday_bars"
    ptrade = base / f"bars_{BAR_INTERVAL_MIN}m_TRADE.parquet"
    if not ptrade.exists():
        raise RuntimeError("No consolidated TRADE bars. Run bars + consolidate first.")
    tb = read_df(ptrade)
    pb, pa = base / f"bars_{BAR_INTERVAL_MIN}m_BID.parquet", base / f"bars_{BAR_INTERVAL_MIN}m_ASK.parquet"
    bb = read_df(pb) if pb.exists() else None
    ab = read_df(pa) if pa.exists() else None
    uni = load_universe()
    eq = [s for s in uni.get("equity", []) if s in set(tb["security"])]
    eq = eq[:limit] if limit else eq
    ctx = context_frame(tb, CONTEXT_FOR_FEATURES)
    pdaily = OUT / "daily" / "daily_equity.parquet"
    daily_all = read_df(pdaily) if pdaily.exists() else None
    outs = []
    for sec in eq:
        tdir = OUT / "ticks" / safe(sec)
        tickf = None
        if tdir.exists():
            parts = [tick_features_day(read_df(p)) for p in sorted(tdir.glob("*.parquet"))]
            parts = [p for p in parts if len(p)]
            tickf = pd.concat(parts) if parts else None
        daily = daily_all[daily_all["security"] == sec] if daily_all is not None else None
        f = features_one(sec, tb[tb["security"] == sec],
                         bb[bb["security"] == sec] if bb is not None else None,
                         ab[ab["security"] == sec] if ab is not None else None, tickf, ctx, daily)
        if len(f):
            outs.append(f)
            log(f"features {sec}: {len(f):,} rows")
    if outs:
        res = pd.concat(outs, ignore_index=True)
        write_df(_stringify_nested(res), OUT / "features" / "features_1m.parquet")
        log(f"features_1m.parquet: {len(res):,} rows x {len(res.columns)} cols")


# =============================================================================
# 14. MANIFEST + EXCEL
# =============================================================================
def _parquet_rows(p: Path):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return len(read_df(p))


def build_manifest() -> pd.DataFrame:
    rows = []
    for p in sorted(OUT.rglob("*.parquet")):
        if "_parts" in p.parts:
            continue
        rel = p.relative_to(OUT)
        if rel.parts[0] in ("ticks", "live", "depth", "snapshots"):
            continue
        if rel.parts[0] == "intraday_bars" and len(rel.parts) > 2:
            continue                                   # per-chunk part files; consolidated files are listed
        try:
            n = _parquet_rows(p)
        except Exception:
            n = None
        rows.append({"file": str(rel), "rows": n, "size_mb": round(p.stat().st_size / 1e6, 3)})
    for top in ("ticks", "live", "depth", "snapshots"):
        files = [p for p in (OUT / top).rglob("*.parquet")] if (OUT / top).exists() else []
        files = [p for p in files if not p.name.startswith("status_")]
        if files:
            rows.append({"file": f"{top}/**/*.parquet ({len(files)} files)",
                         "rows": sum((_parquet_rows(p) or 0) for p in files),
                         "size_mb": round(sum(p.stat().st_size for p in files) / 1e6, 3)})
    man = pd.DataFrame(rows)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT / "reports" / "manifest.csv", index=False)
    log(f"manifest: {len(man)} datasets")
    return man


def coverage_table() -> pd.DataFrame:
    rows = []
    specs = [(p, "date") for p in sorted((OUT / "daily").glob("daily_*.parquet"))] + \
            [(p, "time") for p in sorted((OUT / "intraday_bars").glob("bars_*.parquet"))]
    for p, tc in specs:
        df = read_df(p)
        if df.empty or "security" not in df.columns:
            continue
        df[tc] = pd.to_datetime(df[tc], utc=(tc == "time"))
        for sec, g in df.groupby("security"):
            rows.append({"dataset": p.name, "security": sec, "rows": len(g), "first": g[tc].min().isoformat(), "last": g[tc].max().isoformat(),
                         "avg_null_pct": round(float(g.drop(columns=["security", tc]).isna().mean().mean() * 100), 2)})
    return pd.DataFrame(rows)


def export_excel(path: Path | None = None):
    path = Path(path or OUT / "reports" / "bloomberg_data_summary.xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)

    def tz_naive(df):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                if getattr(df[c].dt, "tz", None) is not None:
                    df[c] = df[c].dt.tz_localize(None)
            elif df[c].dtype == object:
                df[c] = df[c].map(lambda v: v.tz_localize(None) if isinstance(v, pd.Timestamp) and v.tzinfo is not None else v)
        return df

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(DATA_CATALOG, columns=["dataset", "source", "granularity", "history", "content", "used_for"]).to_excel(xw, sheet_name="catalog", index=False)
        pd.DataFrame({"not_available_via_blpapi": NOT_AVAILABLE}).to_excel(xw, sheet_name="not_available", index=False)
        for name, p in (("field_validation", OUT / "reference" / "field_validation.parquet"),
                        ("universe_validation", OUT / "reference" / "universe_validation.parquet")):
            if p.exists():
                tz_naive(_stringify_nested(read_df(p))).to_excel(xw, sheet_name=name, index=False)
        caps = load_caps()
        if caps:
            pd.DataFrame([{"key": k, "value": json.dumps(v, default=str)} for k, v in caps.items()]).to_excel(xw, sheet_name="capabilities", index=False)
        man = build_manifest()
        if len(man):
            man.to_excel(xw, sheet_name="manifest", index=False)
        cov = coverage_table()
        if len(cov):
            tz_naive(cov).to_excel(xw, sheet_name="coverage", index=False)
        ep = ERRORS_PATH()
        if ep.exists():
            errs = pd.read_json(ep, lines=True)
            summ = errs.groupby("kind").size().rename("count").reset_index()
            summ.to_excel(xw, sheet_name="error_summary", index=False)
            tz_naive(errs.tail(5000)).to_excel(xw, sheet_name="errors_recent", index=False)
    log(f"excel summary -> {path}")
    return path


def bundle_for_mac(path: Path | None = None, include_parts: bool = False) -> Path:
    """Zips bbg_data/ (without the resumable _parts / per-chunk files unless asked) for copying to another machine."""
    import zipfile
    path = Path(path or OUT.parent / f"bbg_data_{datetime.now():%Y%m%d_%H%M}.zip")
    n, size = 0, 0
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in sorted(OUT.rglob("*")):
            if not p.is_file() or p.suffix == ".empty":
                continue
            rel = p.relative_to(OUT)
            if not include_parts and ("_parts" in rel.parts or (rel.parts[0] == "intraday_bars" and len(rel.parts) > 2)):
                continue
            z.write(p, Path(OUT.name) / rel)
            n += 1
            size += p.stat().st_size
    log(f"bundle: {n} files, {size / 1e6:,.1f} MB raw -> {path} ({path.stat().st_size / 1e6:,.1f} MB zipped)")
    return path


# =============================================================================
# 15. ORCHESTRATION + ESTIMATE + CLI
# =============================================================================
def estimate(limit=None):
    uni = load_universe(limit)
    lb, tl = _effective_lookbacks()
    n_daily = sum(len(v) * (math.ceil(len(DAILY_FIELDS[_field_key(k)]) / MAX_FIELDS_PER_HIST_REQ) + (1 if _field_key(k) == "equity" else 0))
                  for k, v in uni.items() if k != "unresolved" and _field_key(k) in DAILY_FIELDS)
    n_bars = len(_bar_jobs(uni, limit)) * math.ceil(lb / BAR_CHUNK_DAYS)
    n_ticks = min(limit or TICK_UNIVERSE_N or 10 ** 9, len(uni.get("equity", []))) * int(tl * 5 / 7)
    log(f"~requests: daily {n_daily}, bars {n_bars}, ticks {n_ticks}  (Bloomberg daily/monthly data limits apply)")
    return {"daily": n_daily, "bars": n_bars, "ticks": n_ticks}


STEPS = ["probe", "validate", "universe", "static", "daily", "events", "derivs", "bars", "ticks",
         "consolidate", "features", "manifest", "excel"]


def run_all(limit=None, steps=None, skip=()):
    """Runs every step in order. A failing step is logged and the next one runs; a Bloomberg data-limit
    message stops everything immediately. Safe to re-run: finished chunks are skipped."""
    done, failed = [], []
    for s in (steps or STEPS):
        if s in skip:
            continue
        t0 = time.time()
        try:
            run_step(s, limit)
            done.append(s)
            log(f"--- {s} finished in {(time.time() - t0) / 60:.1f} min")
        except BBGLimitReached as e:
            log(f"!! Bloomberg data limit reached during '{s}': {e}\n   Stopping. Re-run tomorrow; finished chunks are kept.")
            log_error("limit_reached", step=s, detail=str(e))
            failed.append(s)
            break
        except Exception as e:
            log(f"!! step {s} failed: {e!r}")
            log_error("step_failed", step=s, detail=traceback.format_exc()[-1500:])
            failed.append(s)
    log(f"done: {done}   failed/stopped: {failed}")
    return done, failed


def run_step(name, limit=None, minutes=None):
    log(f"=== {name} ===")
    {"probe": lambda: probe(), "validate": lambda: validate_fields(), "universe": lambda: build_universe(),
     "static": lambda: collect_static(limit), "daily": lambda: collect_daily(limit),
     "events": lambda: collect_events(limit), "derivs": lambda: collect_derivatives(limit),
     "bars": lambda: collect_bars(limit), "ticks": lambda: collect_ticks(limit),
     "live": lambda: record_live(limit, minutes), "depth": lambda: record_depth(limit, minutes),
     "consolidate": consolidate_all, "features": lambda: build_features(limit),
     "manifest": build_manifest, "excel": lambda: export_excel(),
     "snapshot": snapshot_options, "bundle": bundle_for_mac}[name]()


def main(argv=None):
    global OUT, HOST, PORT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all", "live", "depth", "snapshot", "estimate", "bundle"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--limit", type=int, default=None, help="only first N securities per class (smoke test)")
    ap.add_argument("--minutes", type=float, default=None, help="duration for live/depth recorders")
    a = ap.parse_args(argv)
    OUT, HOST, PORT = Path(a.out), a.host, a.port
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "estimate":
        return estimate(a.limit)
    if a.step == "all":
        run_all(a.limit)
    else:
        run_step(a.step, a.limit, a.minutes)
    close()


if __name__ == "__main__":
    main()
