"""Pass/fail data-readiness checks over an audit report (docs/DATA_READINESS.md).

Thresholds live in configs/data_readiness.yaml. Coverage requirements depend on the tier (pilot or
research); integrity requirements are the same for every tier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from hftf_data.paths import REPO_ROOT, safe_name

DEFAULT_THRESHOLDS = REPO_ROOT / "configs" / "data_readiness.yaml"


@dataclass
class Check:
    id: str
    name: str
    passed: bool
    measured: str
    required: str
    detail: str = ""


def load_thresholds(tier: str, path: Path = DEFAULT_THRESHOLDS) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if tier not in cfg["tiers"]:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(cfg['tiers'])}")
    return {**cfg["tiers"][tier], **cfg["integrity"]}


def _fails(values: dict, ok) -> str:
    bad = {k: v for k, v in values.items() if not ok(v)}
    return ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))


def evaluate(report: dict, th: dict, features: pd.DataFrame | None = None) -> list[Check]:
    equities = report["intraday_equities"]
    bars = {(b["security"], b["event"]): b for b in report["bars"]}
    checks: list[Check] = []

    def add(id_, name, passed, measured, required, detail=""):
        checks.append(Check(id_, name, bool(passed), str(measured), str(required), detail))

    n = len(equities)
    add(
        "R01",
        "intraday equity universe",
        n >= th["min_intraday_equities"],
        n,
        f">= {th['min_intraday_equities']}",
    )

    sessions = {
        f"{safe_name(s)}/{ev}": bars.get((safe_name(s), ev), {}).get("days") or 0
        for s in equities
        for ev in ("TRADE", "BID", "ASK")
    }
    lo = min(sessions.values(), default=0)
    add(
        "R02",
        "1-min bar sessions per equity/event",
        lo >= th["min_bar_sessions"],
        f"min {lo}",
        f">= {th['min_bar_sessions']}",
        _fails(sessions, lambda v: v >= th["min_bar_sessions"]),
    )

    tracked = [(safe_name(s), ev) for s in equities for ev in ("TRADE", "BID", "ASK")]
    tracked += [(safe_name(s), "TRADE") for s in th["context_instruments"]]
    missing = {f"{s}/{ev}": len(bars.get((s, ev), {}).get("missing_sessions", [])) for s, ev in tracked}
    worst = max(missing.values(), default=0)
    dates = sorted({d for s, ev in tracked for d in bars.get((s, ev), {}).get("missing_sessions", [])})
    add(
        "R03",
        "missing trading sessions in bar window",
        worst <= th["max_missing_bar_sessions"],
        f"max {worst}",
        f"<= {th['max_missing_bar_sessions']}",
        f"dates: {', '.join(dates)}" if dates else "",
    )

    absent = [s for s in th["context_instruments"] if not bars.get((safe_name(s), "TRADE"), {}).get("rows")]
    add(
        "R04",
        "market context bars (NIFTY, BANKNIFTY, VIX)",
        not absent,
        f"{len(th['context_instruments']) - len(absent)}/{len(th['context_instruments'])}",
        "all",
        ", ".join(absent),
    )

    ticks = pd.DataFrame(report["ticks"])
    full = ticks[ticks["last_tick_ist"] >= th["full_tick_day_last_tick_ist"]] if len(ticks) else ticks
    full_days = {
        safe_name(s): int((full["security"] == safe_name(s)).sum()) if len(full) else 0 for s in equities
    }
    # Ticks are pulled for a subset (collector TICK_UNIVERSE_N), so require enough tick-covered names.
    covered = sorted(s for s, v in full_days.items() if v >= th["min_full_tick_days"])
    add(
        "R05",
        f"equities with >= {th['min_full_tick_days']} full tick days",
        len(covered) >= th["min_tick_equities"],
        len(covered),
        f">= {th['min_tick_equities']}",
        "best: " + ", ".join(f"{k}={v}" for k, v in sorted(full_days.items(), key=lambda kv: -kv[1])[:5]),
    )

    if len(full):
        err = (full["clean_over_bar"] - 1).abs()
        bad = full.loc[err > th["max_volume_recon_error"], ["security", "day"]]
        add(
            "R06",
            "tick vs bar volume reconciliation",
            bad.empty,
            f"max err {err.max():.4%}",
            f"<= {th['max_volume_recon_error']:.1%}",
            ", ".join(f"{r.security} {r.day}" for r in bad.itertuples()),
        )
        crossed = int(ticks["crossed_after_clean"].sum())
        add(
            "R07",
            "crossed quotes after cleaning",
            crossed <= th["max_crossed_quotes"],
            crossed,
            f"<= {th['max_crossed_quotes']}",
        )
    else:
        add("R06", "tick vs bar volume reconciliation", False, "no full tick days", "data required")
        add("R07", "crossed quotes after cleaning", False, "no tick days", "data required")

    pearson = {r["security"]: r["pearson"] for r in report["imbalance_proxy"]}
    lo = min(pearson.values(), default=float("nan"))
    add(
        "R08",
        "imbalance proxy vs tick L1 (Pearson)",
        bool(pearson) and lo >= th["min_imbalance_proxy_pearson"],
        f"min {lo}",
        f">= {th['min_imbalance_proxy_pearson']}",
        _fails(pearson, lambda v: v >= th["min_imbalance_proxy_pearson"]),
    )

    tick = {r["security"]: (r["inferred_tick"], r["nse_rule_tick_at_median_px"]) for r in report["tick_size"]}
    bad = {k: f"{a}≠{b}" for k, (a, b) in tick.items() if a != b}
    add(
        "R09",
        "inferred tick size matches NSE band rule",
        tick and not bad,
        f"{len(tick) - len(bad)}/{len(tick)}",
        "all",
        ", ".join(f"{k} {v}" for k, v in bad.items()),
    )

    daily = report["daily"]["equity"].get("sessions_per_security", {})
    dsess = {s: daily.get(s, 0) for s in equities}
    lo = min(dsess.values(), default=0)
    add(
        "R10",
        "daily history per equity",
        lo >= th["min_daily_sessions"],
        f"min {lo}",
        f">= {th['min_daily_sessions']}",
        _fails(dsess, lambda v: v >= th["min_daily_sessions"]),
    )

    if features is None:
        add(
            "R11",
            "features_1m_v2 built and covering eFNN inputs",
            False,
            "not built",
            "run `hftf data build-features`",
        )
    else:
        f = features[~features["dq_partial_day"].astype(bool)]
        cov = f[th["efnn_input_columns"]].notna().mean()
        uncovered = sorted(set(equities) - set(f["security"]))
        ok = cov.min() >= th["min_feature_coverage"] and not uncovered
        detail = ", ".join(f"{c}={v:.3f}" for c, v in cov.items())
        if uncovered:
            detail += f"; missing securities: {', '.join(uncovered)}"
        add(
            "R11",
            "features_1m_v2 built and covering eFNN inputs",
            ok,
            f"min coverage {cov.min():.3f}",
            f">= {th['min_feature_coverage']} for all equities",
            detail,
        )
    return checks


def summary(checks: list[Check]) -> dict:
    return {"ready": all(c.passed for c in checks), "checks": [asdict(c) for c in checks]}
