"""The demo backend's state and logic, without HTTP: market data, policies and config loaded once.

`DemoService.from_config` reads `configs/demo.yaml` (checkpoint globs under the git-ignored `runs/`, report
tables under `runs/reports/`). Tests build one on the synthetic market. Every order runs all requested
policies on one identical episode (hftf_eval.trace), so the replay compares like with like.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hftf_data.paths import REPO_ROOT
from hftf_eval.backtest import sample_episodes
from hftf_eval.baselines import TWAP, VWAP, AlmgrenChriss, PassiveVWAP
from hftf_eval.trace import run_trace
from hftf_sim import EnvConfig, MarketData, TickData
from hftf_sim.episodes import order_episode

DEMO_YAML = REPO_ROOT / "configs" / "demo.yaml"
INPUT_LABELS = [
    "Order-book imbalance (with / against us)",
    "Spread variance (30 min, log)",
    "Realised vol (30 min, log)",
]
BASELINES = {
    "vwap": ("VWAP", "Adaptive VWAP on the expected volume profile (a = 1)", VWAP),
    "twap": ("TWAP", "Equal slices over the horizon", TWAP),
    "almgren_chriss": ("Almgren-Chriss", "Closed-form mean-variance schedule", AlmgrenChriss),
    "vwap_passive100": (
        "VWAP, always passive", "VWAP that posts every child at the near touch", lambda: PassiveVWAP(1.0)
    ),
}  # fmt: skip
PROGRESS_COLUMNS = (
    "iteration", "timesteps", "mean_pace", "mean_passive", "kl", "explained_var", "n_rules", "edit_kl",
    "train_vs_vwap", "val_vs_vwap", "val_vs_vwap_se",
)  # fmt: skip


def clean(x):
    """JSON-safe: NaN/inf → None, numpy scalars → Python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


@dataclass
class PolicyEntry:
    name: str
    label: str
    description: str
    policy: object
    kind: str = "ppo"
    limit_orders: bool = False

    @property
    def explains(self) -> bool:
        return getattr(getattr(self.policy, "model", None), "fuzzy", None) is not None


@dataclass
class DemoService:
    market: MarketData
    cfg: EnvConfig
    policies: dict[str, PolicyEntry]
    ticks: TickData | None = None
    splits: tuple[str, ...] = ("val", "test")
    default_policies: tuple[str, ...] = ("vwap",)
    runs_dir: Path = REPO_ROOT / "runs"
    runs: tuple[str, ...] = ()
    results: list[dict] = field(default_factory=list)  # {id, title, file}

    def __post_init__(self) -> None:
        for name, (label, desc, make) in BASELINES.items():
            self.policies.setdefault(name, PolicyEntry(name, label, desc, make(), kind="baseline"))
        self._rules = lru_cache(maxsize=16)(self._rules_uncached)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, path: str | Path = DEMO_YAML) -> DemoService:
        from hftf_rl.ppo import load_ensemble, load_policy

        conf = yaml.safe_load(Path(path).read_text())
        cfg = EnvConfig.load()
        policies, stats = {}, None
        for p in conf.get("policies", []):
            paths = sorted(glob.glob(str(REPO_ROOT / p["glob"])))
            if not paths:
                continue  # a run that is not trained on this machine is simply not offered
            if len(paths) == 1:
                policy, _, ckpt = load_policy(paths[0], name=p["name"])
            else:
                policy, ckpt = load_ensemble(paths, p["name"]), load_policy(paths[0])[2]
            s = ckpt.get("market_stats")
            if stats is not None and s is not None and s != stats:
                raise ValueError(f"{p['name']}: trained with a different observation scaler")
            stats = stats or s
            limit = bool(ckpt["arch"].get("limit", False))
            policies[p["name"]] = PolicyEntry(
                p["name"], p["label"], p.get("description", ""), policy, "ppo", limit
            )
        market = MarketData.load(cfg.features_path, stats=stats)
        tick_path = REPO_ROOT / cfg.tick.path
        ticks = TickData.load(market, tick_path) if tick_path.exists() else None
        return cls(
            market=market, cfg=cfg, policies=policies, ticks=ticks,
            splits=tuple(conf.get("splits", ["val", "test"])),
            default_policies=tuple(
                n for n in conf.get("default_policies", ["vwap"]) if n in policies or n in BASELINES
            ),
            runs=tuple(r for r in conf.get("runs", []) if (REPO_ROOT / "runs" / r / "progress.csv").exists()),
            results=[t for t in conf.get("results", []) if (REPO_ROOT / t["file"]).exists()],
        )  # fmt: skip

    # ------------------------------------------------------------------- meta
    def _sessions(self) -> pd.DataFrame:
        s = self.market.sessions.reset_index(names="session")
        s = s[s["split"].isin(self.splits)].copy()
        s["date"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
        s["ticks"] = False if self.ticks is None else self.ticks.row[s["session"].to_numpy()] >= 0
        return s

    def meta(self) -> dict:
        s, ep = self._sessions(), self.cfg.episode
        return {
            "securities": sorted(s["security"].unique()),
            "sessions": s[["security", "date", "split", "ticks"]].to_dict("records"),
            "horizons": list(ep.horizons_min),
            "earliest_start": ep.earliest_start,
            "latest_end": ep.latest_end,
            "participation": tuple(ep.participation_loguniform),
            "policies": [
                {
                    "name": e.name,
                    "label": e.label,
                    "kind": e.kind,
                    "description": e.description,
                    "explains": e.explains,
                    "limit_orders": e.limit_orders,
                }
                for e in self.policies.values()
            ],  # fmt: skip
            "default_policies": list(self.default_policies),
            "runs": list(self.runs),
            "regime_inputs": list(self.market.obs_columns[:3]),
        }

    # ------------------------------------------------------------------ order
    def session_id(self, security: str, date: str) -> tuple[int, bool]:
        s = self._sessions()
        hit = s[(s["security"] == security) & (s["date"] == date)]
        if hit.empty:
            raise ValueError(f"no {'/'.join(self.splits)} session for {security} on {date}")
        return int(hit["session"].iloc[0]), bool(hit["ticks"].iloc[0])

    def random_ticket(self, split: str = "val", sim: str = "bar", seed: int = 0) -> dict:
        """A random order ticket from the training distribution (on tick sessions when `sim` is tick)."""
        if split not in self.splits:
            raise ValueError(f"split must be one of {self.splits}")
        sessions = self.ticks.session_ids(self.market, split) if sim == "tick" and self.ticks else None
        if sim == "tick" and (sessions is None or len(sessions) == 0):
            raise ValueError(f"no tick sessions in {split}")
        ep = sample_episodes(self.market, self.cfg, split, 1, seed, sessions)
        s, row0, h = int(ep.session[0]), int(ep.row0[0]), int(ep.horizon[0])
        cp = self.market.cum_prof
        expected = cp[s, row0 + h + 1] - cp[s, row0 + 1]
        sess = self.market.sessions.iloc[s]
        m = 9 * 60 + 15 + row0 + 1
        return {
            "security": str(sess["security"]), "date": pd.Timestamp(sess["date"]).strftime("%Y-%m-%d"),
            "start": f"{m // 60:02d}:{m % 60:02d}", "horizon": h, "side": "buy" if ep.side[0] > 0 else "sell",
            "participation": round(float(ep.qty[0] / expected), 4),
        }  # fmt: skip

    def order(self, req: dict) -> dict:
        names = list(req.get("policies") or self.default_policies)
        unknown = [n for n in names if n not in self.policies]
        if unknown:
            raise KeyError(f"unknown policies {unknown}")
        session, has_ticks = self.session_id(req["security"], req["date"])
        tick = req.get("sim", "bar") == "tick"
        if tick and not has_ticks:
            raise ValueError("TickSim needs a tick session (val Sep 21-30 or test)")
        ep = order_episode(
            self.market,
            self.cfg.episode,
            session,
            req["start"],
            req["horizon"],
            req["side"],
            req["participation"],
        )
        entries = [self.policies[n] for n in names]
        out = run_trace([e.policy for e in entries], self.market, self.cfg, ep, self.ticks if tick else None)
        for e, trace in zip(entries, out["policies"], strict=True):
            trace["name"], trace["label"] = e.name, e.label
        return clean(out)

    # ------------------------------------------------------------------ rules
    def rules(self, name: str) -> dict:
        if name not in self.policies:
            raise KeyError(f"unknown policy {name!r}")
        if not self.policies[name].explains:
            raise ValueError(f"{name} is not a single fuzzy checkpoint")
        return self._rules(name)

    def _rules_uncached(self, name: str) -> dict:
        from hftf_rl.interpret import INPUTS, rule_report, train_regime_terciles

        policy = self.policies[name].policy
        ep = sample_episodes(self.market, self.cfg, "val" if "val" in self.splits else self.splits[0], 600, 1)
        rows = rule_report(policy, self.market, self.cfg, ep)
        rules = [
            {
                "rule": r["rule"], "centres": [r[f"{i}_c"] for i in INPUTS],
                "widths": [r[f"{i}_w"] for i in INPUTS],
                "terms": [r[f"{i}_term"] for i in INPUTS], "pace": r["pace"], "passive": r.get("passive"),
                "share": r["share"],
            }
            for r in sorted(rows, key=lambda r: r["rule"])
        ]  # fmt: skip
        cuts = train_regime_terciles(self.market)
        return clean(
            {
                "policy": name,
                "inputs": list(INPUTS),
                "input_labels": INPUT_LABELS,
                "terciles": cuts.T.tolist(),
                "rules": rules,
            }  # fmt: skip
        )

    # ------------------------------------------------------- runs and results
    def run(self, name: str) -> dict:
        if name not in self.runs:  # only configured run names: no path from the request reaches the disk
            raise KeyError(f"unknown run {name!r}")
        d = self.runs_dir / name
        prog = pd.read_csv(d / "progress.csv")
        progress = {c: prog[c].tolist() for c in PROGRESS_COLUMNS if c in prog.columns}
        events = pd.read_csv(d / "events.csv").to_dict("records") if (d / "events.csv").exists() else []
        return clean({"run": name, "progress": progress, "events": events})

    def result_tables(self) -> dict:
        tables = []
        for t in self.results:
            df = pd.read_csv(REPO_ROOT / t["file"] if not Path(t["file"]).is_absolute() else t["file"])
            df = df[t["columns"]] if "columns" in t else df
            tables.append({"id": t["id"], "title": t["title"], "columns": list(df.columns),
                           "rows": df.to_dict("records")})  # fmt: skip
        return clean({"tables": tables})
