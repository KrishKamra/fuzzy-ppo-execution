"""Report and slide figures from the saved result tables (`runs/reports/*.csv`, `runs/<run>/`).

`hftf report figures` writes each figure as SVG (slides, web) and PNG (200 dpi) into `docs/figures/`, with
`docs/figures/README.md` listing captions and sources. Only aggregate statistics (costs in bps) are drawn,
never market data. Colours follow the web app's validated light palette (web/src/lib/palette.ts): VWAP is
the neutral reference, eFNN violet, MLP aqua, ANFIS magenta, limit orders orange. matplotlib is a dev
dependency, imported here only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hftf_eval.report import regime_table

COLORS = {
    "efnn": "#6250d6",
    "mlp": "#1baf7a",
    "anfis": "#e87ba4",
    "lip_efnn": "#eb6834",
    "lip_mlp": "#2a78d6",
    "vwap": "#6b6e7a",
    "vwap_passive100": "#eda100",
    "ink": "#1c1e26",
    "muted": "#5f6270",
    "grid": "#e6e7ee",
}
LABELS = {
    "mlp": "PPO-MLP",
    "anfis": "PPO-ANFIS",
    "efnn": "PPO-eFNN",
    "lip_mlp": "MLP + limit orders",
    "lip_efnn": "eFNN + limit orders",
    "vwap_passive100": "VWAP, always passive",
}
ENSEMBLES = ["mlp", "anfis", "efnn"]

# The independent replication (3 seeds, episode seed 42), recomputed with our statistics; its episode
# rows are not kept, so they come from docs/RESULTS.md "M4 replication and stricter statistics".
REPLICATION_VS_VWAP = {
    "mlp": (-0.136, -0.271, -0.007, -0.222, -0.053),
    "anfis": (-0.195, -0.354, -0.049, -0.316, -0.082),
    "efnn": (-0.237, -0.436, -0.067, -0.371, -0.119),
}  # mean, session lo, hi, date lo, hi


@dataclass
class Figure:
    name: str
    title: str
    caption: str
    sources: str


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["muted"],
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": COLORS["ink"],
            "axes.titlelocation": "left",
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "figure.dpi": 100,
        }
    )


def _forest(ax, rows: list[dict], xlabel: str) -> None:
    """Point estimates with session-cluster CIs (thick) and date-cluster CIs (thin) where available."""
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows, strict=True):
        c = r.get("color", COLORS["ink"])
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, lw=3, solid_capstyle="round", alpha=0.9)
        if r.get("date_lo") is not None:
            ax.plot([r["date_lo"], r["date_hi"]], [yi - 0.22, yi - 0.22], color=c, lw=1.2, alpha=0.7)
        ax.plot(r["mean"], yi, "o", color=c, ms=7, mec="white", mew=1.5, zorder=3)
        note = r.get("note")
        if note:
            ax.annotate(
                note,
                (1.0, yi),
                xycoords=("axes fraction", "data"),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=COLORS["muted"],
            )
    ax.axvline(0, color=COLORS["vwap"], lw=1, ls="--")
    # A slide often loses the caption, so the plot says what its bars are.
    handles = [ax.plot([], [], color=COLORS["muted"], lw=3)[0]]
    labels = ["95% CI, session clusters"]
    if any(r.get("date_lo") is not None for r in rows):
        handles.append(ax.plot([], [], color=COLORS["muted"], lw=1.2)[0])
        labels.append("95% CI, date clusters")
    ax.legend(handles, labels, loc="lower left", fontsize=8, handlelength=1.6)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def _p(x) -> str:
    return (
        ""
        if x is None or (isinstance(x, float) and np.isnan(x))
        else ("p<0.001" if x < 0.001 else f"p={x:.3f}")
    )


def fig_test_vs_vwap(reports: Path, ax) -> Figure:
    t = pd.read_csv(reports / "final_test_vs_vwap.csv").set_index("policy")
    rows = []
    for e in ENSEMBLES:
        r = t.loc[e]
        rows.append(
            {
                "label": f"{LABELS[e]} · ours (5 seeds)",
                "mean": r["mean"],
                "lo": r["lo"],
                "hi": r["hi"],
                "date_lo": r.get("date_lo"),
                "date_hi": r.get("date_hi"),
                "color": COLORS[e],
                "note": f"Holm {_p(r.get('holm_p'))}",
            }
        )
        m, lo, hi, dlo, dhi = REPLICATION_VS_VWAP[e]
        rows.append(
            {
                "label": f"{LABELS[e]} · replication (3 seeds)",
                "mean": m,
                "lo": lo,
                "hi": hi,
                "date_lo": dlo,
                "date_hi": dhi,
                "color": COLORS[e],
                "note": "replication",
            }
        )
    _forest(ax, rows, "objective − VWAP (bps of the order's value; < 0 is cheaper)")
    ax.set_title("Out of sample: every ensemble beats VWAP, in two independent runs")
    return Figure(
        "test_vs_vwap",
        "Test split vs VWAP, our run and the replication",
        "Held-out October test split (36 stock-sessions). Points: mean paired difference to VWAP; "
        "thick bars: "
        "95% session-cluster bootstrap CI; thin bars: 95% date-cluster CI. Holm-adjusted Wilcoxon p over the "
        "three ensembles.",
        "runs/reports/final_test_vs_vwap.csv; replication values from docs/RESULTS.md",
    )


def fig_architectures(reports: Path, ax) -> Figure:
    t = pd.read_csv(reports / "final_test_vs_group.csv").set_index("policy")
    rows = [
        {
            "label": f"{LABELS[p]} − PPO-MLP",
            "mean": t.loc[p, "mean"],
            "lo": t.loc[p, "lo"],
            "hi": t.loc[p, "hi"],
            "date_lo": t.loc[p].get("date_lo"),
            "date_hi": t.loc[p].get("date_hi"),
            "color": COLORS[p],
            "note": f"Holm {_p(t.loc[p].get('holm_p'))}",
        }
        for p in ("anfis", "efnn")
        if p in t.index
    ]
    _forest(ax, rows, "difference to PPO-MLP (bps)")
    ax.set_title("Fuzzy vs neural network: no difference survives the correction")
    return Figure(
        "architectures",
        "Architecture comparison on the test split",
        "Paired differences between seed ensembles on the test split, with session- and date-cluster CIs and "
        "Holm-adjusted p. The CIs nearly exclude zero, but no difference survives Holm in either run.",
        "runs/reports/final_test_vs_group.csv",
    )


def fig_robustness(reports: Path, axes) -> Figure:
    specs = [
        ("gamma", "robust_gamma.csv", "transient impact strength γ", (0.1, 0.5)),
        ("tau_res_min", "robust_tau_res_min.csv", "impact resilience τ (min)", (2, 15)),
        ("eta", "robust_eta.csv", "temporary impact η", (0.25, 1.0)),
    ]
    for ax, (param, file, xlabel, train) in zip(axes, specs, strict=True):
        t = pd.read_csv(reports / file)
        t = t[t["reference"] == "vwap"]
        ax.axvspan(*train, color=COLORS["grid"], alpha=0.7, lw=0, label="training range")
        for p in ("mlp", "efnn"):
            s = t[t["policy"] == p].sort_values(param)
            ax.fill_between(s[param], s["lo"], s["hi"], color=COLORS[p], alpha=0.15, lw=0)
            ax.plot(s[param], s["mean"], "-o", color=COLORS[p], ms=4, lw=2, label=LABELS[p])
        ax.axhline(0, color=COLORS["vwap"], lw=1, ls="--")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("objective − VWAP (bps)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].set_title("Robustness: the edge is transient-impact management")
    return Figure(
        "robustness",
        "Impact robustness sweeps (validation)",
        "Ensembles vs VWAP as one impact parameter moves inside (shaded) and outside the training range. "
        "The edge vanishes without transient impact and grows as it strengthens or persists; "
        "under very large "
        "temporary impact the policies lose to VWAP. Bands: 95% session-cluster CI.",
        "runs/reports/robust_{gamma,tau_res_min,eta}.csv",
    )


def fig_ticksim(reports: Path, ax) -> Figure:
    bar = pd.read_csv(reports / "m5_val_bar_vs_vwap.csv").set_index("policy")
    tick = pd.read_csv(reports / "m5_val_tick_vs_vwap.csv").set_index("policy")
    x = np.arange(len(ENSEMBLES))
    for off, (t, name, alpha) in zip(
        (-0.18, 0.18), ((bar, "BarSim", 0.45), (tick, "TickSim", 1.0)), strict=True
    ):
        means = np.array([t.loc[e, "mean"] for e in ENSEMBLES])
        err = np.array(
            [
                [t.loc[e, "mean"] - t.loc[e, "lo"] for e in ENSEMBLES],
                [t.loc[e, "hi"] - t.loc[e, "mean"] for e in ENSEMBLES],
            ]
        )
        ax.bar(
            x + off,
            means,
            0.34,
            color=[COLORS[e] for e in ENSEMBLES],
            alpha=alpha,
            label=name,
            yerr=err,
            error_kw={"ecolor": COLORS["muted"], "lw": 1, "capsize": 3},
        )
    ax.axhline(0, color=COLORS["vwap"], lw=1)
    ax.set_xticks(x, [LABELS[e] for e in ENSEMBLES])
    ax.set_ylabel("objective − VWAP (bps)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Replayed against the real order book, the edge holds and grows")
    return Figure(
        "ticksim",
        "BarSim vs TickSim on the val tick sessions",
        "Ensembles vs VWAP on identical episodes in the bar simulator (light) and the tick-replay simulator "
        "(solid), 24 val tick sessions. Error bars: 95% session-cluster CI.",
        "runs/reports/m5_val_{bar,tick}_vs_vwap.csv",
    )


def fig_limit_orders(reports: Path, ax) -> Figure:
    vs_vwap = pd.read_csv(reports / "m5_lip_val_ticksim_vs_vwap.csv").set_index("policy")
    vs_passive = pd.read_csv(reports / "m5_lip_val_ticksim_vs_group.csv").set_index("policy")
    rows = [
        {
            "label": f"{LABELS[p]} − VWAP",
            "mean": vs_vwap.loc[p, "mean"],
            "lo": vs_vwap.loc[p, "lo"],
            "hi": vs_vwap.loc[p, "hi"],
            "date_lo": vs_vwap.loc[p, "date_lo"],
            "date_hi": vs_vwap.loc[p, "date_hi"],
            "color": COLORS.get(p, COLORS["ink"]),
        }
        for p in ("efnn", "vwap_passive100", "lip_efnn", "lip_mlp")
    ]
    rows += [
        {
            "label": f"{LABELS[p]} − always passive",
            "mean": vs_passive.loc[p, "mean"],
            "lo": vs_passive.loc[p, "lo"],
            "hi": vs_passive.loc[p, "hi"],
            "date_lo": vs_passive.loc[p, "date_lo"],
            "date_hi": vs_passive.loc[p, "date_hi"],
            "color": COLORS[p],
            "note": f"Holm {_p(vs_passive.loc[p, 'holm_p'])}",
        }
        for p in ("lip_efnn", "lip_mlp")
    ]
    _forest(ax, rows, "cost difference (bps), TickSim")
    ax.set_title("Learned limit orders beat both VWAP and the naive passive rule")
    return Figure(
        "limit_orders",
        "Limit orders on the val tick sessions (TickSim)",
        "Pace-only eFNN, the naive always-passive VWAP and the learned limit-order policies against "
        "VWAP, and "
        "the learned policies against the naive rule. Passive fills are counterfactual, so absolute passive "
        "gains are an upper bound; the gain over the naive rule shares that bias. The date-cluster Holm p "
        "(0.016) is the floor for 8 dates.",
        "runs/reports/m5_lip_val_ticksim_vs_{vwap,group}.csv",
    )


def fig_training(runs: Path, run: str, axes) -> Figure:
    prog = pd.read_csv(runs / run / "progress.csv")
    ev = prog.dropna(subset=["val_vs_vwap"])
    a, b = axes
    a.fill_between(
        ev["iteration"],
        ev["val_vs_vwap"] - 1.96 * ev["val_vs_vwap_se"],
        ev["val_vs_vwap"] + 1.96 * ev["val_vs_vwap_se"],
        color=COLORS["efnn"],
        alpha=0.15,
        lw=0,
    )
    a.plot(ev["iteration"], ev["val_vs_vwap"], color=COLORS["efnn"], lw=2, label="validation")
    a.plot(ev["iteration"], ev["train_vs_vwap"], color=COLORS["muted"], lw=1.2, alpha=0.7, label="train")
    best = ev.loc[ev["val_cost"].idxmin(), "iteration"]
    a.axvline(best, color=COLORS["ink"], lw=1, ls=":")
    a.annotate(
        "best.pt",
        (best, a.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=8.5,
        color=COLORS["ink"],
    )
    a.axhline(0, color=COLORS["vwap"], lw=1, ls="--")
    a.set_ylabel("objective − VWAP (bps)")
    a.legend(fontsize=8.5, loc="upper right")
    a.set_title(f"Training {run}: the edge appears, and the rule base evolves")
    if "n_rules" in prog:
        b.step(prog["iteration"], prog["n_rules"], where="post", color=COLORS["efnn"], lw=2)
        events_path = runs / run / "events.csv"
        if events_path.exists():
            events = pd.read_csv(events_path)
            events = events[events["accepted"] == 1]
            marks = {
                "add": ("^", COLORS["mlp"]),
                "merge": ("D", COLORS["efnn"]),
                "prune": ("v", COLORS["lip_efnn"]),
            }
            for kind, (m, c) in marks.items():
                e = events[events["kind"] == kind]
                b.scatter(
                    e["iteration"],
                    e["n_rules"],
                    marker=m,
                    color=c,
                    s=28,
                    zorder=3,
                    label=f"{kind} ({len(e)})",
                )
            b.legend(fontsize=8.5, loc="upper right", ncol=3)
        b.set_ylabel("rules")
    b.set_xlabel("PPO iteration")
    return Figure(
        f"training_{run}",
        f"Training curves and rule evolution ({run})",
        "Top: objective vs VWAP on the trainer's fixed validation episodes (±1.96 SE) and on train; "
        "the dotted "
        "line marks the selected checkpoint. Bottom: number of fuzzy rules, with accepted add / merge / "
        "prune "
        "events (every edit passed the KL guard).",
        f"runs/{run}/progress.csv, runs/{run}/events.csv",
    )


def fig_by_security(reports: Path, ax) -> Figure:
    rows = pd.read_csv(reports / "final_test_rows.csv")
    t = regime_table(rows, ENSEMBLES, by="security")
    secs = sorted(t["security"].unique())
    x = np.arange(len(secs))
    for i, e in enumerate(ENSEMBLES):
        s = t[t["policy"] == e].set_index("security").loc[secs]
        ax.bar(
            x + (i - 1) * 0.26,
            s["mean"],
            0.24,
            color=COLORS[e],
            label=LABELS[e],
            yerr=[s["mean"] - s["lo"], s["hi"] - s["mean"]],
            error_kw={"ecolor": COLORS["muted"], "lw": 1, "capsize": 2},
        )
    ax.axhline(0, color=COLORS["vwap"], lw=1)
    ax.set_xticks(x, [s.replace(" IS Equity", "") for s in secs])
    ax.set_ylabel("objective − VWAP (bps)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("By stock: gains everywhere, least for APHS")
    return Figure(
        "by_security",
        "Test split by stock",
        "Ensembles vs VWAP per stock on the test split, 95% session-cluster CIs (12 sessions each).",
        "runs/reports/final_test_rows.csv",
    )


def write_figures(
    out: Path, reports: Path, runs: Path, training_run: str = "m4_efnn_s0"
) -> list[tuple[Figure, list[Path]]]:
    """Render every figure whose inputs exist; returns (figure, files) pairs and writes out/README.md."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    out.mkdir(parents=True, exist_ok=True)
    plans = [
        ("test_vs_vwap", (8.5, 4.6), 1, lambda axs: fig_test_vs_vwap(reports, axs[0])),
        ("architectures", (8.5, 2.6), 1, lambda axs: fig_architectures(reports, axs[0])),
        ("robustness", (12.5, 3.8), 3, lambda axs: fig_robustness(reports, axs)),
        ("ticksim", (7.5, 4.0), 1, lambda axs: fig_ticksim(reports, axs[0])),
        ("limit_orders", (8.5, 4.2), 1, lambda axs: fig_limit_orders(reports, axs[0])),
        ("training", (8.5, 6.0), "rows2", lambda axs: fig_training(runs, training_run, axs)),
        ("by_security", (7.5, 4.0), 1, lambda axs: fig_by_security(reports, axs[0])),
    ]
    done = []
    for name, size, layout, draw in plans:
        if layout == "rows2":
            fig, axs = plt.subplots(2, 1, figsize=size, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        else:
            fig, axs = plt.subplots(1, layout, figsize=size, squeeze=False)
            axs = list(axs[0])
        try:
            meta = draw(list(axs) if layout == "rows2" else axs)
        except (FileNotFoundError, KeyError) as err:  # a result that is not on this machine is skipped
            plt.close(fig)
            print(f"skipped {name}: {err}")
            continue
        fig.tight_layout()
        files = [out / f"{meta.name}.svg", out / f"{meta.name}.png"]
        fig.savefig(files[0])
        fig.savefig(files[1], dpi=200)
        plt.close(fig)
        done.append((meta, files))
    lines = [
        "# Figures",
        "",
        "Generated by `uv run hftf report figures` from the saved result tables; regenerate after",
        "re-running experiments. SVG for slides and the web, PNG (200 dpi) for documents.",
        "Numbers: docs/RESULTS.md.",
        "",
    ]
    for meta, files in done:
        lines += [
            f"## {meta.title}",
            "",
            f"![{meta.title}]({files[0].name})",
            "",
            meta.caption,
            "",
            f"Source: `{meta.sources}`",
            "",
        ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return done
