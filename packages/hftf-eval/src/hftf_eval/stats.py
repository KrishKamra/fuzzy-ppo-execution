"""Paired statistics for backtests (ARCHITECTURE §7).

Episodes drawn from the same (security, date) session share one price path, so their cost differences are
correlated. Confidence intervals therefore resample whole sessions (cluster bootstrap), and the Wilcoxon
signed-rank test runs on per-session mean differences. The per-episode normal CI is kept for reference only:
it is about half as wide as the cluster CI on val (60 sessions).

Two sensitivity checks (contributed with the independent M4 replication, docs/RESULTS.md):
  * date clusters: the three stocks share market-wide moves on a date, so resampling whole dates is the more
    conservative unit (12 test dates is few clusters, so it is a check, not the headline);
  * the unweighted session-mean t-interval, the estimand the session Wilcoxon test is about.
A family of comparisons (one table) gets Holm-adjusted Wilcoxon p-values (`holm`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


def session_key(df: pd.DataFrame) -> pd.Series:
    return df["security"].astype(str) + "|" + df["date"].astype(str)


def date_key(df: pd.DataFrame) -> pd.Series:
    return df["date"].astype(str)


def holm(p: np.ndarray | list[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error), in the input order. The running
    maximum keeps the adjusted values monotone in the raw ones."""
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adjusted
    return out


def _wilcoxon(means: np.ndarray) -> float:
    return float(wilcoxon(means).pvalue) if np.any(means != 0) else 1.0


def cluster_bootstrap_ci(
    d: np.ndarray, clusters: np.ndarray, n_boot: int = 4000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """CI of the episode-weighted mean of `d`, resampling clusters with replacement."""
    g = pd.DataFrame({"d": d, "k": clusters}).groupby("k")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def paired_stats(
    df: pd.DataFrame, policy: str, reference: str, col: str = "cost_bps", n_boot: int = 4000, seed: int = 0
) -> dict[str, float]:
    """`policy − reference` on identical episodes: mean, session-cluster bootstrap CI, Wilcoxon p on
    session means, per-episode normal CI, episode win rate and the number of sessions; plus the date-cluster
    CI and Wilcoxon p and the unweighted session-mean t-interval (sensitivity checks)."""
    wide = df.pivot(index="episode", columns="policy", values=col)
    keys = df.loc[df["policy"] == reference].set_index("episode").loc[wide.index]
    clusters, dates = session_key(keys).to_numpy(), date_key(keys).to_numpy()
    d = (wide[policy] - wide[reference]).to_numpy()
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    lo, hi = cluster_bootstrap_ci(d, clusters, n_boot, seed)
    date_lo, date_hi = cluster_bootstrap_ci(d, dates, n_boot, seed)
    per_session = pd.Series(d).groupby(clusters).mean().to_numpy()
    per_date = pd.Series(d).groupby(dates).mean().to_numpy()
    k = len(per_session)
    half = (
        float(student_t.ppf(0.975, k - 1) * per_session.std(ddof=1) / np.sqrt(k)) if k > 1 else float("nan")
    )
    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "wilcoxon_p": _wilcoxon(per_session),
        "naive_lo": mean - 1.96 * se,
        "naive_hi": mean + 1.96 * se,
        "win": float((d < 0).mean()),
        "sessions": k,
        "date_lo": date_lo,
        "date_hi": date_hi,
        "date_wilcoxon_p": _wilcoxon(per_date),
        "dates": len(per_date),
        "session_t_lo": float(per_session.mean()) - half,
        "session_t_hi": float(per_session.mean()) + half,
    }


def breakdown(
    df: pd.DataFrame, policy: str, reference: str, by: str, col: str = "cost_bps", **kwargs
) -> pd.DataFrame:
    """`paired_stats` within each value of `by` (e.g. the eval-only regime label, horizon or side)."""
    rows = []
    for value, part in df.groupby(by, sort=True):
        rows.append(
            {
                by: value,
                "episodes": part["episode"].nunique(),
                **paired_stats(part, policy, reference, col, **kwargs),
            }
        )
    return pd.DataFrame(rows)


def adjusted_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted mutual information (Vinh, Epps & Bailey 2010; arithmetic-mean normalisation, as in
    scikit-learn): 1 for identical partitions, about 0 for independent ones."""
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    n = len(ia)
    table = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(table, (ia, ib), 1)
    ra, cb = table.sum(1), table.sum(0)
    if len(ra) == 1 and len(cb) == 1 or len(ra) == len(cb) == n:
        return 1.0
    nz = table > 0
    mi = float((table[nz] / n * np.log(n * table[nz] / np.outer(ra, cb)[nz])).sum())
    h = lambda x: float(-(x[x > 0] / n * np.log(x[x > 0] / n)).sum())  # noqa: E731
    ha, hb = h(ra), h(cb)
    # Expected MI under the hypergeometric model of random partitions with these marginals.
    emi = 0.0
    lg = gammaln(np.arange(n + 2))
    for ai in ra:
        for bj in cb:
            lo, hi = int(max(1, ai + bj - n)), int(min(ai, bj))
            nij = np.arange(lo, hi + 1)
            if not len(nij):
                continue
            log_p = (
                lg[int(ai) + 1] + lg[int(bj) + 1] + lg[int(n - ai) + 1] + lg[int(n - bj) + 1]
                - lg[n + 1] - lg[nij + 1] - lg[int(ai) - nij + 1] - lg[int(bj) - nij + 1]
                - lg[int(n - ai - bj) + nij + 1]
            )  # fmt: skip
            emi += float((nij / n * np.log(n * nij / (ai * bj)) * np.exp(log_p)).sum())
    denom = (ha + hb) / 2 - emi
    return float((mi - emi) / denom) if abs(denom) > 1e-12 else 0.0
