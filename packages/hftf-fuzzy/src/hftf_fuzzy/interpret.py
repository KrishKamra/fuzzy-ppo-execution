"""Linguistic rule report (ARCHITECTURE §5.4): each rule centre mapped to Low/Med/High per input, using
terciles of the training-split regime inputs."""

from __future__ import annotations

import numpy as np
import torch

from hftf_fuzzy.layer import FuzzyTSK

DEFAULT_TERMS = ("LOW", "MED", "HIGH")


def terciles(x: np.ndarray) -> np.ndarray:
    """(2, d): the 1/3 and 2/3 quantiles of each input, ignoring NaN."""
    return np.nanquantile(x, [1 / 3, 2 / 3], axis=0)  # pre-09:20 rows are NaN for windowed inputs


def linguistic(value: float, cuts: np.ndarray, terms: tuple[str, str, str] = DEFAULT_TERMS) -> str:
    return terms[int(value > cuts[0]) + int(value > cuts[1])]


def rule_table(
    layer: FuzzyTSK,
    cuts: np.ndarray,
    input_names: list[str],
    input_terms: list[tuple[str, str, str]] | None = None,
) -> list[dict]:
    """One dict per rule: centre, width and term per input."""
    terms = input_terms or [DEFAULT_TERMS] * layer.d
    c, s = layer.centres.detach().numpy(), layer.widths().detach().numpy()
    rows = []
    for r in range(layer.n_rules):
        rows.append(
            {
                "rule": r,
                **{f"{n}_term": linguistic(c[r, j], cuts[:, j], terms[j]) for j, n in enumerate(input_names)},
                **{f"{n}_c": float(c[r, j]) for j, n in enumerate(input_names)},
                **{f"{n}_w": float(s[r, j]) for j, n in enumerate(input_names)},
            }
        )
    return rows


@torch.no_grad()
def firing_summary(layer: FuzzyTSK, x: torch.Tensor, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per rule: mean normalised firing (share) and firing-weighted mean of `value` (e.g. the pace)."""
    phi = layer.phi(x).numpy()
    share = phi.mean(0)
    weighted = (phi * value[:, None]).sum(0) / np.maximum(phi.sum(0), 1e-12)
    return share, weighted
