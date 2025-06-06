"""Estimate emergent spatial dimension from scaling of geodesic distances."""

from __future__ import annotations

import numpy as np
from typing import Iterable


def avg_pairwise_distance(dist: np.ndarray) -> float:
    n = dist.shape[0]
    idx = np.triu_indices(n, 1)
    return float(dist[idx].mean())


def fit_dimension(system_sizes: Iterable[int], avg_dists: Iterable[float]) -> float:
    """Return estimated spatial dimension D from ⟨d⟩ ∝ n^(1/D)."""
    sizes = np.asarray(list(system_sizes), dtype=float)
    dists = np.asarray(list(avg_dists), dtype=float)
    log_n = np.log(sizes)
    log_d = np.log(dists)
    slope, _ = np.polyfit(log_n, log_d, 1)
    if slope == 0:
        return float('inf')
    return float(1.0 / slope)
