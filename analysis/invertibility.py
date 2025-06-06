"""Evaluate whether learned geometry reproduces entanglement."""

from __future__ import annotations

import numpy as np
from typing import Iterable

from quantumproject.utils.tree import BulkTree


def reconstruct_entropies(tree: BulkTree, weights: Iterable[float], intervals: Iterable[tuple[int, ...]]):
    """Approximate entropies via min-cut sums of edge weights."""
    weights = np.asarray(weights)
    ent = []
    for interval in intervals:
        edges = tree.interval_cut_edges(interval, return_indices=True)
        ent.append(weights[edges].sum())
    return np.array(ent)


def compare_entropies(true_ent: np.ndarray, recon_ent: np.ndarray) -> dict[str, float]:
    diff = true_ent - recon_ent
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    cos_sim = float(np.dot(true_ent, recon_ent) / (np.linalg.norm(true_ent) * np.linalg.norm(recon_ent) + 1e-12))

    if np.std(true_ent) > 1e-8 and np.std(recon_ent) > 1e-8:
        corr = float(np.corrcoef(true_ent, recon_ent)[0, 1])
    else:
        corr = 0.0
    return {"rmse": rmse, "cosine": cos_sim, "corr": corr}


def entropy_round_trip(
    tree: BulkTree,
    weights: Iterable[float],
    true_ent: np.ndarray,
    intervals: Iterable[tuple[int, ...]],
) -> dict[str, float]:
    """Reconstruct entropies from weights and compare to ground truth."""
    recon = reconstruct_entropies(tree, weights, intervals)
    metrics = compare_entropies(true_ent, recon)
    return metrics
