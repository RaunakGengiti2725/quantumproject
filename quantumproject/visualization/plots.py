"""Reusable plotting utilities for quantum geometry experiments."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx


_DEF_DPI = 300


def _save_fig(path_base: str) -> None:
    """Save current matplotlib figure to PNG and PDF with base filename."""
    png = f"{path_base}.png"
    pdf = f"{path_base}.pdf"
    plt.savefig(png, dpi=_DEF_DPI, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def plot_einstein_correlation(times: Sequence[float], r_values: Sequence[float], outdir: str) -> None:
    """Plot Pearson correlation between curvature and ΔE over time."""
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(times, r_values, marker="o", label="Pearson r")
    plt.xlabel("Time")
    plt.ylabel("Correlation r")
    plt.title("Curvature vs ΔE Correlation")
    plt.legend()
    plt.grid(alpha=0.3)
    _save_fig(os.path.join(outdir, "einstein_correlation"))


def plot_bulk_tree(tree: nx.Graph, edge_weights: Sequence[float], outdir: str, filename: str = "bulk_tree") -> None:
    """Visualize binary tree with edge weights encoded as thickness and color."""
    os.makedirs(outdir, exist_ok=True)
    pos = nx.nx_agraph.graphviz_layout(tree, prog="dot") if hasattr(nx, "nx_agraph") else nx.spring_layout(tree, seed=0)
    widths = [max(w, 1e-2) * 3 for w in edge_weights]
    colors = edge_weights
    plt.figure()
    nx.draw(
        tree,
        pos=pos,
        with_labels=False,
        node_size=300,
        width=widths,
        edge_color=colors,
        edge_cmap=plt.cm.plasma,
    )
    plt.title("Learned Bulk Edge Weights")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    plt.colorbar(sm, label="Weight")
    _save_fig(os.path.join(outdir, filename))


def plot_entropy_over_time(times: Sequence[float], interval_series: Dict[Iterable[int], Sequence[float]], outdir: str, filename: str = "entropy_over_time") -> None:
    """Plot entropy across time for selected intervals."""
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    for interval, series in interval_series.items():
        plt.plot(times, series, marker="o", label=f"{tuple(interval)}")
    plt.xlabel("Time")
    plt.ylabel("Entropy")
    plt.title("Entanglement Entropy Over Time")
    plt.legend(title="Interval")
    plt.grid(alpha=0.3)
    _save_fig(os.path.join(outdir, filename))
