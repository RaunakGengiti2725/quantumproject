"""Visualize attention weights on bulk tree edges."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_attention(tree: nx.Graph, attn: np.ndarray, outdir: str = "figures/phase4"):
    """Plot attention weights as edge thickness."""
    norm = plt.Normalize(vmin=np.min(attn), vmax=np.max(attn))
    cmap = plt.cm.viridis
    widths = 2 + 4 * (attn - attn.min()) / (attn.ptp() + 1e-12)

    try:
        pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")
    except Exception:
        pos = nx.spring_layout(tree, seed=0)

    plt.figure(figsize=(6, 4))
    nx.draw_networkx_nodes(tree, pos, node_color="#eeeeee", edgecolors="#333333", node_size=400)
    for i, (u, v) in enumerate(tree.edges()):
        plt.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=cmap(norm(attn[i])),
            linewidth=widths[i],
        )
    nx.draw_networkx_labels(tree, pos)
    plt.axis("off")
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "attention_map.png"))
    plt.close()
