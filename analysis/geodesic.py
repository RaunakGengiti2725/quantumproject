"""Geodesic distance reconstruction utilities."""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
from sklearn.manifold import MDS

from quantumproject.utils.tree import BulkTree


def geodesic_matrix(tree: BulkTree, weights: Iterable[float]) -> np.ndarray:
    """Return pairwise geodesic distances between leaf nodes."""
    g = tree.tree.copy()
    for (u, v), w in zip(tree.edge_list, weights):
        g[u][v]["weight"] = float(w)
    leaves = tree.leaf_nodes()
    n = len(leaves)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = nx.shortest_path_length(g, leaves[i], leaves[j], weight="weight")
            dist[i, j] = dist[j, i] = d
    return dist


def embed_mds(
    dist: np.ndarray, n_components: int = 2, random_state: int = 0
) -> np.ndarray:
    """Embed distance matrix using multidimensional scaling."""
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
    )
    coords = mds.fit_transform(dist)
    return coords
