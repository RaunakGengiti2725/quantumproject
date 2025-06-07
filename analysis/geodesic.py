"""Geodesic distance reconstruction utilities."""

from __future__ import annotations

from typing import Iterable

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from numpy import float64
from sklearn.manifold import MDS  # type: ignore[import-untyped]

from quantumproject.utils.tree import BulkTree


def geodesic_matrix(tree: BulkTree, weights: Iterable[float]) -> NDArray[float64]:
    """Return pairwise geodesic distances between leaf nodes."""
    g = tree.tree.copy()
    for (u, v), w in zip(tree.edge_list, weights):
        g[u][v]["weight"] = float(w)
    leaves = tree.leaf_nodes()
    n = len(leaves)
    dist: NDArray[float64] = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = nx.shortest_path_length(g, leaves[i], leaves[j], weight="weight")
            dist[i, j] = dist[j, i] = d
    return dist


def embed_mds(
    dist: NDArray[float64], n_components: int = 2, random_state: int = 0
) -> NDArray[float64]:
    """Embed distance matrix using multidimensional scaling."""
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
    )
    coords = mds.fit_transform(dist)
    return coords
