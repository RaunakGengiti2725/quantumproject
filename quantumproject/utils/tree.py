"""Bulk tree utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import math
import networkx as nx


class BulkTree:
    def __init__(self, n_qubits: int):
        self.tree = nx.balanced_tree(r=2, h=int(math.log2(n_qubits)))
        leaves = sorted(
            [
                node
                for node in self.tree.nodes
                if self.tree.degree[node] == 1 and node != 0
            ]
        )
        assert len(leaves) == n_qubits
        self.leaf_to_qubit = {leaf: idx for idx, leaf in enumerate(leaves)}
        self.edge_list = list(self.tree.edges)
        self.edge_to_index = {e: i for i, e in enumerate(self.edge_list)}
        self.edge_to_index.update(
            {(v, u): i for (u, v), i in self.edge_to_index.items()}
        )

    @lru_cache(maxsize=None)
    def interval_cut_edges(self, interval: Tuple[int, ...]) -> List[int]:
        qubits = list(interval)
        region_leaves = [leaf for leaf, q in self.leaf_to_qubit.items() if q in qubits]
        outside_leaves = [
            leaf for leaf, q in self.leaf_to_qubit.items() if q not in qubits
        ]
        min_cut = float("inf")
        best_edges = None
        for u in region_leaves:
            for v in outside_leaves:
                cut_edges = nx.minimum_edge_cut(self.tree, u, v)
                if len(cut_edges) < min_cut:
                    min_cut = len(cut_edges)
                    best_edges = cut_edges
        return [self.edge_to_index[(u, v)] for (u, v) in best_edges]
