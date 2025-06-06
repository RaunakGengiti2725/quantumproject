"""General bulk graph supporting custom topologies."""

from __future__ import annotations

import networkx as nx
from collections import deque
from typing import Iterable


class BulkGraph:
    def __init__(self, graph: nx.Graph):
        self.tree = graph
        self.leaf_nodes_list = [n for n in graph.nodes if graph.degree[n] == 1 and n.startswith("q")]
        self.edge_list = list(graph.edges)
        self.edge_to_index = {e: i for i, e in enumerate(self.edge_list)}
        self.edge_to_index.update({(v, u): i for (u, v), i in self.edge_to_index.items()})
        self.n_qubits = len(self.leaf_nodes_list)

    def leaf_nodes(self) -> list[str]:
        return list(self.leaf_nodes_list)

    def leaf_descendants(self) -> dict[str, list[str]]:
        descendants = {}
        leaves = set(self.leaf_nodes_list)
        for node in self.tree.nodes:
            if node in leaves:
                continue
            visited = {node}
            queue = deque([node])
            nodes = []
            while queue:
                curr = queue.popleft()
                for nbr in self.tree.neighbors(curr):
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    if nbr in leaves:
                        nodes.append(nbr)
                    else:
                        queue.append(nbr)
            descendants[node] = nodes
        return descendants

    def compute_curvatures(self, weights: Iterable[float]) -> dict[str, float]:
        weights = list(weights)
        curv = {}
        for i, node in enumerate(self.tree.nodes):
            curv[node] = weights[i % len(weights)]
        return curv

    def interval_cut_edges(self, interval: tuple[int, ...], *, return_indices: bool = False) -> list:
        target_leaves = {f"q{i}" for i in interval}
        others = set(self.leaf_nodes_list) - target_leaves
        if not others:
            raise ValueError("Interval covers all leaves")
        # connect source to target_leaves, sink to others
        g = self.tree.copy()
        g.add_node("source")
        g.add_node("sink")
        for t in target_leaves:
            g.add_edge("source", t, weight=0.0)
        for o in others:
            g.add_edge("sink", o, weight=0.0)
        cutset = nx.minimum_edge_cut(g, "source", "sink")
        edges = [e for e in cutset if "source" not in e and "sink" not in e]
        if return_indices:
            return [self.edge_to_index[e] for e in edges]
        return edges
