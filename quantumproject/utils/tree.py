import networkx as nx
from collections import deque


class BulkTree:
    """
    A balanced binary tree whose leaves correspond one-to-one with qubits.
    Provides utilities for interval edge cuts, leaf descendants, and curvatures.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.tree = nx.Graph()
        self.leaf_nodes_list = []           # Will store names of leaves: "q0", "q1", ...
        self._build_balanced_binary_tree()  # Build the tree and populate leaf_nodes_list
        # Now expose the list of edges in a plain attribute (for GNN usage)
        self.edge_list = list(self.tree.edges)
        # Map each undirected edge to a unique index for convenience
        self.edge_to_index = {e: i for i, e in enumerate(self.edge_list)}
        self.edge_to_index.update({(v, u): i for (u, v), i in self.edge_to_index.items()})

    def _build_balanced_binary_tree(self):
        """
        Recursively builds a binary tree with exactly self.n_qubits leaves.
        Leaves are named "q0", "q1", ..., "q{n_qubits-1}".
        Internal nodes are named "v0", "v1", etc.
        """

        counter = {"leaf_idx": 0, "internal_idx": 0}

        def build_subtree(leaf_count: int):
            # Base case: exactly 1 leaf → create it, name "q{leaf_idx}"
            if leaf_count == 1:
                leaf_name = f"q{counter['leaf_idx']}"
                counter["leaf_idx"] += 1
                self.tree.add_node(leaf_name)
                self.leaf_nodes_list.append(leaf_name)
                return leaf_name

            # Otherwise split into two subtrees
            left_count = leaf_count // 2
            right_count = leaf_count - left_count

            left_child = build_subtree(left_count)
            right_child = build_subtree(right_count)

            internal_node = f"v{counter['internal_idx']}"
            counter["internal_idx"] += 1
            self.tree.add_node(internal_node)
            self.tree.add_edge(internal_node, left_child)
            self.tree.add_edge(internal_node, right_child)
            return internal_node

        # Build the entire tree
        build_subtree(self.n_qubits)

        # Verify correct number of leaves
        assert len(self.leaf_nodes_list) == self.n_qubits, (
            f"Expected {self.n_qubits} leaves, got {len(self.leaf_nodes_list)}"
        )

    def leaf_nodes(self) -> list[str]:
        """
        Returns a list of all leaf-node names, i.e. nodes of degree 1 named "q*".
        """
        return list(self.leaf_nodes_list)

    def leaf_descendants(self) -> dict[str, list[str]]:
        """
        For each internal (non-leaf) node, return the list of leaf-node names reachable from it.
        Only includes nodes whose degree > 1.
        """
        descendants = {}
        all_leaves = set(self.leaf_nodes_list)

        for node in self.tree.nodes:
            # Skip actual leaves
            if self.tree.degree[node] == 1 and node in all_leaves:
                continue

            visited = {node}
            queue = deque([node])
            node_leaves = []

            while queue:
                curr = queue.popleft()
                for nbr in self.tree.neighbors(curr):
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    if nbr in all_leaves:
                        node_leaves.append(nbr)
                    else:
                        queue.append(nbr)

            descendants[node] = node_leaves

        return descendants

    def compute_curvatures(self, weights: list[float]) -> dict[str, float]:
        """
        Placeholder: assign each node a "curvature" based on the provided weights list.
        Replace this stub with actual curvature logic as needed.
        """
        curvature = {}
        nodes = list(self.tree.nodes)
        for i, node in enumerate(nodes):
            curvature[node] = weights[i % len(weights)]
        return curvature

    def interval_cut_edges(
        self, interval: tuple[int, ...], *, return_indices: bool = False
    ) -> list:
        """
        Given an interval of qubit indices (e.g. ``(2,)`` or ``(3, 4)``), find all
        edges whose removal separates exactly that set of leaves from the rest of
        the tree.

        Parameters
        ----------
        interval:
            Tuple of contiguous qubit indices. Example ``(0,)`` isolates leaf
            ``q0``; ``(2, 3)`` isolates leaves ``q2`` and ``q3``.
        return_indices:
            If ``True``, return indices of the edges instead of edge tuples using
            ``self.edge_to_index``.

        Returns
        -------
        list
            Edges (or edge indices if ``return_indices`` is ``True``) that
            isolate the given interval when removed.
        """
        # Convert qubit indices → leaf names
        target_leaves = {f"q{i}" for i in interval}

        # Special case: single‐leaf interval → return that leaf’s unique connecting edge
        if len(target_leaves) == 1:
            leaf = next(iter(target_leaves))
            if leaf not in self.tree:
                raise ValueError(f"Leaf {leaf} not found in tree.")
            neighbors = list(self.tree.neighbors(leaf))
            if not neighbors:
                raise ValueError(f"Leaf {leaf} has no connecting edge.")
            edge = (leaf, neighbors[0])
            if return_indices:
                return [self.edge_to_index[edge]]
            return [edge]

        # General case: interval length ≥ 2
        valid_edges = []
        for u, v in list(self.tree.edges):
            # Temporarily remove this edge
            self.tree.remove_edge(u, v)
            components = [set(c) for c in nx.connected_components(self.tree)]
            # Re-add the edge
            self.tree.add_edge(u, v)

            if len(components) != 2:
                continue

            leaf_components = [
                {n for n in comp if n in self.leaf_nodes_list} for comp in components
            ]
            if any(target_leaves == leaves for leaves in leaf_components):
                valid_edges.append((u, v))

        if not valid_edges:
            raise ValueError(f"No valid edge cut found for interval {interval}")

        if return_indices:
            return [self.edge_to_index[(u, v)] for (u, v) in valid_edges]
        return valid_edges
