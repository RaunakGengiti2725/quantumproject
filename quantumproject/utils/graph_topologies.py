"""Generate alternative bulk graph topologies."""

from __future__ import annotations

import networkx as nx


def binary_tree(n_leaves: int) -> nx.Graph:
    tree = nx.Graph()
    counter = {"leaf_idx": 0, "internal_idx": 0}

    def build_subtree(num_leaves: int):
        if num_leaves == 1:
            name = f"q{counter['leaf_idx']}"
            counter['leaf_idx'] += 1
            tree.add_node(name)
            return name
        left = num_leaves // 2
        right = num_leaves - left
        l = build_subtree(left)
        r = build_subtree(right)
        node = f"v{counter['internal_idx']}"
        counter['internal_idx'] += 1
        tree.add_node(node)
        tree.add_edge(node, l)
        tree.add_edge(node, r)
        return node

    build_subtree(n_leaves)
    return tree


def ternary_tree(n_leaves: int) -> nx.Graph:
    tree = nx.Graph()
    counter = {"leaf_idx": 0, "internal_idx": 0}

    def build(num: int):
        if num <= 1:
            name = f"q{counter['leaf_idx']}"
            counter['leaf_idx'] += 1
            tree.add_node(name)
            return name
        split = [num // 3, num // 3, num - 2 * (num // 3)]
        children = [build(s) for s in split if s > 0]
        node = f"v{counter['internal_idx']}"
        counter['internal_idx'] += 1
        tree.add_node(node)
        for c in children:
            tree.add_edge(node, c)
        return node

    build(n_leaves)
    return tree


def random_tree(n_leaves: int, seed: int | None = None) -> nx.Graph:
    g = nx.random_tree(n_leaves * 2 - 1, seed=seed)
    mapping = {i: f"v{i}" for i in g.nodes}
    tree = nx.relabel_nodes(g, mapping)
    leaves = list(tree.nodes)[:n_leaves]
    for i, leaf in enumerate(leaves):
        new_name = f"q{i}"
        nx.relabel_nodes(tree, {leaf: new_name}, copy=False)
    return tree
