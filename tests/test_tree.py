import pytest
from quantumproject.utils.tree import BulkTree


def test_interval_cut_edges_returns_indices():
    tree = BulkTree(4)
    edges = tree.interval_cut_edges((0,), return_indices=True)
    assert isinstance(edges[0], int)
    assert len(edges) == 1


def test_interval_cut_edges_error():
    tree = BulkTree(4)
    with pytest.raises(ValueError):
        tree.interval_cut_edges((0, 3))
