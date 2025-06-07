import numpy as np
import pytest

from analysis.geodesic import embed_mds, geodesic_matrix
from quantumproject.utils.tree import BulkTree

pytest.importorskip(
    "sklearn.manifold", reason="scikit-learn not installed, skipping geodesic tests"
)


def test_geodesic_matrix_simple():
    tree = BulkTree(4)
    weights = [1.0] * len(tree.edge_list)
    dist = geodesic_matrix(tree, weights)
    assert dist.shape == (4, 4)
    assert np.isclose(dist[0, 1], 2.0)


def test_embed_mds_shape():
    tree = BulkTree(4)
    weights = [1.0] * len(tree.edge_list)
    dist = geodesic_matrix(tree, weights)
    coords = embed_mds(dist)
    assert coords.shape == (4, 2)
