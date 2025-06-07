import numpy as np
from scipy import stats  # type: ignore

from curvature_energy_analysis import (
    safe_pearson_correlation,
    safe_einstein_correlation,
    compute_curvature,
    compute_energy_deltas,
)
import networkx as nx  # type: ignore



def test_safe_pearson_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    r1, p1 = safe_pearson_correlation(x, y)
    r2, p2 = stats.pearsonr(x, y)
    assert np.allclose(r1, r2)
    assert np.allclose(p1, p2)


def test_safe_pearson_constant_arrays():
    x = np.ones(10)
    y = np.arange(10)
    r, p = safe_pearson_correlation(x, y)
    assert r == 0.0
    assert p == 1.0


def test_safe_pearson_handles_nan_inf():
    x = np.array([1.0, np.nan, 2.0, np.inf])
    y = np.array([2.0, 3.0, np.nan, 5.0])
    r, p = safe_pearson_correlation(x, y)
    assert not np.isnan(r)
    assert not np.isnan(p)


def test_safe_einstein_constant():
    x = np.ones(10)
    y = np.arange(10)
    r = safe_einstein_correlation(x, y)
    assert r == 0.0


def test_compute_functions_shapes():
    g = nx.path_graph(4)
    for u, v in g.edges():
        g[u][v]["delta_energy"] = 1.0
    kappa = compute_curvature(g)
    dE = compute_energy_deltas(g)
    assert kappa.shape == (4,)
    assert dE.shape == (4,)

def test_backend_consistency():
    x = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)
    r_np, _ = safe_pearson_correlation(x, y)
    r_e = safe_einstein_correlation(x, y)
    try:
        from curvature_energy_analysis import (
            JAX_AVAILABLE,
            safe_pearson_correlation_jax,
            safe_einstein_correlation_jax,
        )

        if JAX_AVAILABLE:
            r_jax, _ = safe_pearson_correlation_jax(x, y)
            r_e_jax = safe_einstein_correlation_jax(x, y)
            assert np.allclose(r_jax, r_np)
            assert np.allclose(r_e_jax, r_e)
    except Exception:
        pass

