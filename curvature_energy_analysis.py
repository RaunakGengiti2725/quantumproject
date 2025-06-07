"""High-performance curvature and energy analysis utilities.

This module offers robust correlation routines and fast aggregation of
curvature and energy metrics on large graphs. It automatically detects GPU and
JAX backends when available and logs timing information for each public
function.
"""


from __future__ import annotations

import argparse
import logging
import time
from functools import wraps
from typing import Any, Callable, Tuple, TypeVar, cast

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

try:
    from numba import jit  # type: ignore[import-not-found]

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba not installed
    jit = None
    NUMBA_AVAILABLE = False

try:  # GPU backend detection
    import cupy as cp  # type: ignore[import-not-found]

    xp = cp
    BACKEND = "cupy"
except Exception:  # pragma: no cover - GPU not installed
    xp = np
    BACKEND = "numpy"

try:  # JAX detection
    from jax import jit as jax_jit  # type: ignore[import-not-found]

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - jax not installed
    jax_jit = None
    JAX_AVAILABLE = False


logger = logging.getLogger(__name__)

__all__ = [
    "compute_curvature",
    "compute_energy_deltas",
    "safe_pearson_correlation",
    "safe_einstein_correlation",
]

T = TypeVar("T")


def _jit(nopython: bool = True) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a Numba ``jit`` decorator if available."""

    if NUMBA_AVAILABLE and jit is not None:
        return cast(
            Callable[[Callable[..., T]], Callable[..., T]], jit(nopython=nopython)
        )

    def wrapper(fn: Callable[..., T]) -> Callable[..., T]:
        return fn

    return wrapper


def timed(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator that logs the execution time of ``fn``."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        result = fn(*args, **kwargs)
        duration = time.time() - start
        logger.info("%s executed in %.4f s", fn.__name__, duration)
        return result
    return wrapper


@timed
def safe_pearson_correlation(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> Tuple[float, float]:
    """Return Pearson correlation of ``x`` and ``y`` with robust handling.

    Parameters
    ----------
    x, y : np.ndarray
        Input arrays of equal length.

    Returns
    -------
    Tuple[float, float]
        Correlation coefficient ``r`` and two-tailed p-value ``p``.
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")

    mask = np.isfinite(x) & np.isfinite(y)
    cleaned = np.count_nonzero(~mask)
    if cleaned:
        logger.debug("Removed %d non-finite entries", cleaned)
    x = x[mask]
    y = y[mask]<

    if x.size < 2 or y.size < 2:
        logger.warning("Insufficient data for correlation; returning default")
        return 0.0, 1.0

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        logger.warning("Zero variance detected; returning default")
        return 0.0, 1.0

    try:
        r, p = stats.pearsonr(x, y)
        if np.isnan(r) or np.isnan(p):
            raise ValueError("nan result")
        return float(r), float(p)
    except Exception as exc:  # pragma: no cover - rarely executed
        logger.warning("SciPy pearsonr failed (%s); falling back to numpy", exc)
        xm = x - x.mean()
        ym = y - y.mean()
        r_num = np.dot(xm, ym)
        r_den = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
        if r_den == 0:
            return 0.0, 1.0
        r = r_num / r_den
        n = len(x)
        if n > 2 and abs(r) < 1:
            t = r * np.sqrt((n - 2)/(1 - r**2))
            p = 2 * stats.t.sf(abs(t), n - 2)
        else:
            p = 1.0
        return float(r), float(p)


@timed
def safe_einstein_correlation(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> float:
    """Return correlation coefficient using Einstein summation.

    This function mirrors :func:`safe_pearson_correlation` but computes the
    covariance and correlation coefficient via ``np.einsum``. Only the
    correlation ``r`` is returned.
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")

    mask = np.isfinite(x) & np.isfinite(y)
    cleaned = np.count_nonzero(~mask)
    if cleaned:
        logger.debug("Removed %d non-finite entries", cleaned)
    x = x[mask]
    y = y[mask]

    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        logger.warning("Insufficient data for correlation; returning default")
        return 0.0

    dx = x - x.mean()
    dy = y - y.mean()
    cov = np.einsum("i,i->", dx, dy) / (dx.size - 1)
    r = cov / (dx.std(ddof=1) * dy.std(ddof=1))
    return float(r)


@timed
def compute_curvature(graph: nx.Graph) -> NDArray[np.floating]:
    r"""Vectorized toy curvature estimate for each node.

    Uses a simple combinatorial expression based on node degrees:

    .. math:: k_i = 1 - \frac{d_i}{2} + \sum_{j \in N(i)} \frac{1}{d_j}

    Parameters
    ----------
    graph : nx.Graph
        Input undirected graph.

    Returns
    -------
    np.ndarray
        Array of curvatures ordered by ``graph.nodes()``.
    """

    nodelist = list(graph.nodes())
    A = nx.to_scipy_sparse_array(
        graph, nodelist=nodelist, weight=None, format="csr", dtype=float
    )
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg != 0)
    neighbor_sum = A.dot(inv_deg)
    curvature = 1.0 - deg / 2.0 + neighbor_sum
    return curvature


@_jit(nopython=True)
def _aggregate_energy(
    edges_u: NDArray[np.int_],
    edges_v: NDArray[np.int_],
    deltas: NDArray[np.floating],
    out: NDArray[np.floating],
) -> None:
    for i in range(edges_u.shape[0]):
        u = edges_u[i]
        v = edges_v[i]
        d = deltas[i]
        out[u] += d
        out[v] += d


@timed
def compute_energy_deltas(
    graph: nx.Graph, *, attr: str = "delta_energy"
) -> NDArray[np.floating]:
    """Aggregate energy deltas for each node.

    Parameters
    ----------
    graph : nx.Graph
        Graph with per-edge ``attr`` values representing energy change.
    attr : str, optional
        Edge attribute storing the energy delta.
        Defaults to ``"delta_energy"``.

    Returns
    -------
    np.ndarray
        Sum of energy deltas incident to each node.
    """

    nodelist = list(graph.nodes())
    index = {n: i for i, n in enumerate(nodelist)}
    edges = list(graph.edges(data=True))
    m = len(edges)
    u_idx = np.empty(m, dtype=np.int64)
    v_idx = np.empty(m, dtype=np.int64)
    delta = np.empty(m, dtype=np.float64)
    for i, (u, v, d) in enumerate(edges):
        u_idx[i] = index[u]
        v_idx[i] = index[v]
        delta[i] = float(d.get(attr, 0.0))
    out = np.zeros(len(nodelist), dtype=np.float64)
    _aggregate_energy(u_idx, v_idx, delta, out)
    return out


if JAX_AVAILABLE:
    safe_pearson_correlation_jax = jax_jit(safe_pearson_correlation)
    safe_einstein_correlation_jax = jax_jit(safe_einstein_correlation)
    compute_curvature_jax = jax_jit(compute_curvature)
    compute_energy_deltas_jax = jax_jit(compute_energy_deltas)
    __all__ += [
        "safe_pearson_correlation_jax",
        "safe_einstein_correlation_jax",
        "compute_curvature_jax",
        "compute_energy_deltas_jax",
    ]


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Benchmark curvature-energy analysis")
    parser.add_argument(
        "--nodes",
        type=int,
        default=1_000_000,
        help="Number of nodes in the random graph",
    )

    parser.add_argument("--p", type=float, default=0.000001, help="Edge probability")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logger.info("Generating random graph with %d nodes", args.nodes)
    start = time.time()
    g = nx.fast_gnp_random_graph(args.nodes, args.p, seed=42)
    logger.info("Graph generated in %.2f s", time.time() - start)

    timings = {}

    start = time.time()
    curv = compute_curvature(g)
    timings["curvature"] = time.time() - start

    for u, v in g.edges():
        g[u][v]["delta_energy"] = np.random.randn()

    start = time.time()
    dE = compute_energy_deltas(g)
    timings["energy"] = time.time() - start

    start = time.time()
    r, p = safe_pearson_correlation(curv, dE)
    timings["pearson"] = time.time() - start

    start = time.time()
    r_e = safe_einstein_correlation(curv, dE)
    timings["einstein"] = time.time() - start

    print("| Metric | Time (s) |")
    print("|---|---|")
    for k, v in timings.items():
        print(f"| {k} | {v:.4f} |")
