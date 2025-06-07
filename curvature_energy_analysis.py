"""High-performance curvature and energy analysis utilities."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Tuple

import numpy as np
import networkx as nx
from scipy import stats

try:
    from numba import jit

    def _jit(nopython=True):
        return jit(nopython=nopython)

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba not installed
    def _jit(nopython=True):
        def wrapper(fn):
            return fn
        return wrapper

    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


def safe_pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return Pearson correlation of ``x`` and ``y`` with robust handling."""
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
    except Exception as exc:  # pragma: no cover
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
            t = r * np.sqrt((n - 2) / (1 - r**2))
            p = 2 * stats.t.sf(abs(t), n - 2)
        else:
            p = 1.0
        return float(r), float(p)


@_jit(nopython=True)
def _aggregate_energy(
    edges_u: np.ndarray, edges_v: np.ndarray, deltas: np.ndarray, out: np.ndarray
) -> None:
    for i in range(edges_u.shape[0]):
        u = edges_u[i]
        v = edges_v[i]
        d = deltas[i]
        out[u] += d
        out[v] += d


def compute_curvature(graph: nx.Graph) -> np.ndarray:
    """Vectorized toy curvature estimate for each node."""
    nodelist = list(graph.nodes())
    A = nx.to_scipy_sparse_array(
        graph, nodelist=nodelist, weight=None, format="csr", dtype=float
    )
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg != 0)
    neighbor_sum = A.dot(inv_deg)
    return 1.0 - deg / 2.0 + neighbor_sum


def compute_energy_deltas(graph: nx.Graph, *, attr: str = "delta_energy") -> np.ndarray:
    """Aggregate energy deltas for each node."""
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


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Benchmark curvature-energy analysis"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1_000_000,
        help="Number of nodes in the random graph",
    )
    parser.add_argument("--p", type=float, default=1e-6, help="Edge probability")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logger.info("Generating random graph with %d nodes", args.nodes)
    start = time.time()
    g = nx.fast_gnp_random_graph(args.nodes, args.p, seed=42)
    logger.info("Graph generated in %.2f s", time.time() - start)

    start = time.time()
    curv = compute_curvature(g)
    logger.info("Curvature computed in %.2f s", time.time() - start)

    for u, v in g.edges():
        g[u][v]["delta_energy"] = np.random.randn()
    start = time.time()
    dE = compute_energy_deltas(g)
    logger.info("Energy deltas computed in %.2f s", time.time() - start)

    start = time.time()
    r, p = safe_pearson_correlation(curv, dE)
    logger.info("Correlation: r=%.5f p=%.5f (%.2f s)", r, p, time.time() - start)
