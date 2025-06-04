"""Generate paper-ready visualizations for the quantum geometry project."""

from __future__ import annotations

import os
import argparse
import numpy as np
import torch
from scipy.stats import pearsonr

from quantumproject.quantum.simulations import (
    Simulator,
    contiguous_intervals,
    von_neumann_entropy,
    boundary_energy_delta,
)
from quantumproject.training.pipeline import train_step
from quantumproject.utils.tree import BulkTree
from quantumproject.visualization.plots import (
    plot_einstein_correlation,
    plot_bulk_tree,
    plot_entropy_over_time,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--hamiltonian", choices=["tfim", "xxz"], default="tfim")
    parser.add_argument("--steps", type=int, default=5, help="number of time steps")
    parser.add_argument("--t_max", type=float, default=np.pi)
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    times = np.linspace(0.0, args.t_max, args.steps)
    sim = Simulator(args.n_qubits)
    H = sim.build_hamiltonian(args.hamiltonian)

    regions = contiguous_intervals(args.n_qubits)
    tree = BulkTree(args.n_qubits)

    state0 = sim.time_evolved_state(H, 0.0)
    ent_series = []
    correlations = []
    weights_last = None

    for t in times:
        state_t = sim.time_evolved_state(H, t)
        entropies = np.array([von_neumann_entropy(state_t, r) for r in regions])
        ent_series.append(entropies)
        ent_torch = torch.tensor(entropies, dtype=torch.float32)
        weights = train_step(ent_torch, tree, writer=None, steps=100)
        weights_last = weights.detach().numpy()
        curv = tree.compute_curvatures(weights_last)
        dE = boundary_energy_delta(state0, state_t)
        pairs = []
        leaves = tree.leaf_descendants()
        for node in curv:
            if tree.tree.degree[node] > 1:
                delta_sum = sum(dE[i] for i in leaves[node])
                pairs.append((curv[node], delta_sum))
        if len(pairs) >= 2:
            r, _ = pearsonr(*zip(*pairs))
        else:
            r = float("nan")
        correlations.append(r)

    ent_series = np.stack(ent_series)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    plot_einstein_correlation(times, correlations, outdir)
    if weights_last is not None:
        plot_bulk_tree(tree.tree, weights_last, outdir)

    key_intervals = [regions[0], regions[1], regions[3]]
    series_dict = {r: ent_series[:, regions.index(r)] for r in key_intervals}
    plot_entropy_over_time(times, series_dict, outdir)


if __name__ == "__main__":
    main()
