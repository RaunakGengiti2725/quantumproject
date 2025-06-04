"""CLI for running entanglement geometry experiments."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from quantumproject.quantum.simulations import (
    Simulator,
    contiguous_intervals,
    von_neumann_entropy,
)
from quantumproject.training.pipeline import train_step
from quantumproject.utils.tree import BulkTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--hamiltonian", choices=["tfim", "xxz"], default="tfim")
    parser.add_argument("--time", type=float, default=np.pi / 2)
    parser.add_argument("--logdir", default="runs")
    args = parser.parse_args()

    sim = Simulator(args.n_qubits)
    H = sim.build_hamiltonian(args.hamiltonian)
    state = sim.time_evolved_state(H, args.time)

    regions = contiguous_intervals(args.n_qubits)
    entropies = np.array([von_neumann_entropy(state, r) for r in regions])
    ent_torch = torch.tensor(entropies, dtype=torch.float32)

    tree = BulkTree(args.n_qubits)
    writer = SummaryWriter(args.logdir)
    weights = train_step(ent_torch, tree, writer)
    os.makedirs(args.logdir, exist_ok=True)
    np.save(os.path.join(args.logdir, "edge_weights.npy"), weights.numpy())
    writer.close()


if __name__ == "__main__":
    main()
