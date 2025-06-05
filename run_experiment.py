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
    parser = argparse.ArgumentParser(description="Run entanglement geometry experiment")
    parser.add_argument("--n_qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument("--hamiltonian", choices=["tfim", "xxz"], default="tfim", help="Hamiltonian type")
    parser.add_argument("--time", type=float, default=np.pi / 2, help="Evolution time")
    parser.add_argument("--logdir", default="runs", help="Directory to save logs and weights")
    args = parser.parse_args()

    print(f"🔧 Initializing simulator with {args.n_qubits} qubits using '{args.hamiltonian}' Hamiltonian...")

    sim = Simulator(args.n_qubits)
    H = sim.build_hamiltonian(args.hamiltonian)
    state = sim.time_evolved_state(H, args.time)

    print("📊 Computing entanglement entropies...")
    regions = contiguous_intervals(args.n_qubits)
    entropies = np.array([von_neumann_entropy(state, r) for r in regions])
    ent_torch = torch.tensor(entropies, dtype=torch.float32)

    print("🌳 Building tree structure and training model...")
    tree = BulkTree(args.n_qubits)
    writer = SummaryWriter(args.logdir)
    weights = train_step(ent_torch, tree, writer)

    os.makedirs(args.logdir, exist_ok=True)
    weights_path = os.path.join(args.logdir, "edge_weights.npy")
    np.save(weights_path, weights.numpy())
    writer.close()

    print(f"✅ Training complete. Weights saved to: {weights_path}")
    print(f"📁 TensorBoard logs saved to: {args.logdir}")
    print("📈 To generate visuals, run: python generate_figures.py")
    print("🧠 Done.")


if __name__ == "__main__":
    main()
