"""Causal perturbation experiments for quantum geometry."""

from __future__ import annotations

import torch
import os
import numpy as np
from typing import Iterable

import pennylane as qml

from .simulations import Simulator, contiguous_intervals, von_neumann_entropy, boundary_energy_delta
from ..utils.tree import BulkTree
from ..training.pipeline import train_step


def _apply_perturbation(dev: qml.Device, qubits: Iterable[int], angle: float, kind: str) -> None:
    """Apply a perturbation gate to the given device wires."""
    for q in qubits:
        if kind.lower() == "x":
            qml.RX(angle, wires=q)
        else:
            qml.RZ(angle, wires=q)


def perturb_and_compare(
    n_qubits: int,
    hamiltonian: str,
    qubits: Iterable[int],
    angle: float = 0.1,
    kind: str = "z",
    time: float = np.pi / 4,
    *,
    outdir: str = "results/causal",
) -> dict[str, np.ndarray]:
    """Run a causal perturbation experiment and save differences."""
    os.makedirs(outdir, exist_ok=True)
    sim = Simulator(n_qubits)
    H = sim.build_hamiltonian(hamiltonian)

    # baseline evolution
    state_base = sim.time_evolved_state(H, time)
    intervals = contiguous_intervals(n_qubits)
    ent_base = np.array([von_neumann_entropy(state_base, r) for r in intervals])
    tree = BulkTree(n_qubits)
    weights_base = train_step(torch.tensor(ent_base, dtype=torch.float32), tree, steps=50)
    curv_base = tree.compute_curvatures(weights_base.numpy())
    dE_base = boundary_energy_delta(sim.time_evolved_state(H, 0.0), state_base)

    # perturbed evolution
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def perturbed_circuit():
        _apply_perturbation(dev, qubits, angle, kind)
        qml.templates.ApproxTimeEvolution(H, time, 1)
        return qml.state()

    state_pert = perturbed_circuit()
    ent_pert = np.array([von_neumann_entropy(state_pert, r) for r in intervals])
    weights_pert = train_step(torch.tensor(ent_pert, dtype=torch.float32), tree, steps=50)
    curv_pert = tree.compute_curvatures(weights_pert.numpy())
    dE_pert = boundary_energy_delta(sim.time_evolved_state(H, 0.0), state_pert)

    delta_S = ent_pert - ent_base
    delta_kappa = np.array([curv_pert[k] - curv_base[k] for k in curv_base])
    delta_E = np.array(dE_pert) - np.array(dE_base)


    if np.std(ent_base) > 1e-8:
        corr_before = np.corrcoef(ent_base)
    else:
        corr_before = np.zeros((1, 1))

    if np.std(ent_pert) > 1e-8:
        corr_after = np.corrcoef(ent_pert)
    else:
        corr_after = np.zeros((1, 1))
        
    np.save(os.path.join(outdir, "delta_entropy.npy"), delta_S)
    np.save(os.path.join(outdir, "delta_curvature.npy"), delta_kappa)
    np.save(os.path.join(outdir, "delta_energy.npy"), delta_E)
    np.save(os.path.join(outdir, "corr_before.npy"), corr_before)
    np.save(os.path.join(outdir, "corr_after.npy"), corr_after)

    return {
        "delta_entropy": delta_S,
        "delta_curvature": delta_kappa,
        "delta_energy": delta_E,
        "corr_before": corr_before,
        "corr_after": corr_after,
    }

def perturb_time_series(
    n_qubits: int,
    hamiltonian: str,
    qubits: Iterable[int],
    times: Iterable[float],
    angle: float = 0.1,
    kind: str = "z",
) -> dict[str, np.ndarray]:
    """Track perturbation effects across multiple time steps."""
    sim = Simulator(n_qubits)
    H = sim.build_hamiltonian(hamiltonian)
    tree = BulkTree(n_qubits)
    intervals = contiguous_intervals(n_qubits)

    base_state0 = sim.time_evolved_state(H, 0.0)
    base_series = []
    pert_series = []
    for t in times:
        state_t = sim.time_evolved_state(H, t)
        ent_base = np.array([von_neumann_entropy(state_t, r) for r in intervals])
        weights_base = train_step(torch.tensor(ent_base, dtype=torch.float32), tree, steps=50)
        curv_base = tree.compute_curvatures(weights_base.numpy())
        dE_base = boundary_energy_delta(base_state0, state_t)
        base_series.append((ent_base, curv_base, dE_base))

        dev = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(dev)
        def pert():
            _apply_perturbation(dev, qubits, angle, kind)
            qml.templates.ApproxTimeEvolution(H, t, 1)
            return qml.state()

        stp = pert()
        ent_pert = np.array([von_neumann_entropy(stp, r) for r in intervals])
        w_pert = train_step(torch.tensor(ent_pert, dtype=torch.float32), tree, steps=50)
        curv_pert = tree.compute_curvatures(w_pert.numpy())
        dE_pert = boundary_energy_delta(base_state0, stp)
        pert_series.append((ent_pert, curv_pert, dE_pert))

    delta_entropy = np.stack([p[0] - b[0] for b, p in zip(base_series, pert_series)])
    delta_energy = np.stack([np.array(p[2]) - np.array(b[2]) for b, p in zip(base_series, pert_series)])
    delta_curv = np.stack([
        np.array([p[1][k] - b[1][k] for k in b[1]])
        for b, p in zip(base_series, pert_series)
    ])

    return {
        "delta_entropy": delta_entropy,
        "delta_curvature": delta_curv,
        "delta_energy": delta_energy,
    }
