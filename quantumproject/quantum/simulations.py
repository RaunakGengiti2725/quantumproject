"""Utilities for time evolution and entropy computation."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .hamiltonians import tfim, heisenberg_xxz, heisenberg


class Simulator:
    """Simple wrapper around PennyLane devices for state evolution."""

    def __init__(self, n_qubits: int, backend: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(backend, wires=n_qubits, shots=None)
        self.hamiltonians = {
            "tfim": tfim,
            "xxz": heisenberg_xxz,
            "heisenberg": heisenberg,
        }

    def build_hamiltonian(self, name: str, **params) -> qml.Hamiltonian:
        if name not in self.hamiltonians:
            raise ValueError(f"Unknown Hamiltonian {name}")
        return self.hamiltonians[name](self.n_qubits, **params)

    def time_evolved_state(self, H: qml.Hamiltonian, t: float):
        """Return state vector after time t."""

        @qml.qnode(self.dev)
        def circuit():
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            qml.templates.ApproxTimeEvolution(H, t, 1)
            return qml.state()

        return circuit()


@lru_cache(maxsize=None)
def contiguous_intervals(n: int, max_len: int | None = None) -> list[tuple[int, ...]]:
    """Return all contiguous intervals up to ``max_len`` (exclusive of full chain)."""
    regions: list[tuple[int, ...]] = []
    limit = n if max_len is None else min(n, max_len + 1)
    for length in range(1, limit):
        for start in range(0, n - length):
            regions.append(tuple(range(start, start + length)))
    return regions


def reduced_density_matrix(state: Sequence[complex], subsys: Iterable[int]):
    n = int(np.log2(len(state)))
    psi = pnp.reshape(state, [2] * n)
    keep = list(subsys)
    trace_out = [i for i in range(n) if i not in keep]
    rho = pnp.tensordot(psi, pnp.conj(psi), axes=(trace_out, trace_out))
    dim_sub = 2 ** len(subsys)
    return pnp.reshape(rho, (dim_sub, dim_sub))


def von_neumann_entropy(state: Sequence[complex], subsys: Iterable[int]) -> float:
    rho = reduced_density_matrix(state, subsys)
    evs = qml.math.eigvalsh(rho)
    evs = pnp.clip(evs, 1e-9, 1)
    return float(-pnp.sum(evs * pnp.log(evs)))
  
def z_expectation(state: Sequence[complex], wire: int) -> float:
    """Expectation value of Z on given qubit from state vector."""
    rho = reduced_density_matrix(state, [wire])
    z = pnp.array([[1, 0], [0, -1]], dtype=complex)
    return float(pnp.trace(rho @ z).real)

def boundary_energy_delta(state_base: Sequence[complex], state_t: Sequence[complex]) -> list[float]:
    """Return ΔE_i = <Z_i>(t) - <Z_i>(0) for each qubit."""
    n = int(np.log2(len(state_base)))
    return [z_expectation(state_t, i) - z_expectation(state_base, i) for i in range(n)]

