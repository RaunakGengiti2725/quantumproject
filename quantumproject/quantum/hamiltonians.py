"""Quantum Hamiltonians used for simulations."""

from __future__ import annotations

import pennylane as qml


def tfim(n: int, J: float = 1.0, h: float = 1.0) -> qml.Hamiltonian:
    """Return transverse field Ising model Hamiltonian."""
    coeffs = []
    ops = []
    for i in range(n):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n))
    for i in range(n):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def heisenberg_xxz(n: int, J: float = 1.0, Delta: float = 1.0) -> qml.Hamiltonian:
    """Return XXZ Heisenberg Hamiltonian with anisotropy Delta."""
    coeffs = []
    ops = []
    for i in range(n):
        nxt = (i + 1) % n
        coeffs.append(J)
        ops.append(qml.PauliX(i) @ qml.PauliX(nxt))
        coeffs.append(J)
        ops.append(qml.PauliY(i) @ qml.PauliY(nxt))
        coeffs.append(Delta)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(nxt))
    return qml.Hamiltonian(coeffs, ops)


def heisenberg(n: int, J: float = 1.0) -> qml.Hamiltonian:
    """Return the isotropic Heisenberg Hamiltonian J * (XX + YY + ZZ)."""
    coeffs = []
    ops = []
    for i in range(n):
        nxt = (i + 1) % n
        coeffs.append(J)
        ops.append(qml.PauliX(i) @ qml.PauliX(nxt))
        coeffs.append(J)
        ops.append(qml.PauliY(i) @ qml.PauliY(nxt))
        coeffs.append(J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(nxt))
    return qml.Hamiltonian(coeffs, ops)
