# File: quantumproject/quantum/simulations.py

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# ───────────────────────────────
# 1) TIME EVOLUTION QNODE
# ───────────────────────────────

# We will override the default device on demand inside run_one_time_step,
# so here we define a “placeholder” device. The actual device should be created
# in the training pipeline when needed (e.g., with 5 Trotter steps).
dev_placeholder = None


@qml.qnode(qml.device("default.qubit", wires=1), interface="autograd")
def time_evolved_state(t):
    """
    Placeholder QNode. This function is redefined at runtime in run_one_time_step
    with the appropriate number of wires and Hamiltonian. If you call this
    version directly, it will simply return |0⟩ for a single qubit.
    """
    # Default: do nothing, return state of a 1-qubit |0⟩
    return qml.state()


# ───────────────────────────────
# 2) HAMILTONIAN BUILDERS
# ───────────────────────────────


def make_tfim_hamiltonian(n, J=1.0, h=1.0):
    """
    Construct a transverse‐field Ising model (TFIM) Hamiltonian on n qubits:
      H = -J * Σ Z_i Z_{i+1}  - h * Σ X_i
    with periodic boundary conditions.
    """
    coeffs = []
    ops = []
    for i in range(n):
        Zi = qml.PauliZ(i)
        Zj = qml.PauliZ((i + 1) % n)
        coeffs.append(-J)
        ops.append(Zi @ Zj)
    for i in range(n):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def make_xxz_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0):
    """
    Construct an XXZ Hamiltonian on n qubits:
      H = Σ [ Jx * (X_i X_{i+1}) + Jy * (Y_i Y_{i+1}) + Jz * (Z_i Z_{i+1}) ]
          + h * Σ Z_i   (optional longitudinal field)
    with periodic boundary conditions.
    """
    coeffs = []
    ops = []
    for i in range(n):
        ip1 = (i + 1) % n
        Xi = qml.PauliX(i)
        Xj = qml.PauliX(ip1)
        Yi = qml.PauliY(i)
        Yj = qml.PauliY(ip1)
        Zi = qml.PauliZ(i)
        Zj = qml.PauliZ(ip1)

        # X_i X_{i+1}
        coeffs.append(Jx)
        ops.append(Xi @ Xj)
        # Y_i Y_{i+1}
        coeffs.append(Jy)
        ops.append(Yi @ Yj)
        # Z_i Z_{i+1}
        coeffs.append(Jz)
        ops.append(Zi @ Zj)

    # Optional longitudinal field term (on Z)
    if abs(h) > 1e-12:
        for i in range(n):
            coeffs.append(h)
            ops.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, ops)


# ───────────────────────────────
# 3) CONTIGUOUS INTERVAL ENUMERATION
# ───────────────────────────────


def contiguous_intervals(n_qubits, max_interval_size=None):
    """
    Return a list of all contiguous intervals of boundary qubits [0..n_qubits-1].
    If max_interval_size is provided (an integer between 1 and n_qubits-1),
    only intervals up to that length will be returned.
    """
    regions = []
    max_len = max_interval_size if max_interval_size is not None else n_qubits - 1
    for length in range(1, max_len + 1):
        for start in range(0, n_qubits - length + 1):
            regions.append(tuple(range(start, start + length)))
    return regions


# ───────────────────────────────
# 4) VON NEUMANN ENTROPY
# ───────────────────────────────


def reduced_density_matrix(state, subsys):
    """
    Given a full state vector `state` of length 2^n_qubits and a list `subsys`
    of qubit indices, compute the reduced density matrix on that subsystem
    by tracing out all other qubits.
    """
    n = int(np.log2(len(state)))
    dims = [2] * n
    psi = pnp.reshape(state, dims)
    keep = list(subsys)
    trace_out = [i for i in range(n) if i not in keep]
    # Partial trace
    rho = pnp.tensordot(psi, pnp.conj(psi), axes=(trace_out, trace_out))
    dim_sub = 2 ** len(subsys)
    return pnp.reshape(rho, (dim_sub, dim_sub))


def von_neumann_entropy(state, subsys):
    """
    Compute the von Neumann entropy S(ρ) = -Tr(ρ log ρ) of the subsystem 'subsys'
    from the full state vector `state`.
    """
    rho = reduced_density_matrix(state, subsys)
    # Compute eigenvalues
    evs = qml.math.eigvalsh(rho)
    # Clip for numerical stability
    evs = pnp.clip(evs, 1e-12, 1.0)
    return float(-pnp.sum(evs * pnp.log(evs)))


# ───────────────────────────────
# 5) REDEFINE time_evolved_state ON THE FLY
# ───────────────────────────────


def get_time_evolution_qnode(n_qubits, hamiltonian, trotter_steps=1):
    """
    Return a QNode that, given a time t, prepares |+>^n, then applies
    e^{-i H t} with `trotter_steps` slices, and returns the state vector.
    """
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, interface="autograd")
    def evolve(t):
        # Prepare |+> on each qubit
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        # Apply Trotterized time evolution
        qml.templates.ApproxTimeEvolution(hamiltonian, t, trotter_steps)
        return qml.state()

    return evolve
