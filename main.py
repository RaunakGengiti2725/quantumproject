import os
import argparse
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt

# ───────────────────────── CONFIGURATION ─────────────────────────
CFG = {
    "n_qubits": 8,             # Number of boundary qubits (power of 2)
    "n_layers": 2,             # Layers in the small MERA-like “quenched” circuit
    "time_steps": 5,           # Number of discrete time points to sample
    "t_max": np.pi,            # Max evolution time
    "lr": 1e-2,                # Learning rate for GNN‐MLP
    "epochs": 500,             # Training epochs per time step
    "save_dir": "qemar_dyn_gnn"  # Output directory
}
os.makedirs(CFG["save_dir"], exist_ok=True)

# Validate n_qubits is power of 2
assert CFG["n_qubits"] & (CFG["n_qubits"] - 1) == 0, "n_qubits must be a power of 2"
TREE_HEIGHT = int(np.log2(CFG["n_qubits"]))

# PennyLane device for state evolution and measurements
dev_q = qml.device("default.qubit", wires=CFG["n_qubits"], shots=None)

# ───────────────────────── HAMILTONIAN & TIME EVOLUTION ─────────────────────────
# Example: transverse-field Ising (TFIM) Hamiltonian on a ring with periodic boundaries
def make_tfim_hamiltonian(n, J=1.0, h=1.0):
    """Construct TFIM Hamiltonian: H = -J sum Z_i Z_{i+1} - h sum X_i."""
    coeffs = []
    ops = []
    for i in range(n):
        zi = qml.PauliZ(i)
        znext = qml.PauliZ((i + 1) % n)
        coeffs.append(-J)
        ops.append(zi @ znext)
    for i in range(n):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)

H_tfim = make_tfim_hamiltonian(CFG["n_qubits"], J=1.0, h=1.0)

@qml.qnode(dev_q, interface="autograd")
def time_evolved_state(t):
    """
    Prepare |+>^n initial state, then apply e^{-i H t} to simulate a quench.
    Returns full state vector.
    """
    # Prepare |+> on each qubit
    for i in range(CFG["n_qubits"]):
        qml.Hadamard(wires=i)
    # Approximate time evolution with one Trotter step (for simplicity)
    qml.templates.ApproxTimeEvolution(H_tfim, t, 1)
    return qml.state()

# ───────────────────────── REGION ENUMERATION ─────────────────────────
def contiguous_intervals(n):
    """Return list of all contiguous intervals (as tuples) on [0..n-1]."""
    regions = []
    for length in range(1, n):
        for start in range(0, n - length + 1):
            regions.append(tuple(range(start, start + length)))
    return regions

REGIONS = contiguous_intervals(CFG["n_qubits"])  # 28 intervals for n=8

# ───────────────────────── PARTIAL-TRACE & ENTROPY ─────────────────────────
def reduced_density_matrix(state, subsys):
    """
    Given full state vector (2^n), compute reduced density matrix on 'subsys' qubit indices.
    """
    n = CFG["n_qubits"]
    dims = [2] * n
    psi = pnp.reshape(state, dims)
    keep = list(subsys)
    trace_out = [i for i in range(n) if i not in keep]
    rho = pnp.tensordot(psi, pnp.conj(psi), axes=(trace_out, trace_out))
    dim_sub = 2 ** len(subsys)
    return pnp.reshape(rho, (dim_sub, dim_sub))

def von_neumann_entropy(state, subsys):
    """Compute von Neumann entropy of 'subsys' from full state vector."""
    rho = reduced_density_matrix(state, subsys)
    evs = qml.math.eigvalsh(rho)
    evs = pnp.clip(evs, 1e-9, 1)
    return float(-pnp.sum(evs * pnp.log(evs)))

# ───────────────────────── BUILD BULK TREE & CUT PRECOMPUTATION ─────────────────────────
# Build balanced binary tree with n_leaf = n_qubits
bulk_tree = nx.balanced_tree(r=2, h=TREE_HEIGHT)  # nodes = 2^(h+1)-1
# Identify leaves (degree=1, excluding the root if it happens to have degree=1)
leaves = sorted([node for node in bulk_tree.nodes if bulk_tree.degree[node] == 1 and node != 0])
assert len(leaves) == CFG["n_qubits"], "Leaf count must match n_qubits"
# Map leaf node -> qubit index (0..n_qubits-1)
leaf_to_qubit = {leaf: idx for idx, leaf in enumerate(leaves)}

# Build edge list and mapping to index
edge_list = list(bulk_tree.edges)
edge_to_index = {edge: i for i, edge in enumerate(edge_list)}
# For undirected graph, ensure both (u,v) and (v,u) map to same index
edge_to_index.update({(v, u): i for (u, v), i in edge_to_index.items()})

# Precompute for each contiguous interval which edges form the minimum cut on the unweighted tree
from functools import lru_cache

@lru_cache(maxsize=None)
def interval_cut_edges(interval):
    """
    Given interval (tuple of qubit indices), return list of edge indices in the minimal edge cut
    separating those leaves from the rest.
    """
    qubits = list(interval)
    # Find corresponding leaf nodes
    region_leaves = [leaf for leaf, q in leaf_to_qubit.items() if q in qubits]
    outside_leaves = [leaf for leaf, q in leaf_to_qubit.items() if q not in qubits]
    min_cut = float("inf")
    best_edges = None
    # Compute minimum cut on unweighted tree by checking every pair (u, v)
    for u in region_leaves:
        for v in outside_leaves:
            cut_edges = nx.minimum_edge_cut(bulk_tree, u, v)
            if len(cut_edges) < min_cut:
                min_cut = len(cut_edges)
                best_edges = cut_edges
    # Convert edge set to indices
    return [edge_to_index[(u, v)] for (u, v) in best_edges]

NUM_INTERVALS = len(REGIONS)              # 28 for n=8
NUM_EDGES = len(edge_list)               # 14 for a height-3 binary tree

# Build a list of edge-index lists for each interval
INTERVAL_CUTS = [interval_cut_edges(region) for region in REGIONS]  # list of 28 lists

# ───────────────────────── GNN-MLP MODEL DEFINITION ─────────────────────────
class IntervalToEdgeMLP(nn.Module):
    """
    Simple MLP that maps a vector of interval entropies (length NUM_INTERVALS)
    to a vector of predicted edge weights (length NUM_EDGES).
    """
    def __init__(self):
        super().__init__()
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(NUM_INTERVALS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_EDGES),
            nn.Softplus()  # ensures all weights are positive
        )

    def forward(self, x):
        # x: (batch_size=1, NUM_INTERVALS)
        return self.net(x)  # returns (1, NUM_EDGES)

# Instantiate model & optimizer (we will reset per time step)
def create_model_and_optimizer():
    model = IntervalToEdgeMLP()
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    return model, optimizer

# ───────────────────────── LOSS FUNCTION ─────────────────────────
def cut_loss(pred_weights, entropies):
    """
    Given predicted edge weights (torch Tensor of shape [NUM_EDGES])
    and entropies (Tensor [NUM_INTERVALS]), compute MSE loss:
      sum_{interval i} ( sum_{e in cut(i)} w_e  - S_i )^2
    """
    # pred_weights: shape [NUM_EDGES]
    # entropies: shape [NUM_INTERVALS]
    losses = []
    for i, edges in enumerate(INTERVAL_CUTS):
        # sum of weights for those edges
        cut_sum = torch.sum(pred_weights[edges])
        losses.append((cut_sum - entropies[i]) ** 2)
    return torch.stack(losses).mean()

# ───────────────────────── BOUNDARY ENERGY PROXY ─────────────────────────
@qml.qnode(dev_q, interface="autograd")
def z_expectation(state_vector, wire):
    """
    QNode that, given a full n-qubit state vector, returns ⟨Z⟩ on ``wire``.
    PennyLane 0.41 removed ``QubitStateVector`` so we use ``StatePrep`` for
    state-vector initialization.
    """
    # ``StatePrep`` handles both normalization and padding if required.
    qml.StatePrep(state_vector, wires=range(CFG["n_qubits"]))
    return qml.expval(qml.PauliZ(wire))

def boundary_energy_delta(state_base, state_t):
    """
    Compute ΔE_i = <Z_i>(t) - <Z_i>(0) for each boundary qubit i.
    Returns list of length n_qubits.
    """
    deltas = []
    for i in range(CFG["n_qubits"]):
        E0 = z_expectation(state_base, i)
        Et = z_expectation(state_t, i)
        deltas.append(float(Et - E0))
    return deltas

# ───────────────────────── MAIN PIPELINE ─────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "test"], default="run")
    args = parser.parse_args()

    if args.mode == "test":
        # Simple sanity check: Bell state entropy
        dev2 = qml.device("default.qubit", wires=2)
        @qml.qnode(dev2)
        def bell():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()
        s = bell()
        S = von_neumann_entropy(s, [0])
        assert abs(S - np.log(2)) < 1e-6, "Bell entropy test failed"
        print("✅ Bell entropy test passed.")
        return

    # Sample times
    t_list = np.linspace(0, CFG["t_max"], CFG["time_steps"])

    # Precompute the “base” state at t=0
    state_0 = time_evolved_state(0.0)
    # Precompute its entropies for all contiguous intervals
    entropies_0 = np.array([von_neumann_entropy(state_0, list(r)) for r in REGIONS])

    # Initialize lists to store final correlations
    correlations = []

    for t_idx, t_val in enumerate(t_list):
        print(f"\n=== Time Step {t_idx+1}/{len(t_list)}, t = {t_val:.3f} ===")
        # 1) Compute time-evolved state
        state_t = time_evolved_state(t_val)

        # 2) Compute entropies on all contiguous intervals
        entropies_t = np.array([von_neumann_entropy(state_t, list(r)) for r in REGIONS])
        # Convert to torch Tensor
        ent_torch = torch.tensor(entropies_t, dtype=torch.float32).unsqueeze(0)  # shape [1, NUM_INTERVALS]

        # 3) Train MLP to predict edge weights so that cuts ≈ entropies
        model, optimizer = create_model_and_optimizer()
        model.train()
        for epoch in range(CFG["epochs"]):
            optimizer.zero_grad()
            pred_w = model(ent_torch).squeeze(0)  # shape [NUM_EDGES]
            loss = cut_loss(pred_w, ent_torch.squeeze(0))
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"  [Epoch {epoch:4d}] loss = {loss.item():.6f}")
        print(f"  Final training loss = {loss.item():.6f}")

        # Get learned edge weights
        learned_weights = model(ent_torch).detach().squeeze(0).numpy()  # shape [NUM_EDGES]

        # 4) Compute discrete curvature at internal nodes
        # Simplest proxy: κ_v = - sum(weights of edges incident on v) for internal v
        curvatures = {}
        for v in bulk_tree.nodes:
            deg = bulk_tree.degree[v]
            if deg > 1:  # internal node
                incident_edges = [edge_to_index[(v, nbr)] for nbr in bulk_tree.neighbors(v)]
                curvatures[v] = -np.sum(learned_weights[incident_edges])
            else:
                curvatures[v] = 0.0  # leaves

        # 5) Compute boundary energy shifts ΔE_i = <Z_i>(t) - <Z_i>(0)
        deltas = boundary_energy_delta(state_0, state_t)  # list of length n_qubits

        # 6) For each internal node v, sum ΔE_i over leaves under v
        # Precompute leaves under each internal node
        leaves_under = {}
        for v in bulk_tree.nodes:
            if bulk_tree.degree[v] > 1:
                # Induce subtree rooted at v: collect leaves reachable without passing through parent
                # but easiest is: in tree, find all nodes in component after removing v's parent edge
                # Instead, do: remove v’s incident edges and see which leaf sets are separated?
                # Simpler: do a DFS from v that only visits descendants in the subtree "below" v.
                # But the tree is unrooted; pick an arbitrary root (say 0) and build a rooted tree.
                pass
        # Instead, build a rooted version with root=0
        rooted = nx.bfs_tree(bulk_tree, source=0)
        # Build mapping: for each node, the set of leaf descendants
        leaf_desc = {}
        def dfs_collect(u):
            if u in leaf_to_qubit:
                return {leaf_to_qubit[u]}
            s = set()
            for w in rooted.successors(u):
                s |= dfs_collect(w)
            return s
        for v in rooted.nodes:
            leaf_desc[v] = dfs_collect(v)

        # Now, sum ΔE_i for leaves under each internal node (excluding root if desired)
        corr_pairs = []
        for v in bulk_tree.nodes:
            if bulk_tree.degree[v] > 1:
                leaves_set = leaf_desc[v]
                delta_sum = sum(deltas[i] for i in leaves_set)
                corr_pairs.append((curvatures[v], delta_sum))
        corr_arr = np.array(corr_pairs)  # shape [num_internal, 2]
        # Compute Pearson correlation coefficient
        if corr_arr.size > 0:
            x, y = corr_arr.T
            if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                r, _ = pearsonr(x, y)
            else:
                r = 0.0
        else:
            r = np.nan
        print(f"  Pearson correlation (curvature vs. ΔE) = {r:.3f}")
        correlations.append((t_val, r))

        # Save learned edge weights and curvatures for this time
        np.save(os.path.join(CFG["save_dir"], f"weights_t{t_idx}.npy"), learned_weights)
        np.save(os.path.join(CFG["save_dir"], f"curvatures_t{t_idx}.npy"), np.array(list(curvatures.values())))

    # After looping over time steps, plot correlation vs. t
    times, r_vals = zip(*correlations)
    plt.figure(figsize=(5, 4))
    plt.plot(times, r_vals, 'o-', color='blue')
    plt.xlabel("Time t")
    plt.ylabel("Pearson r (κ vs. ΔE)")
    plt.title("Discrete Einstein‑Equation Correlation Over Time")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(CFG["save_dir"], "einstein_correlation.png"), dpi=150)
    plt.close()

    print("\n🚀 Dynamic GNN‑inferred bulk geometry study complete.")
    print(f"Results saved in: {CFG['save_dir']}")

if __name__ == "__main__":
    main()
