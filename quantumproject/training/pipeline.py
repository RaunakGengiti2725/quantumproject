# mypy: ignore-errors
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from quantumproject.quantum.simulations import (
    contiguous_intervals,
    get_time_evolution_qnode,
    make_xxz_hamiltonian,
    von_neumann_entropy,
)
from quantumproject.utils.tree import BulkTree


def train_step(
    ent: torch.Tensor, tree: BulkTree, writer=None, steps: int = 10
) -> torch.Tensor:
    """Simple placeholder training loop used in unit tests."""

    mean_val = float(ent.mean())
    return torch.full((len(tree.edge_list),), mean_val)


# ─────────────────────────────────────────────────────────────
# 1) Interval‐to‐Edge MLP (unchanged except noise)
# ─────────────────────────────────────────────────────────────
class IntervalToEdgeMLP(nn.Module):
    """MLP mapping normalized entropy vector to edge weights.

    Architecture: Linear → ReLU → Linear → Softplus.
    """

    def __init__(self, num_intervals: int, num_edges: int):
        super().__init__()
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(num_intervals, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_edges),
            nn.Softplus(),  # ensures positivity of weights
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (1, NUM_INTERVALS)
        # returns shape (1, NUM_EDGES)
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# 2) train_model_for_time (increased dither noise)
# ─────────────────────────────────────────────────────────────
def train_model_for_time(
    tree: BulkTree,
    state_vector: np.ndarray,
    num_epochs: int = 500,
    lr: float = 1e-2,
    noise_scale: float = 1e-3,  # We keep noise_scale=1e-3
):
    """
    Train an IntervalToEdgeMLP to map normalized entropies → edge weights.
    - We use 20 Trotter steps in the QNode instead of just 5.
    - We keep noise_scale=1e-3 to break exact symmetry in entropies.
    """

    # A) Build list of all contiguous intervals
    n_qubits = tree.n_qubits
    INTERVALS = contiguous_intervals(n_qubits)
    NUM_INTERVALS = len(INTERVALS)
    NUM_EDGES = len(tree.edge_list)

    # B) Compute raw entropies
    raw_entropies_list: list[float] = []
    for region in INTERVALS:
        Si = von_neumann_entropy(state_vector, list(region))
        raw_entropies_list.append(float(Si))
    raw_entropies = np.array(raw_entropies_list, dtype=np.float32)

    # C) Normalize to mean=0, std=1
    mean_S = raw_entropies.mean()
    std_S = raw_entropies.std() if raw_entropies.std() > 1e-9 else 1.0
    norm_entropies = (raw_entropies - mean_S) / std_S

    # D) Add small Gaussian noise to break symmetry
    noise = np.random.normal(
        loc=0.0, scale=noise_scale, size=norm_entropies.shape
    ).astype(np.float32)
    norm_entropies = norm_entropies + noise

    ent_torch = torch.tensor(norm_entropies, dtype=torch.float32).unsqueeze(0)

    # E) Instantiate model & optimizer
    model = IntervalToEdgeMLP(NUM_INTERVALS, NUM_EDGES)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # F) Precompute interval cuts (return_indices=True) with 20 Trotter RKT
    interval_cuts = []
    for region in INTERVALS:
        try:
            cut_edge_indices = tree.interval_cut_edges(region, return_indices=True)
        except ValueError:
            cut_edge_indices = []
        interval_cuts.append(cut_edge_indices)

    # G) Define cut_loss
    def cut_loss(pred_w: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, cut_edges in enumerate(interval_cuts):
            if len(cut_edges) > 0:
                cut_sum = torch.sum(pred_w[cut_edges])
            else:
                cut_sum = torch.tensor(0.0, dtype=pred_w.dtype, device=pred_w.device)
            target_val = ent_torch[0, i]
            losses.append((cut_sum - target_val) ** 2)
        return torch.stack(losses).mean()

    # H) Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(ent_torch).squeeze(0)
        loss = cut_loss(preds)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"  [Epoch {epoch:4d}] loss = {loss.item():.6f}")

    return model(ent_torch).detach().squeeze(0).numpy()


# ─────────────────────────────────────────────────────────────
# 3) run_one_time_step (uses 20 Trotter steps)
# ─────────────────────────────────────────────────────────────
def run_one_time_step(
    tree: BulkTree,
    t: float,
    base_state: np.ndarray,
    ham_params: dict,
    num_epochs: int = 500,
):
    """
    1) Build XXZ Hamiltonian via make_xxz_hamiltonian(...).
    2) Create a 20‐step Trotter QNode (get_time_evolution_qnode).
    3) Evolve |+>^n → get state_t.
    4) Compute ΔE_i = ⟨Z_i⟩(t) - ⟨Z_i⟩(0) for each boundary qubit.
    5) Train MLP on entropies of state_t (train_model_for_time).
    6) Compute curvature κ_v = -∑_{e∈incident(v)} w_e for each internal v.
    """

    n = tree.n_qubits

    # A) XXZ with a nonzero transverse field h=1.0
    Jx = ham_params.get("Jx", 1.0)
    Jy = ham_params.get("Jy", 0.8)
    Jz = ham_params.get("Jz", 0.6)
    h_field = ham_params.get("h", 1.0)
    H_xxz = make_xxz_hamiltonian(n, Jx, Jy, Jz, h_field)

    # B) Use 20 Trotter steps instead of 5
    evolve = get_time_evolution_qnode(n_qubits=n, hamiltonian=H_xxz, trotter_steps=20)

    # C) Evolve state
    state_t = evolve(t)

    # D) Compute boundary ΔE
    def z_expectation(state_vec: np.ndarray, wire: int) -> float:
        psi = state_vec.reshape([2] * n)
        subsys = [wire]
        trace_out = [i for i in range(n) if i not in subsys]
        rho = np.tensordot(psi, psi.conj(), axes=(trace_out, trace_out))
        return float(
            np.real(np.trace(rho @ np.array([[1, 0], [0, -1]], dtype=complex)))
        )

    deltaE = []
    for i in range(n):
        E0 = z_expectation(base_state, i)
        Et = z_expectation(state_t, i)
        deltaE.append(Et - E0)

    # E) Train MLP with noise_scale=1e-3 (unchanged)
    learned_w = train_model_for_time(
        tree, state_t, num_epochs=num_epochs, lr=1e-2, noise_scale=1e-3
    )

    # F) Compute curvature
    curvatures = {}
    for v in tree.tree.nodes:
        if tree.tree.degree[v] > 1:
            incident = [tree.edge_to_index[(v, nbr)] for nbr in tree.tree.neighbors(v)]
            curvatures[v] = -np.sum(learned_w[incident])
        else:
            curvatures[v] = 0.0

    return curvatures, deltaE, learned_w
