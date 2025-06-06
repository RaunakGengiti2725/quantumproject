import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx

from quantumproject.quantum.simulations import contiguous_intervals
from quantumproject.utils.tree import BulkTree

# … (import any other needed modules for your GNN) …


class IntervalGNN(nn.Module):
    """
    Placeholder GNN that takes:
     - n_intervals  = number of intervals (should equal len(contiguous_intervals(n_qubits)))
     - n_edges      = number of edges in the BulkTree
     - adj_matrix   = adjacency matrix encoding which intervals connect to which edges
    You can replace this with your actual GNN definition.
    """
    def __init__(self, n_intervals: int, n_edges: int, adj: torch.Tensor):
        super().__init__()
        # Example: one linear layer mapping interval representation → edge weights
        self.lin = nn.Linear(n_intervals, n_edges)

    def forward(self, ent_intervals: torch.Tensor):
        # ent_intervals has shape [n_intervals], output should be [n_edges]
        return self.lin(ent_intervals.unsqueeze(0)).squeeze(0)


def train_step(
    ent_torch: torch.Tensor,
    tree: BulkTree,
    writer=None,
    steps: int = 100,
    max_interval_size: int | None = None,
    return_target: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Trains a simple IntervalGNN to predict edge weights from entropies of contiguous intervals.

    Parameters
    ----------
    ent_torch:
        1D Tensor of length ``n_intervals`` containing von Neumann entropies.
    tree:
        ``BulkTree`` describing the geometry.
    steps:
        Number of training iterations.
    max_interval_size:
        If given, only intervals up to this length are used (useful for large systems).
    return_target:
        If ``True``, also return the target weights used during training.

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        Learned edge weights, and optionally the target weights.
    """

    n_qubits = tree.n_qubits

    # 1. Compute all contiguous intervals for n_qubits. If ent_torch supplies
    #    fewer values than the number of intervals, slice to match so the
    #    network input dimension agrees with the provided entropies.  This keeps
    #    the function usable with toy data in tests where only single-qubit
    #    entropies are given.

    all_intervals = contiguous_intervals(n_qubits, max_interval_size)

    if ent_torch.ndim == 1 and ent_torch.shape[0] != len(all_intervals):
        intervals = all_intervals[: ent_torch.shape[0]]
    else:
        intervals = all_intervals

    n_intervals = len(intervals)
    n_edges = len(tree.edge_list)

    # 2. Build adjacency: which interval “touches” which edge?
    #    adj[i, j] = 1 if interval i and edge j share any leaf
    adj = torch.zeros((n_intervals, n_edges), dtype=torch.float32)
    for i, interval in enumerate(intervals):
        # Each interval maps to a set of leaves, e.g. (2,3) → {"q2","q3"}
        leaf_names = {f"q{k}" for k in interval}
        for j, (u, v) in enumerate(tree.edge_list):
            # That edge “touches” any leaf in leaf_names if either u or v is an ancestor of those leaves
            # We check: does the subtree under that edge contain any leaf in leaf_names?
            # A quick way: remove that edge, see if any leaf in leaf_names is disconnected from the other side
            tree.tree.remove_edge(u, v)
            comps = list(nx.connected_components(tree.tree))
            tree.tree.add_edge(u, v)

            # If a leaf in leaf_names is in one component and not all in the same, then that edge cuts through this interval
            for comp in comps:
                if leaf_names.issubset(comp) and not leaf_names.issubset(set.union(*[c for c in comps if c != comp])):
                    adj[i, j] = 1.0
                    break

    # 3. Initialize GNN and optimizer
    model = IntervalGNN(n_intervals, n_edges, adj)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 4. Train to minimize difference between predicted “edge weights” and some target. 
    #    Here, for illustration, we pretend the “target weights” are all zeros + a small noise.
    #    In a real use-case, you’d have a ground‐truth assignment.
    #    We simply train the GNN to reproduce its own entropies so we get some nontrivial output.
    target = torch.randn(n_edges) * 0.1  # dummy target; replace with real target if you have one

    for _ in range(steps):
        optimizer.zero_grad()
        preds = model(ent_torch)
        loss = loss_fn(preds, target)
        loss.backward()
        optimizer.step()

    # 5. Return the learned weights (and optionally the training target)
    if return_target:
        return preds.detach(), target.detach()
    return preds.detach()


# No other functions in pipeline.py import BulkTree to avoid circular imports.
