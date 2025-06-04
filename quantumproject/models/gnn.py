"""Minimal graph neural network using adjacency matrices."""

from __future__ import annotations

import torch
import torch.nn as nn


class IntervalGNN(nn.Module):
    """Simple message passing network."""

    def __init__(
        self, num_intervals: int, num_edges: int, adj: torch.Tensor, hidden: int = 64
    ):
        super().__init__()
        self.adj = adj  # adjacency matrix [num_intervals, num_intervals]
        self.fc1 = nn.Linear(num_intervals, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, num_edges)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = torch.matmul(self.adj, x.T).T
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        out = torch.nn.functional.softplus(self.readout(h))
        return out.squeeze(0)
