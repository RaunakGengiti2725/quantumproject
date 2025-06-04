"""Training routines for entanglement-to-curvature mapping."""

from __future__ import annotations

from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter

from ..models.gnn import IntervalGNN
from ..utils.tree import BulkTree


def build_adjacency(num_intervals: int) -> torch.Tensor:
    """Return adjacency matrix connecting all interval nodes."""
    adj = torch.ones(num_intervals, num_intervals)
    adj.fill_diagonal_(0)
    return adj


def cut_loss(
    pred_weights: torch.Tensor, entropies: torch.Tensor, cuts: List[List[int]]
) -> torch.Tensor:
    losses = []
    for i, edges in enumerate(cuts):
        cut_sum = pred_weights[edges].sum()
        losses.append((cut_sum - entropies[i]) ** 2)
    return torch.stack(losses).mean()


def train_step(
    entropies: torch.Tensor, tree: BulkTree, writer: SummaryWriter, steps: int = 500
) -> torch.Tensor:
    adj = build_adjacency(len(entropies))
    model = IntervalGNN(len(entropies), len(tree.edge_list), adj)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    cuts = [
        tree.interval_cut_edges(tuple(range(i, i + 1))) for i in range(len(entropies))
    ]
    for epoch in range(steps):
        model.train()
        opt.zero_grad()
        pred = model(entropies)
        loss = cut_loss(pred, entropies, cuts)
        loss.backward()
        opt.step()
        if writer is not None and epoch % 10 == 0:
            writer.add_scalar("loss", loss.item(), epoch)
    return pred.detach()
