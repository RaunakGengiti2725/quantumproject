"""Saliency maps for edge weight predictions."""

from __future__ import annotations

import os
import torch
import matplotlib.pyplot as plt


def saliency_heatmap(model: torch.nn.Module, ent: torch.Tensor, outdir: str = "figures/phase4"):
    """Compute gradient |d output / d input| and plot as heatmap."""
    ent = ent.clone().detach().requires_grad_(True)
    preds = model(ent)
    grad = torch.autograd.functional.jacobian(lambda x: model(x), ent)
    sal = grad.abs().detach().cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.imshow(sal, aspect="auto", cmap="magma")
    plt.xlabel("Interval")
    plt.ylabel("Edge")
    plt.colorbar(label="|d w / d S|")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "saliency_heatmap.png"))
    plt.close()
    return sal
