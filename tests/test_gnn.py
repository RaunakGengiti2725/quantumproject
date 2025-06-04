import torch

from quantumproject.training.pipeline import train_step
from quantumproject.utils.tree import BulkTree


def test_train_step():
    ent = torch.rand(4)
    tree = BulkTree(4)
    w = train_step(ent, tree, writer=None, steps=10)
    assert w.shape[0] == len(tree.edge_list)
