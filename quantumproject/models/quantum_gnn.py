"""Hybrid quantum-classical neural network using PennyLane."""

from __future__ import annotations

import torch
import pennylane as qml
import pennylane.numpy as pnp


class HybridQuantumGNN(torch.nn.Module):
    def __init__(self, n_intervals: int, n_edges: int, n_layers: int = 2):
        super().__init__()
        self.n_intervals = n_intervals
        self.n_edges = n_edges
        self.n_qubits = max(n_intervals, n_edges)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        weight_shapes = {"weights": (n_layers, self.n_qubits)}

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(self.n_intervals))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_edges)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x
        return torch.nn.functional.softplus(self.qlayer(x))
