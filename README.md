# Quantum Geometry Project

This project explores the mapping between entanglement entropy of a quantum spin chain and discrete bulk geometry. It simulates quenches of various Hamiltonians using PennyLane and infers edge weights of a bulk tree via a graph neural network implemented with DGL.

## Features
- Modular architecture with separate modules for quantum simulation, models, and training.
- Support for multiple Hamiltonians (TFIM, XXZ) and easy extension.
- GPU acceleration via PyTorch when available.
- Training logs stored with TensorBoard.
- Save and load intermediate results.
- Jupyter notebooks in `notebooks/` provide examples of usage and visualisation.

## Usage
```bash
pip install -r requirements.txt
python run_experiment.py --n_qubits 8 --hamiltonian tfim --time 1.0 --logdir runs/demo
```
The script saves learned edge weights in `runs/demo/` and writes TensorBoard logs that can be visualized with:
```bash
tensorboard --logdir runs/demo
```
To generate publication-ready figures after running experiments:
```bash
python generate_figures.py --n_qubits 8 --hamiltonian tfim --steps 5 --t_max 3.14
```
Outputs are saved in `figures/` as PNG and PDF.

## Theory
A quenched spin chain develops entanglement across contiguous intervals. A graph neural network learns edge weights on a binary bulk tree so that minimal cuts reproduce these entropies. The resulting curvature correlates with boundary energy shifts, inspired by AdS/CFT intuition.

## Tests
Simple unit tests for entropy computation and GNN training are located in `tests/`.


## Phase 4 Features
- Causal perturbations via `quantum.perturb`
- Entanglement inversion metrics in `analysis/invertibility.py`
- Saliency and attention visualizations under `quantumproject.visualization`

## Advanced Analysis
- `analysis.geodesic` reconstructs geodesic distance matrices and 2D embeddings.
- `analysis.scaling` fits the scaling law ⟨d⟩ ∝ n^(1/D) to estimate spatial dimension.
- `analysis.invertibility.entropy_round_trip` tests entropy ↔ geometry fidelity.
- `quantum.perturb.perturb_time_series` visualizes causal propagation of boundary perturbations.
- `models.quantum_gnn.HybridQuantumGNN` provides a quantum–classical predictor.
- `utils.graph_topologies` and `utils.bulk_graph` support custom bulk graphs beyond the default binary tree.
