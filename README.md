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

## Curvature & Energy Analysis
The module `curvature_energy_analysis.py` provides utilities to study how local
curvature relates to boundary energy shifts on very large graphs.

### Installation
Install Python dependencies:
```bash
pip install -r requirements.txt
```

### API
```python
from curvature_energy_analysis import (
    compute_curvature,
    compute_energy_deltas,
    safe_pearson_correlation,
)
```
- `compute_curvature(graph)` – return per-node scalar curvature as a NumPy array.
- `compute_energy_deltas(graph, attr="delta_energy")` – sum edge energy deltas
  touching each node.
- `safe_pearson_correlation(x, y)` – correlation with cleaning and robust
  fallback returning `(r, p)`.

### Example
```python
import networkx as nx
from curvature_energy_analysis import compute_curvature, compute_energy_deltas, safe_pearson_correlation

g = nx.path_graph(8)
for u, v in g.edges():
    g[u][v]["delta_energy"] = 0.5
curv = compute_curvature(g)
delta = compute_energy_deltas(g)
r, p = safe_pearson_correlation(curv, delta)
print(r, p)
```

### Benchmark
Run the built-in benchmark on a synthetic graph:
```bash
python curvature_energy_analysis.py --nodes 100000 --p 1e-5
```
The script reports timings for curvature computation, energy aggregation, and
correlation evaluation. On a modern workstation the analysis on a 100k-node
graph finishes in a few seconds.
