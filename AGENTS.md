Quantum Geometry Inference Pipeline — Agent Responsibilities
Last updated: Phase 5

This project simulates emergent discrete bulk geometry from quantum entanglement dynamics using machine learning and graph theory. Each agent below encapsulates a conceptual or functional unit within the pipeline.

📏 EntropyAgent
Role: Measures entanglement structure
Input: Full state vector from quantum evolution
Output: Vector of von Neumann entropies over all contiguous boundary intervals

Responsibilities:
Evaluate the quantum state at multiple time steps.

Compute partial traces and entropy for all region intervals.

Provide entropy data to learning and evaluation agents.

🧠 GeometryAgent
Role: Learns bulk geometry from entanglement
Input: Interval entropy vector
Output: Edge weights for latent bulk graph (tree or general graph)

Responsibilities:
Train a neural network (MLP or quantum GNN) to predict graph edge weights.

Minimize the discrepancy between entanglement and cut-based predictions.

Support both forward and inverse reconstruction of entropy from geometry.

🌐 BulkTopologyAgent
Role: Manages bulk geometry graph structure
Input: Number of boundary qubits and topology type
Output: NetworkX graph representing the latent geometry

Responsibilities:
Generate balanced trees, small-world graphs, or loopy topologies.

Map boundary qubits to graph leaves.

Provide precomputed interval-to-mincut edge mappings.

⛓️ CutEvaluatorAgent
Role: Computes entropy proxies from learned geometry
Input: Edge weights and cut mappings
Output: Reconstructed entropies

Responsibilities:
Calculate min-cut edge sums per interval.

Evaluate round-trip reconstruction accuracy.

Support bidirectional testing (entropy ↔ geometry).

🧭 GeodesicAgent
Role: Infers spatial structure from learned graph
Input: Weighted bulk graph and boundary mapping
Output: Distance matrix (geodesics) and 2D spatial embedding

Responsibilities:
Compute shortest paths between all boundary pairs.

Derive spatial layouts via MDS or other dimensionality reduction.

Estimate emergent spatial dimension from distance scaling.

⏱️ CausalityAgent
Role: Tracks propagation of local perturbations
Input: Perturbed and baseline state evolutions
Output: Δ entropy, Δ curvature, and Δ energy vectors over time

Responsibilities:
Inject local boundary perturbations.

Measure time-dependent changes across modules.

Animate or analyze causal flow through the learned geometry.

⚛️ QuantumGNNAgent
Role: Learns geometry using a quantum-enhanced model
Input: Entropy vector
Output: Predicted edge weights

Responsibilities:
Encode entropy using angle/amplitude schemes.

Define and train a hybrid quantum-classical GNN.

Compare performance with classical counterparts.

📊 EvaluationAgent
Role: Performs model diagnostics and correlation studies
Input: Curvatures, ΔE shifts, and other observables
Output: Correlation metrics, plots, animations

Responsibilities:
Quantify Einstein-like correlations: curvature vs. boundary energy shifts.

Plot entropy reconstructions, geodesics, and causality maps.

Summarize performance across topologies and model types.

🗂️ DataAgent
Role: Handles persistent storage and output files
Responsibilities:

Save model weights, curvature maps, and entropy vectors.

Store visualizations, animations, and diagnostic results.

Organize outputs into versioned phase folders.
