import argparse
import os

import numpy as np

from quantumproject.training.pipeline import run_one_time_step
from quantumproject.utils.tree import BulkTree


def main():
    parser = argparse.ArgumentParser(description="Run quantum bulk–boundary experiment")
    parser.add_argument(
        "--n_qubits", type=int, default=8, help="Number of boundary qubits (power of 2)"
    )
    parser.add_argument(
        "--time_steps", type=int, default=5, help="Number of time points to sample"
    )
    parser.add_argument(
        "--t_max", type=float, default=np.pi, help="Maximum evolution time"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qemar_dyn_gnn",
        help="Directory to save weights, curvatures, and ΔE arrays",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs per time step",
    )
    args = parser.parse_args()

    n_qubits = args.n_qubits
    time_steps = args.time_steps
    t_max = args.t_max
    output_dir = args.output_dir
    num_epochs = args.epochs

    os.makedirs(output_dir, exist_ok=True)

    # 1) Build the bulk tree
    tree = BulkTree(n_qubits=n_qubits)

    # 2) XXZ Hamiltonian parameters (with h=1.0)
    ham_params = {"Jx": 1.0, "Jy": 0.8, "Jz": 0.6, "h": 1.0}

    # 3) Prepare base state at t=0 using 20 Trotter steps
    from quantumproject.quantum.simulations import (
        get_time_evolution_qnode,
        make_xxz_hamiltonian,
    )

    H_xxz = make_xxz_hamiltonian(n_qubits, **ham_params)
    evolve0 = get_time_evolution_qnode(
        n_qubits=n_qubits, hamiltonian=H_xxz, trotter_steps=20
    )
    base_state = evolve0(0.0)

    # 4) Sample times between 0 and t_max
    t_list = np.linspace(0.0, t_max, time_steps)

    correlations = []
    for idx, t_val in enumerate(t_list):
        print(f"=== Running time step {idx+1}/{time_steps}, t = {t_val:.4f} ===")

        # 5) Run the pipeline with 20 Trotter steps
        curvatures, deltaE, learned_w = run_one_time_step(
            tree,
            t_val,
            base_state,
            ham_params=ham_params,
            num_epochs=num_epochs,
        )

        # 6) Save weights, curvatures, ΔE
        np.save(os.path.join(output_dir, f"weights_t{idx}.npy"), learned_w)
        np.save(
            os.path.join(output_dir, f"curvatures_t{idx}.npy"),
            np.array(list(curvatures.values())),
        )
        np.save(os.path.join(output_dir, f"deltaE_t{idx}.npy"), np.array(deltaE))

        # 7) Skip t=0
        if idx == 0:
            print("  Skipping Pearson correlation at t=0 (all values constant).")
            correlations.append((t_val, np.nan))
            continue

        # 8) Identify internal nodes (degree > 1)
        internal_nodes = [v for v in tree.tree.nodes if tree.tree.degree[v] > 1]

        # 9) Precompute leaf_descendants (returns dict: node → list of "qX" names)
        all_leaf_desc = tree.leaf_descendants()

        # 10) Convert leaf names "qX" → int X, sum ΔE
        delta_sums = []
        for v in internal_nodes:
            leaf_names = all_leaf_desc[v]
            qubit_idxs = [int(name[1:]) for name in leaf_names]
            sum_DE = sum(deltaE[q] for q in qubit_idxs)
            delta_sums.append(sum_DE)

        curv_list = np.array([curvatures[v] for v in internal_nodes])
        de_list = np.array(delta_sums)

        # 11) Compute Pearson r if variances > 0
        if np.std(curv_list) > 1e-9 and np.std(de_list) > 1e-9:
            r = np.corrcoef(curv_list, de_list)[0, 1]
        else:
            r = np.nan
        print(f"  Pearson correlation (curvature vs. ΣΔE_under_node) = {r:.4f}")
        correlations.append((t_val, r))

    # 12) Plot correlations vs. time (skip nan)
    times = [t for t, r in correlations if not np.isnan(r)]
    r_vals = [r for _, r in correlations if not np.isnan(r)]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.plot(times, r_vals, "o-", color="blue", label="Pearson r")
    plt.xlabel("Time t")
    plt.ylabel("Pearson r (curvature vs. ΣΔE)")
    plt.title("Discrete Einstein‐Equation Correlation Over Time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "einstein_correlation.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✅ Saved Einstein correlation plot to: {fig_path}")

    print("🏁 Experiment complete.")


if __name__ == "__main__":
    main()
