import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, zscore

from quantumproject.quantum.simulations import (
    Simulator,
    boundary_energy_delta,
    contiguous_intervals,
    von_neumann_entropy,
)
from quantumproject.training.pipeline import train_step
from quantumproject.utils.tree import BulkTree
from quantumproject.visualization.plots import (
    plot_bulk_tree,
    plot_bulk_tree_3d,
    plot_entropy_over_time,
    plot_weight_comparison,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication‐ready visualizations for quantum geometry"
    )
    parser.add_argument("--n_qubits", type=int, default=12, help="Number of qubits")
    parser.add_argument(
        "--hamiltonian",
        choices=["tfim", "xxz", "heisenberg"],
        default="xxz",
        help="Which Hamiltonian to use",
    )
    parser.add_argument(
        "--max_interval_size",
        type=int,
        default=2,
        help="Maximum interval length to use when training",
    )
    parser.add_argument("--steps", type=int, default=16, help="Number of time steps")
    parser.add_argument(
        "--t_max", type=float, default=np.pi, help="Maximum evolution time"
    )
    parser.add_argument(
        "--outdir", default="figures", help="Output directory for figures"
    )
    parser.add_argument(
        "--inject_noise",
        action="store_true",
        help="If ΔE is flat, inject small noise for plotting sanity checks",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    times = np.linspace(0.0, args.t_max, args.steps)

    sim = Simulator(args.n_qubits)
    H = sim.build_hamiltonian(args.hamiltonian)

    regions = contiguous_intervals(args.n_qubits, args.max_interval_size)
    tree = BulkTree(args.n_qubits)
    state0 = sim.time_evolved_state(H, 0.0)

    ent_series = []
    pearson_corrs = []
    spearman_corrs = []
    all_dE = []
    weights_last = None
    weight_history = []
    target_history = []

    for t in times:
        print(f"\n⏱️ Time step t = {t:.2f}")

        state_t = sim.time_evolved_state(H, t)
        entropies = np.array([von_neumann_entropy(state_t, r) for r in regions])
        ent_series.append(entropies)

        ent_torch = torch.tensor(entropies, dtype=torch.float32)
        weights, target = train_step(
            ent_torch,
            tree,
            writer=None,
            steps=100,
            max_interval_size=args.max_interval_size,
            return_target=True,
        )
        weights_last = weights.detach().cpu().numpy()
        target_last = target.detach().cpu().numpy()
        weight_history.append(weights_last)
        target_history.append(target_last)

        curvatures = tree.compute_curvatures(weights_last)
        dE = boundary_energy_delta(state0, state_t)
        all_dE.append(dE.copy())

        if args.inject_noise and np.allclose(dE, 0, atol=1e-6):
            print("  🔧 Injecting noise into ΔE for visualization sanity check.")
            dE += np.random.normal(0, 0.01, size=len(dE))

        leaves_map = tree.leaf_descendants()
        x_vals, y_vals = [], []

        for node, curv in curvatures.items():
            if tree.tree.degree[node] > 1:
                leaf_list = leaves_map.get(node, [])
                idxs = [int(name[1:]) for name in leaf_list]
                delta_sum = sum(dE[i] for i in idxs)
                x_vals.append(curv)
                y_vals.append(delta_sum)

        if len(x_vals) >= 2 and np.std(y_vals) > 1e-8:
            x_norm = zscore(x_vals)
            y_norm = zscore(y_vals)
            try:
                r_pearson, _ = pearsonr(x_norm, y_norm)
            except Exception:
                r_pearson = 0.0
            try:
                r_spearman, _ = spearmanr(x_vals, y_vals)
            except Exception:
                r_spearman = 0.0

            print(f"  📈 Pearson r = {r_pearson:.4f}, Spearman ρ = {r_spearman:.4f}")
            pearson_corrs.append(r_pearson)
            spearman_corrs.append(r_spearman)

            plt.figure(figsize=(5, 4), dpi=300)
            plt.scatter(
                x_vals, y_vals, c=y_vals, cmap="viridis", alpha=0.8, edgecolors="k"
            )
            plt.xlabel("Curvature", fontsize=12, fontweight="bold")
            plt.ylabel("ΔE sum", fontsize=12, fontweight="bold")
            plt.title(f"ΔE vs Curvature at t = {t:.2f}", fontsize=14)
            plt.colorbar(label="ΔE sum")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"scatter_t_{t:.2f}.png"))
            plt.close()
        else:
            print("  ⚠️ Flat ΔE → setting r = 0.0 for both Pearson and Spearman")
            pearson_corrs.append(0.0)
            spearman_corrs.append(0.0)

            plt.figure(figsize=(5, 4), dpi=300)
            plt.text(
                0.5,
                0.5,
                "Flat Data",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
            plt.xlabel("Curvature", fontsize=12, fontweight="bold")
            plt.ylabel("ΔE sum", fontsize=12, fontweight="bold")
            plt.title(f"ΔE vs Curvature at t = {t:.2f}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"scatter_t_{t:.2f}.png"))
            plt.close()

    ent_series = np.stack(ent_series)
    all_dE = np.stack(all_dE)
    np.save(os.path.join(args.outdir, "all_dE_series.npy"), all_dE)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.imshow(
        all_dE.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        extent=[times[0], times[-1], 0, args.n_qubits],
    )
    plt.colorbar(label="ΔE")
    plt.xlabel("Time", fontsize=12, fontweight="bold")
    plt.ylabel("Qubit index", fontsize=12, fontweight="bold")
    plt.title("ΔE spread over time", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "delta_E_heatmap.png"))
    plt.close()

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(
        times,
        pearson_corrs,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Pearson",
        color="#1f77b4",
    )
    plt.plot(
        times,
        spearman_corrs,
        marker="s",
        linestyle="-",
        linewidth=2,
        label="Spearman",
        color="#ff7f0e",
    )
    plt.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    plt.title(
        "Einstein Correlation: Pearson vs Spearman", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Time", fontsize=12, fontweight="bold")
    plt.ylabel("Correlation (r)", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "einstein_correlation_compare.png"))
    plt.close()

    if weights_last is not None:
        plot_bulk_tree(tree.tree, weights_last, args.outdir)
        plot_bulk_tree_3d(tree.tree, weights_last, args.outdir)
        plot_weight_comparison(target_history[-1], weight_history[-1], args.outdir)

    key_intervals = []
    for candidate in [(0, 1), (1, 2), (2, 3)]:
        if candidate in regions:
            key_intervals.append(candidate)

    series_dict = {r: ent_series[:, regions.index(r)] for r in key_intervals}
    plot_entropy_over_time(times, series_dict, args.outdir)


if __name__ == "__main__":
    main()
