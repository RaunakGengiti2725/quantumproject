import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import deque


def plot_bulk_tree(tree: nx.Graph, weights: np.ndarray, outdir: str):
    """
    2D Bulk Tree (Edge Weights):
    - Nodes relabeled 1,2,3,... instead of 'q0','v0', etc.
    - High-contrast 'plasma' colormap for edges.
    - Zoomed-out layout with a 15% margin so nodes don't sit at the very edge.
    - Figure window starts at ~800×600 pixels.
    """
    try:
        pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")
    except Exception:
        print("⚠️ pygraphviz not found — using spring_layout fallback.")
        pos = nx.spring_layout(tree, seed=0)

    # Relabel each node as an integer "1,2,3,..."
    node_list = list(tree.nodes())
    labels = {node: str(i + 1) for i, node in enumerate(node_list)}

    # Normalize weights for edge coloring
    norm = plt.Normalize(vmin=np.min(weights), vmax=np.max(weights))
    cmap = plt.cm.plasma
    edge_colors = [cmap(norm(w)) for w in weights]

    # ─── Create a smaller figure window ─────────────────────────────────────
    fig = plt.figure(figsize=(6, 4.5), dpi=100)  # ≈600×450 pixels by default
    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("800x600")  # Force window size (pixel) to 800×600
    except Exception:
        pass
    ax = plt.gca()
    ax.set_title("Bulk Tree (2D) — Edge Weights", fontsize=16, fontweight='bold')

    # Draw nodes (larger circles, semi-transparent fill)
    nx.draw_networkx_nodes(
        tree,
        pos,
        node_size=500,
        node_color="#eeeeee",
        edgecolors="#333333",
        linewidths=1.0,
        alpha=0.9,
        ax=ax,
    )
    # Draw edges (thick lines colored by weight)
    nx.draw_networkx_edges(
        tree,
        pos,
        edge_color=edge_colors,
        width=3,
        ax=ax,
    )
    # Draw labels (integers) in bold
    nx.draw_networkx_labels(
        tree,
        pos,
        labels=labels,
        font_size=12,
        font_family="sans-serif",
        font_weight='bold',
        ax=ax,
    )

    # Explicit colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Edge Weight", fontsize=12)

    ax.axis('off')

    # Add a 15% margin around the data so nodes are not squished at edges
    all_x = np.array([xy[0] for xy in pos.values()])
    all_y = np.array([xy[1] for xy in pos.values()])
    x_margin = (all_x.max() - all_x.min()) * 0.15
    y_margin = (all_y.max() - all_y.min()) * 0.15
    ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bulk_tree.png"))
    plt.savefig(os.path.join(outdir, "bulk_tree.svg"))
    plt.show()
    plt.close()


def plot_bulk_tree_3d(tree: nx.Graph, weights: np.ndarray, outdir: str = "figures"):
    """
    3D Bulk Tree (True Depth):
    - Nodes are arranged so that x = depth (Layer (X)), y & z form a grid, scaled by spread_factor.
    - Uses a common max_range across x, y, z to center and 'zoom out'.
    - Axis labels: Layer (X), Y Index, Height (Z).
    - Occupies 75% of figure width; colorbar on right occupying ~25%.
    - Larger, semi-transparent markers and thick, high-contrast edges.
    - Figure window starts at ~800×600 pixels.
    """
    # 1. Find the 'root' whose descendant leaves = all leaves
    descendants = _compute_leaf_descendants(tree)
    total_leaves = len([n for n in tree.nodes if tree.degree[n] == 1])
    root = None
    for node, leaf_list in descendants.items():
        if len(leaf_list) == total_leaves:
            root = node
            break
    if root is None:
        root = next((n for n in tree.nodes if tree.degree[n] > 1), list(tree.nodes)[0])

    # 2. Compute BFS depth from the root (shortest path length)
    depth_map = nx.single_source_shortest_path_length(tree, source=root)

    # 3. Arrange nodes: x = depth, y & z = grid index, then scale by spread_factor
    coords_by_depth = {}
    for node, depth in depth_map.items():
        coords_by_depth.setdefault(depth, []).append(node)

    spread_factor = 2.5  # Increase to spread out more along Y and Z
    pos3d = {}
    for depth, nodes_in_depth in coords_by_depth.items():
        m = len(nodes_in_depth)
        grid_size = int(np.ceil(np.sqrt(m)))
        for idx, node in enumerate(nodes_in_depth):
            raw_y = idx % grid_size
            raw_z = idx // grid_size
            pos3d[node] = (depth, raw_y * spread_factor, raw_z * spread_factor)

    xs = np.array([pos3d[n][0] for n in tree.nodes])
    ys = np.array([pos3d[n][1] for n in tree.nodes])
    zs = np.array([pos3d[n][2] for n in tree.nodes])

    # 4. Create a smaller figure window
    fig = plt.figure(figsize=(6, 4.5), dpi=100)  # ~600×450 pixels
    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("800x600")
    except Exception:
        pass
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Bulk Tree (True Depth)", fontsize=16, fontweight='bold')

    # 5. Draw nodes (larger, semi-transparent)
    ax.scatter(xs, ys, zs, s=120, c="#333333", alpha=0.85, depthshade=True)

    # 6. Draw edges (colored by weight)
    norm = plt.Normalize(vmin=np.min(weights), vmax=np.max(weights))
    cmap = plt.cm.plasma
    for idx, (u, v) in enumerate(tree.edges()):
        xu, yu, zu = pos3d[u]
        xv, yv, zv = pos3d[v]
        ax.plot(
            [xu, xv],
            [yu, yv],
            [zu, zv],
            color=cmap(norm(weights[idx])),
            linewidth=2.5,
            alpha=0.9,
        )

    # 7. Axis labels exactly as requested
    ax.set_xlabel("Layer (X)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Y Index", fontsize=12, fontweight='bold')
    ax.set_zlabel("Height (Z)", fontsize=12, fontweight='bold')

    # 8. Center & equalize axis limits with padding
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min())
    mid_x = (xs.max() + xs.min()) / 2
    mid_y = (ys.max() + ys.min()) / 2
    mid_z = (zs.max() + zs.min()) / 2

    half = max_range / 2 * 1.4  # 40% padding for breathing room
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_zlim(mid_z - half, mid_z + half)

    # 9. Shift the 3D axes left (occupy 75% of figure width)
    ax.set_position([0.05, 0.05, 0.75, 0.90])

    # 10. Attach a colorbar on the right, explicitly using this ax
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label="Edge Weight")

    # 11. Adjust view angle for a visually pleasing diagonal perspective
    ax.view_init(elev=25, azim=130)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "bulk_tree_3d.png"))
    plt.savefig(os.path.join(outdir, "bulk_tree_3d.svg"))
    plt.show()
    plt.close()


def plot_einstein_correlation(times: np.ndarray, correlations: list[float], outdir: str):
    """
    2D plot of Einstein correlation vs. time:
    - Larger markers, bold lines, dashed grid lines.
    """
    plt.figure(figsize=(6, 4.5), dpi=100)
    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("800x600")
    except Exception:
        pass

    plt.plot(times, correlations, marker="o", markersize=6, linestyle="-", linewidth=2, color="#1f77b4")
    plt.title("Einstein Correlation Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Time", fontsize=12, fontweight='bold')
    plt.ylabel("Correlation (r)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "einstein_correlation.png"))
    plt.savefig(os.path.join(outdir, "einstein_correlation.svg"))
    plt.show()
    plt.close()


def plot_entropy_over_time(times: np.ndarray, ent_dict: dict[tuple[int, ...], np.ndarray], outdir: str):
    """
    Plot entanglement entropy vs. time (shifted to zero):
    - Each curve’s minimum is subtracted so everything starts at 0.
    - Distinct 'viridis' colors, bold labels, and plain y-axis formatting.
    """
    plt.figure(figsize=(6, 4.5), dpi=100)
    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("800x600")
    except Exception:
        pass

    cmap = plt.cm.viridis
    num_series = len(ent_dict)
    for idx, (region, series) in enumerate(ent_dict.items()):
        shifted = series - np.min(series)  # baseline to zero
        color = cmap(idx / max(1, num_series - 1))
        label = f"[{region[0]}, {region[-1]}]"
        plt.plot(times, shifted, label=label, linewidth=2, color=color)

    plt.title("Entropy Dynamics Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Time", fontsize=12, fontweight='bold')
    plt.ylabel("Entanglement Entropy (shifted)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.ticklabel_format(style='plain', axis='y')  # force plain formatting on y-axis
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "entropy_over_time.png"))
    plt.savefig(os.path.join(outdir, "entropy_over_time.svg"))
    plt.show()
    plt.close()


def _compute_leaf_descendants(tree: nx.Graph) -> dict[str, list[str]]:
    """
    For each internal node, return a list of leaf-nodes reachable from it.
    Leaves are nodes of degree 1.
    """
    descendants = {}
    leaves = {n for n in tree.nodes if tree.degree[n] == 1}

    for node in tree.nodes:
        if tree.degree[node] == 1:
            continue
        visited = {node}
        queue = deque([node])
        node_leaves = []

        while queue:
            curr = queue.popleft()
            for nbr in tree.neighbors(curr):
                if nbr in visited:
                    continue
                visited.add(nbr)
                if nbr in leaves:
                    node_leaves.append(nbr)
                else:
                    queue.append(nbr)
        descendants[node] = node_leaves

    return descendants
