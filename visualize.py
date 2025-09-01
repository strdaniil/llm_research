import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

def is_attn_node(n):
    # attention nodes look like "a{layer}.h{head}"
    # e.g., "a4.h7"
    return n.startswith("a") and (".h" in n)

def is_mlp_node(n):
    # mlp nodes look like "m{layer}"
    return n.startswith("m") and n[1:].isdigit()

def parse_attn(n):
    # returns (layer, head) for "a{L}.h{H}"
    # fall back returns None if not a valid attn node
    try:
        L = int(n.split(".")[0][1:])
        H = int(n.split(".")[1][1:])
        return L, H
    except Exception:
        return None

def parse_mlp(n):
    # returns layer for "m{L}"
    try:
        return int(n[1:])
    except Exception:
        return None

def main(pt_path):
    data = torch.load(pt_path, map_location="cpu")
    src_nodes = data["src_nodes"]
    dst_nodes = data["dst_nodes"]
    edge_scores = data["edges_scores"].detach().cpu().numpy()  # shape [N_src, N_dst]
    in_graph = data["edges_in_graph"].detach().cpu().numpy().astype(bool)  # same shape

    # We'll use absolute magnitude of scores and mask to included edges
    W = np.abs(edge_scores) * in_graph

    # --- 1) Aggregate importance by ATTENTION HEAD (outgoing edges) ---
    # For each src node that is an attention head, sum outgoing weights
    # Then project to (layers x heads) heatmap
    attn_nodes = [(i, n, parse_attn(n)) for i, n in enumerate(src_nodes) if is_attn_node(n)]
    if attn_nodes:
        max_layer = max(L for _, _, (L, H) in attn_nodes if (L is not None))
        max_head  = max(H for _, _, (L, H) in attn_nodes if (H is not None))
        heat = np.zeros((max_layer + 1, max_head + 1), dtype=float)
        for i, n, parsed in attn_nodes:
            if parsed is None: 
                continue
            L, H = parsed
            heat[L, H] += W[i, :].sum()  # outgoing contribution

    # --- 2) Aggregate importance by MLP layer (outgoing edges) ---
    mlp_nodes = [(i, n, parse_mlp(n)) for i, n in enumerate(src_nodes) if is_mlp_node(n)]
    mlp_layers = {}
    for i, n, L in mlp_nodes:
        if L is None:
            continue
        mlp_layers[L] = mlp_layers.get(L, 0.0) + W[i, :].sum()

    # --- 3) Top edges table ---
    # Flatten W and grab top-K entries
    K = 30
    flat = W.flatten()
    top_idx = np.argpartition(-flat, K)[:K]
    top_idx = top_idx[np.argsort(-flat[top_idx])]
    top_list = []
    n_src = len(src_nodes)
    for idx in top_idx:
        i = idx // W.shape[1]
        j = idx % W.shape[1]
        top_list.append((src_nodes[i], dst_nodes[j], float(W[i, j])))

    # === Plot ===
    fig_parts = 1
    if len(attn_nodes) > 0:
        fig_parts += 1
    if len(mlp_layers) > 0:
        fig_parts += 1

    fig = plt.figure(figsize=(12, 4*fig_parts))
    row = 1

    # Attention heatmap
    if len(attn_nodes) > 0:
        ax1 = plt.subplot(fig_parts, 1, row); row += 1
        im = ax1.imshow(heat, aspect="auto")
        ax1.set_title("Attention head importance (sum of outgoing edge magnitudes)")
        ax1.set_xlabel("Head")
        ax1.set_ylabel("Layer")
        plt.colorbar(im, ax=ax1)

    # MLP importance
    if len(mlp_layers) > 0:
        ax2 = plt.subplot(fig_parts, 1, row); row += 1
        layers = sorted(mlp_layers.keys())
        vals = [mlp_layers[L] for L in layers]
        ax2.bar(layers, vals)
        ax2.set_title("MLP importance by layer (sum of outgoing edge magnitudes)")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Importance")

    # Top edges text table
    ax3 = plt.subplot(fig_parts, 1, row)
    ax3.axis("off")
    lines = ["Top edges (|score|, included only):"]
    for s, d, w in top_list:
        lines.append(f"{s:>12}  ->  {d:<12}    {w:.4g}")
    ax3.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace")

    pt_path = Path(pt_path)
    fig.suptitle(f"{pt_path.stem} — {pt_path.parent.name}")
    out_png = pt_path.with_suffix(".viz.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"Saved visualization to: {out_png}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_eap_graph.py <path_to_graph.pt>")
        sys.exit(1)
    main(sys.argv[1])
