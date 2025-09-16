import argparse, torch, numpy as np, os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

def node_label(n):
    if isinstance(n, dict):
        typ = n.get("type", n.get("kind", "node"))
        L   = n.get("layer", n.get("L", "?"))
        H   = n.get("head",  n.get("H", n.get("index","?")))
        if typ in ("attn","attention","head"):
            return f"L{L}H{H}"
        elif typ in ("mlp","ff","mlp_in","mlp_out"):
            return f"L{L}-MLP"
        else:
            return f"{typ}@L{L}:{H}"
    elif isinstance(n, (list, tuple)):
        return " ".join(map(str, n))
    return str(n)

def get_layer(n):
    if isinstance(n, dict):
        return n.get("layer", n.get("L", 0))
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--out_png", default=None)
    ap.add_argument("--out_pdf", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    data = torch.load(args.pt, map_location="cpu")
    src_nodes = data["src_nodes"]
    dst_nodes = data["dst_nodes"]
    scores    = np.asarray(data["edges_scores"], dtype=float)
    inc       = data["edges_in_graph"]

    # normalize 'inc' to indices of included edges
    if isinstance(inc, (list, tuple)):
        inc = np.array(inc)
    if hasattr(inc, "dtype") and getattr(inc, "dtype", None) == torch.bool:
        inc = inc.numpy()
    if isinstance(inc, np.ndarray):
        idx = np.where(inc)[0] if inc.dtype == bool and inc.shape[0]==scores.shape[0] else inc.astype(int)
    else:
        idx = np.array(inc, dtype=int)

    scores_inc = scores[idx]
    order = np.argsort(-scores_inc)[: args.topk]
    idx_top = idx[order]

    G = nx.DiGraph()
    for i in idx_top:
        s, t, w = src_nodes[i], dst_nodes[i], float(scores[i])
        sl, tl = node_label(s), node_label(t)
        G.add_node(sl, layer=get_layer(s))
        G.add_node(tl, layer=get_layer(t))
        G.add_edge(sl, tl, weight=w)

    # layered layout by layer (left->right), fallback to spring if no layer info
    layers = {}
    for n, d in G.nodes(data=True):
        L = d.get("layer", 0)
        layers.setdefault(L, []).append(n)
    if len(layers) > 1:
        # simple manual layered layout
        pos = {}
        xs = sorted(layers.keys())
        for xi, L in enumerate(xs):
            ys = layers[L]
            for yi, n in enumerate(sorted(ys)):
                pos[n] = (xi, -yi)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=200)

    plt.figure(figsize=(12, 8))
    widths = [max(0.5, 6.0*G[u][v]["weight"]/max(1e-6, max(nx.get_edge_attributes(G, "weight").values()))) for u,v in G.edges()]
    nx.draw(G, pos,
            with_labels=True, node_size=900, font_size=8,
            arrows=True, width=widths)
    title = args.label or Path(args.pt).stem
    plt.title(title)

    if args.out_png or args.out_pdf:
        outdir = args.out_png or args.out_pdf
        Path(os.path.dirname(outdir)).mkdir(parents=True, exist_ok=True)
    if args.out_png:
        plt.savefig(args.out_png, dpi=200, bbox_inches="tight")
        print("Wrote", args.out_png)
    if args.out_pdf:
        plt.savefig(args.out_pdf, bbox_inches="tight")
        print("Wrote", args.out_pdf)
    plt.show()

if __name__ == "__main__":
    main()
