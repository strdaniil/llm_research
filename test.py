# visualize_circuit.py
import os, re, math
import torch, numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ---- path to the saved IOI circuit ----
PT_PATH = r"C:\Users\danii\OneDrive\Documents\GitHub\llm_research\formal-functional-dissociation\graphs\EAP-IG-inputs\gpt2\ewok-social-relations.pt"
assert os.path.exists(PT_PATH), f"Not found: {PT_PATH}"

d = torch.load(PT_PATH, map_location="cpu")

src_nodes = [str(x) for x in d["src_nodes"]]
dst_nodes = [str(x) for x in d["dst_nodes"]]
scores    = d["edges_scores"].cpu().numpy()                 # [n_src, n_dst]
mask      = d["edges_in_graph"].cpu().numpy().astype(bool)  # [n_src, n_dst]

print("src_nodes:", len(src_nodes), "dst_nodes:", len(dst_nodes))
print("matrix shapes  scores:", scores.shape, " mask:", mask.shape)
print("nodes_in_graph:", int(d["nodes_in_graph"].sum()))

# ---- collect included edges (i,j) with their signed weight ----
inds = np.argwhere(mask)       # array of [ [i,j], ... ]
edges = [(int(i), int(j), float(scores[i, j])) for i, j in inds]

# keep only strongest edges for readability
TOP_K = 120
edges.sort(key=lambda x: abs(x[2]), reverse=True)
edges = edges[:TOP_K]

# ---- pretty labels & layout helpers ----
def pretty(n: str):
    if n == "input":  return "input"
    if n == "logits": return "logits"
    m = re.fullmatch(r"a(\d+)\.h(\d+)", n)   # e.g., a3.h6 -> L3H6
    if m: return f"L{int(m.group(1))}H{int(m.group(2))}"
    m = re.fullmatch(r"m(\d+)", n)           # e.g., m3    -> L3-MLP
    if m: return f"L{int(m.group(1))}-MLP"
    return n

def layer_of(label: str):
    if label == "input":  return -1
    if label == "logits": return  99
    m = re.search(r"L(\d+)", label)
    return int(m.group(1)) if m else 0

def band(label: str):
    s = label.lower()
    if s == "input":   return 0
    if "-mlp" in s:    return 2
    if "h" in s and "l" in s: return 1   # attention heads
    if s == "logits":  return 3
    return 4

# ---- build graph ----
G = nx.DiGraph()
for i, j, w in edges:
    u, v = pretty(src_nodes[i]), pretty(dst_nodes[j])
    G.add_node(u, layer=layer_of(u))
    G.add_node(v, layer=layer_of(v))
    G.add_edge(u, v, weight=w)

# layered-ish layout: x by layer, y by type band with slight spread
pos, buckets = {}, {}
for n, data in G.nodes(data=True):
    buckets.setdefault((data["layer"], band(n)), []).append(n)
for (L, B), ns in buckets.items():
    ns_sorted = sorted(ns)
    for k, n in enumerate(ns_sorted):
        pos[n] = (L, B + (k - (len(ns_sorted)-1)/2) * 0.18)

# edge widths ~ |weight|
ws = [abs(G[u][v]["weight"]) for u, v in G.edges()] or [1.0]
wmax = max(ws)
widths = [0.6 + 2.4*(w/wmax) for w in ws]
colors = ["tab:blue" if G[u][v]["weight"] >= 0 else "tab:red" for u, v in G.edges()]

plt.figure(figsize=(max(12, math.log2(max(30, G.number_of_edges()+1))), 8))
nx.draw(G, pos, with_labels=True, node_size=540, font_size=8,
        arrows=True, width=widths, edge_color=colors)
plt.title(f"IOI — GPT-2 circuit (top-{TOP_K} edges)")
plt.tight_layout()

out_png = os.path.join(os.path.dirname(PT_PATH), f"ioi_circuit_top{TOP_K}.png")
plt.savefig(out_png, dpi=200)
print("Saved:", out_png)
plt.show()
