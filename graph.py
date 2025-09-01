import torch
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Load your .pt file
path = r"C:\Users\danii\OneDrive\Documents\GitHub\llm_research\formal-functional-dissociation\graphs\EAP-IG-inputs\gpt2\ewok-social-relations.pt"
graph_data = torch.load(path, map_location="cpu")

src_nodes = graph_data['src_nodes']
dst_nodes = graph_data['dst_nodes']
edges_scores = graph_data['edges_scores']
edges_in_graph = graph_data['edges_in_graph']

# Build graph with weights
G = nx.DiGraph()
for i, src in enumerate(src_nodes):
    for j, dst in enumerate(dst_nodes):
        if edges_in_graph[i, j]:
            score = float(edges_scores[i, j])
            if abs(score) > 1e-5:  # filter small edges
                G.add_edge(src, dst, weight=score)

# --- Assign layers for visualization ---
def get_layer(node):
    if node == "input":
        return 0
    if node.startswith("a"):  # attention head
        layer = int(node[1:node.index(".")])
        return 1 + layer
    if node.startswith("m"):  # mlp
        return 1 + int(node[1:])  # after attn of same layer
    if node == "logits":
        return 1 + 12 + 1  # after final layer
    return 999  # fallback

layers = defaultdict(list)
for n in G.nodes():
    l = get_layer(n)
    layers[l].append(n)

# Position nodes in layered style
pos = {}
y_spacing = 1
x_spacing = 5
for l, nodes in sorted(layers.items()):
    for i, n in enumerate(nodes):
        pos[n] = (l * x_spacing, -i * y_spacing)

# Draw graph
plt.figure(figsize=(14, 8))
edges = G.edges(data=True)
weights = [abs(d['weight']) for (_, _, d) in edges]

# Normalize edge thickness
max_w = max(weights) if weights else 1
weights = [3 * (w / max_w) for w in weights]

nx.draw_networkx_nodes(G, pos, node_size=200, node_color="skyblue")
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color="gray", arrows=False)
nx.draw_networkx_labels(G, pos, font_size=6)

plt.title("Circuit visualization (layered style)")
plt.axis("off")
plt.show()
