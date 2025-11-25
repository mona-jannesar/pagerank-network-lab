# examples/example_toy_graph.py
import sys, os
print(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pagerank_lab.graph import build_graph_from_edges
from pagerank_lab.pagerank import pagerank


if __name__ == "__main__":
    edges = [
        ("Home", "About"),
        ("Home", "Products"),
        ("Products", "Cart"),
        ("Cart", "Checkout"),
        ("Checkout", "Home"),
        ("About", "Home"),
    ]

    graph = build_graph_from_edges(edges)
    ranks = pagerank(graph, alpha=0.85)

    print("PageRank scores:")
    for node, score in sorted(ranks.items(), key=lambda x: -x[1]):
        print(f"{node:10s} -> {score:.4f}")
