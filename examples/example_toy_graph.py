from pagerank_lab.graph import DiGraph
from pagerank_lab.pagerank import pagerank


def main():
    g = DiGraph()
    g.add_edge('A', 'B')
    g.add_edge('A', 'C')
    g.add_edge('B', 'C')
    g.add_edge('C', 'A')

    pr = pagerank(g)
    for n, score in sorted(pr.items()):
        print(f"{n}: {score:.6f}")


if __name__ == '__main__':
    main()
# examples/example_toy_graph.py
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
