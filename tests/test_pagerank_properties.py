# tests/test_pagerank_properties.py
import numpy as np
import networkx as nx

from pagerank_lab.graph import build_graph_from_edges
from pagerank_lab.pagerank import pagerank


def test_pagerank_is_probability_distribution():
    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("C", "A"),
        ("C", "B"),
    ]
    graph = build_graph_from_edges(edges)
    ranks = pagerank(graph, alpha=0.9)

    values = np.array(list(ranks.values()))
    # All ranks non-negative
    assert (values >= 0).all()
    # Sum close to 1
    assert abs(values.sum() - 1.0) < 1e-8


def test_pagerank_matches_networkx():
    # moderately small graph
    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("B", "D"),
        ("C", "A"),
        ("D", "C"),
    ]
    alpha = 0.85
    graph = build_graph_from_edges(edges)

    # our implementation
    my_ranks = pagerank(graph, alpha=alpha)

    # networkx implementation
    G = nx.DiGraph()
    G.add_edges_from(edges)
    nx_ranks = nx.pagerank(G, alpha=alpha)

    # compare each node
    for node in my_ranks:
        assert node in nx_ranks
        assert abs(my_ranks[node] - nx_ranks[node]) < 1e-6
