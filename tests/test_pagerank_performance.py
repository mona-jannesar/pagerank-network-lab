import time

import networkx as nx
import pytest

from pagerank_lab import build_graph_from_edges, pagerank


@pytest.mark.performance
def test_pagerank_large_random_graph_runs_fast():
    """
    Performance test on a random directed graph.

    Goal:
    - Check that PageRank runs in reasonable time on a graph
      with ~1000 nodes and O(10^4) edges.
    """

    # Graph size / density
    n = 1000          # number of nodes
    p = 0.005         # edge probability

    # Build a random directed graph with NetworkX
    G_nx = nx.gnp_random_graph(n, p, directed=True)
    edges = list(G_nx.edges())

    # Use your own helper to build adjacency list
    graph = build_graph_from_edges(edges)

    # Time the PageRank computation
    start = time.perf_counter()
    ranks = pagerank(graph, alpha=0.85, tol=1e-6, max_iter=100)
    duration = time.perf_counter() - start

    # Sanity checks
    assert len(ranks) == n
    assert abs(sum(ranks.values()) - 1.0) < 1e-6

    # Performance check: should be comfortably under 1 second
    # (adjust this if your machine is slower/faster)
    assert duration < 1.0, f"PageRank took too long: {duration:.3f} seconds"
