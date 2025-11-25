# tests/test_pagerank_basic.py
from pagerank_lab.graph import build_graph_from_edges
from pagerank_lab.pagerank import pagerank


def test_two_node_chain():
    # 0 -> 1
    graph = build_graph_from_edges([(0, 1)])
    ranks = pagerank(graph, alpha=0.85)

    # Both nodes should have positive rank and sum to ~1
    assert set(ranks.keys()) == {0, 1}
    assert all(v > 0 for v in ranks.values())
    assert abs(sum(ranks.values()) - 1.0) < 1e-8

    # Node 1 should be more important than node 0
    assert ranks[1] > ranks[0]


def test_triangle_graph():
    # 0 -> 1 -> 2 -> 0 (directed cycle)
    graph = build_graph_from_edges([(0, 1), (1, 2), (2, 0)])
    ranks = pagerank(graph, alpha=0.85)

    # Symmetry: all nodes should have roughly equal rank
    diff01 = abs(ranks[0] - ranks[1])
    diff12 = abs(ranks[1] - ranks[2])
    diff20 = abs(ranks[2] - ranks[0])

    assert diff01 < 1e-6
    assert diff12 < 1e-6
    assert diff20 < 1e-6
