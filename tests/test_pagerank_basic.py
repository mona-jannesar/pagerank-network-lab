# tests/test_pagerank_basic.py
from shapely import node
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


def test_star_graph_center_dominates():
    """
    Star graph:
          1
          |
      2---0---3
          |
          4
          
          """  
    edges = [(1, 0), (2, 0), (3, 0), (4, 0)]
    graph = build_graph_from_edges(edges)
    ranks = pagerank(graph, alpha=0.85)

    # Rank vector should sum to ~1
    assert abs(sum(ranks.values()) - 1.0) < 1e-6

    # Center node 0 should have the highest rank
    center_rank = ranks[0]
    leaf_ranks = [ranks[node] for node in [1, 2, 3, 4]]

    assert all(center_rank > leaf_rank for leaf_rank in leaf_ranks)

    #leaves should be symmetric
    diffs = [
        abs(ranks[1] - ranks[2]),
        abs(ranks[2] - ranks[3]),
        abs(ranks[3] - ranks[4]),
    ]
    
    assert all(d < 1e-6 for d in diffs)
