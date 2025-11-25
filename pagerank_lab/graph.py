# pagerank_lab/graph.py
from typing import Dict, List, Hashable, Iterable, Tuple

Node = Hashable

def build_graph_from_edges(edges: Iterable[Tuple[Node, Node]]) -> Dict[Node, List[Node]]:
    """
    Build a simple adjacency list representation of a directed graph.

    Parameters
    ----------
    edges : iterable of (u, v)
        Directed edges u -> v.

    Returns
    -------
    graph : dict
        Mapping node -> list of outgoing neighbors.
    """
    graph: Dict[Node, List[Node]] = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []  # ensure isolated nodes exist
        graph[u].append(v)
    return graph
