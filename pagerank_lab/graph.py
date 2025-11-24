"""Simple directed graph representation used by pagerank_lab.

This module implements a minimal DiGraph class with the operations
the pagerank implementation expects: add_node, add_edge,
iterating nodes, out_neighbors and in_neighbors, and degree queries.
"""
from collections import defaultdict


class DiGraph:
    def __init__(self):
        self._out = defaultdict(set)
        self._in = defaultdict(set)

    def add_node(self, node):
        # touch node in both maps
        self._out.setdefault(node, set())
        self._in.setdefault(node, set())

    def add_edge(self, src, dst):
        self.add_node(src)
        self.add_node(dst)
        if dst not in self._out[src]:
            self._out[src].add(dst)
            self._in[dst].add(src)

    def nodes(self):
        return list(set(list(self._out.keys()) + list(self._in.keys())))

    def out_neighbors(self, node):
        return list(self._out.get(node, []))

    def in_neighbors(self, node):
        return list(self._in.get(node, []))

    def out_degree(self, node):
        return len(self._out.get(node, []))

    def in_degree(self, node):
        return len(self._in.get(node, []))

    def __repr__(self):
        return f"DiGraph(nodes={len(self.nodes())})"
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
