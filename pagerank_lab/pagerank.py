"""PageRank algorithm implementation.

Implements the power-iteration PageRank algorithm returning a
dictionary mapping nodes to their PageRank score (normalized to sum=1).
"""
from typing import Dict


def pagerank(graph, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict:
    nodes = graph.nodes()
    if not nodes:
        return {}
    N = len(nodes)
    # initialize
    rank = {n: 1.0 / N for n in nodes}

    for i in range(max_iter):
        new_rank = {n: (1.0 - damping) / N for n in nodes}
        # distribute ranks
        for n in nodes:
            out_deg = graph.out_degree(n)
            if out_deg == 0:
                # dangling node: distribute evenly
                share = damping * rank[n] / N
                for m in nodes:
                    new_rank[m] += share
            else:
                share = damping * rank[n] / out_deg
                for m in graph.out_neighbors(n):
                    new_rank[m] += share

        # check convergence (L1)
        diff = sum(abs(new_rank[n] - rank[n]) for n in nodes)
        rank = new_rank
        if diff < tol:
            break

    # normalize (numerical safety)
    s = sum(rank.values())
    if s == 0:
        return {n: 0.0 for n in nodes}
    return {n: v / s for n, v in rank.items()}
# pagerank_lab/pagerank.py
from typing import Dict, Hashable, List
import numpy as np

Node = Hashable


def pagerank(
    graph: Dict[Node, List[Node]],
    alpha: float = 0.85,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Dict[Node, float]:
    """
    Compute PageRank for a directed graph using power iteration.

    Parameters
    ----------
    graph : dict
        Mapping node -> list of outgoing neighbors.
    alpha : float
        Damping factor (probability of following links).
    tol : float
        Convergence tolerance on the L1 norm.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    ranks : dict
        Mapping node -> PageRank score (sum ≈ 1).
    """
    if not graph:
        return {}

    nodes: List[Node] = list(graph.keys())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    # Build out-degree and handle dangling nodes
    out_degree = np.zeros(n, dtype=float)
    for u, neighbors in graph.items():
        out_degree[index[u]] = len(neighbors)

    # Initialize rank vector π^0 as uniform
    pr = np.ones(n, dtype=float) / n
    teleport = np.ones(n, dtype=float) / n

    # Precompute adjacency in a “who points to me” way for efficiency
    incoming = [[] for _ in range(n)]
    for u, neighbors in graph.items():
        u_idx = index[u]
        for v in neighbors:
            v_idx = index[v]
            incoming[v_idx].append(u_idx)

    for _ in range(max_iter):
        new_pr = np.zeros_like(pr)

        # dangling mass: nodes with zero out-degree
        dangling_mask = (out_degree == 0)
        dangling_mass = pr[dangling_mask].sum()

        for i in range(n):
            # Sum contributions from incoming neighbors
            s = 0.0
            for j in incoming[i]:
                if out_degree[j] > 0:
                    s += pr[j] / out_degree[j]

            # Add contribution from dangling nodes and teleportation
            new_pr[i] = alpha * (s + dangling_mass / n) + (1.0 - alpha) * teleport[i]

        # Check convergence (L1 norm)
        diff = np.abs(new_pr - pr).sum()
        pr = new_pr
        if diff < tol:
            break

    # Normalize just in case of numerical drift
    pr /= pr.sum()

    return {node: float(pr[index[node]]) for node in nodes}
