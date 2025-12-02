# pagerank_lab/__init__.py

"""
pagerank_lab: Tiny lab for PageRank and network algorithms.

This package is intentionally small but engineered like a real library:
- clear API
- tested with pytest
- wired into CI
"""

from .graph import build_graph_from_edges
from .pagerank import pagerank

__all__ = ["build_graph_from_edges", "pagerank"]
