# pagerank_lab/__init__.py

"""
pagerank_lab package

This package contains:
- Graph construction utilities
- A pure Python PageRank implementation
"""

from .graph import build_graph_from_edges
from .pagerank import pagerank

__all__ = ["build_graph_from_edges", "pagerank"]
