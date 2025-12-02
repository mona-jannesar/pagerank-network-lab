# PageRank Network Lab

A small Python project that implements the PageRank algorithm from scratch,
validates it with PyTest (including comparison to NetworkX), and runs tests
automatically via GitHub Actions CI.

## Features

- Pure Python PageRank implementation using power iteration
- Handles dangling nodes and teleportation (damping factor)
- Deterministic tests on toy graphs
- Property-based checks (non-negativity, normalization)
- Cross-validation with `networkx.pagerank`
- GitHub Actions CI pipeline

## Quickstart

```bash
pip install -r requirements.txt
pytest -v
python examples/example_toy_graph.py
