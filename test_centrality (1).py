"""
tests/test_centrality.py — Unit tests for centrality computations.
Validates against NetworkX on small graphs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import networkx as nx
from cndp.graph import LargeGraph
from cndp.centrality import (
    degree_centrality,
    approximate_betweenness,
    kcore_score,
    connectivity_impact,
    _find_articulation_points,
)


def _make_path_graph(n: int) -> LargeGraph:
    """0 - 1 - 2 - ... - (n-1)"""
    g = LargeGraph()
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def _make_star_graph(n: int) -> LargeGraph:
    """Hub at 0, spokes 1..n-1"""
    g = LargeGraph()
    for i in range(1, n):
        g.add_edge(0, i)
    return g


# ─────────────────────────────────────────────────────────────────────────────
#  Degree centrality
# ─────────────────────────────────────────────────────────────────────────────

class TestDegreeCentrality:
    def test_star_hub_has_highest(self):
        g = _make_star_graph(6)
        dc = degree_centrality(g)
        hub_score = dc[0]
        for i in range(1, 6):
            assert hub_score > dc[i], f"Hub should have highest degree centrality"

    def test_matches_networkx(self):
        g = _make_path_graph(5)
        dc = degree_centrality(g)
        G_nx = g.to_networkx()
        nx_dc = nx.degree_centrality(G_nx)
        for v in dc:
            assert abs(dc[v] - nx_dc[v]) < 1e-6, f"Mismatch at node {v}"


# ─────────────────────────────────────────────────────────────────────────────
#  K-Core
# ─────────────────────────────────────────────────────────────────────────────

class TestKCore:
    def test_path_all_1core(self):
        """Path graph: all internal nodes have coreness 1 (except endpoints can be 1 too)."""
        g = _make_path_graph(5)
        kc = kcore_score(g)
        for v in g.nodes():
            assert kc[v] <= 1

    def test_triangle_is_2core(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        kc = kcore_score(g)
        for v in [0, 1, 2]:
            assert kc[v] == 2

    def test_matches_networkx(self):
        g = _make_star_graph(5)
        kc = kcore_score(g)
        G_nx = g.to_networkx()
        nx_kc = nx.core_number(G_nx)
        for v in kc:
            assert kc[v] == nx_kc[v], f"Mismatch at node {v}: {kc[v]} vs {nx_kc[v]}"


# ─────────────────────────────────────────────────────────────────────────────
#  Articulation points
# ─────────────────────────────────────────────────────────────────────────────

class TestArticulationPoints:
    def test_path_graph(self):
        """In a path 0-1-2-3-4, nodes 1,2,3 are articulation points."""
        g = _make_path_graph(5)
        aps = _find_articulation_points(g)
        assert aps == {1, 2, 3}

    def test_triangle(self):
        """No articulation points in a complete triangle."""
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        aps = _find_articulation_points(g)
        assert len(aps) == 0

    def test_bridge_node(self):
        """
        Two triangles connected by a bridge node:
        0-1-2-0 and 3-4-5-3, connected by edge 2-3.
        Nodes 2 and 3 are articulation points.
        """
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        g.add_edge(3, 4)
        g.add_edge(4, 5)
        g.add_edge(3, 5)
        g.add_edge(2, 3)
        aps = _find_articulation_points(g)
        assert 2 in aps
        assert 3 in aps


# ─────────────────────────────────────────────────────────────────────────────
#  Connectivity Impact
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectivityImpact:
    def test_star_hub_has_highest_ci(self):
        g = _make_star_graph(6)
        ci = connectivity_impact(g)
        hub_ci = ci[0]
        for i in range(1, 6):
            assert hub_ci > ci[i], f"Hub should have highest CI"

    def test_triangle_zero_ci(self):
        """No node in a triangle is an articulation point → CI should be small."""
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        ci = connectivity_impact(g)
        # All non-AP, so CI is just degree/max_degree (small)
        for v in [0, 1, 2]:
            assert ci[v] < 1.0
