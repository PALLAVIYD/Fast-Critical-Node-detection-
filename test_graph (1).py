"""
tests/test_graph.py — Unit tests for LargeGraph and UnionFind.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cndp.graph import LargeGraph, UnionFind, generate_barabasi_albert


# ─────────────────────────────────────────────────────────────────────────────
#  LargeGraph basics
# ─────────────────────────────────────────────────────────────────────────────

class TestLargeGraph:
    def test_add_node(self):
        g = LargeGraph()
        g.add_node(0)
        g.add_node(1)
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 0

    def test_add_edge(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 2
        assert 1 in g.neighbors(0)
        assert 0 in g.neighbors(1)

    def test_self_loop_ignored(self):
        g = LargeGraph()
        g.add_edge(0, 0)
        assert g.number_of_edges() == 0

    def test_duplicate_edge(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 1)
        assert g.number_of_edges() == 1

    def test_remove_node(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        g.remove_node(1)
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 1
        assert not g.has_node(1)

    def test_remove_edge(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.remove_edge(0, 1)
        assert g.number_of_edges() == 1
        assert 1 not in g.neighbors(0)

    def test_connected_components_single(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        comps = g.connected_components()
        assert len(comps) == 1
        assert set(comps[0]) == {0, 1, 2}

    def test_connected_components_two(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        comps = g.connected_components()
        assert len(comps) == 2

    def test_lcc_size(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        assert g.largest_connected_component_size() == 3

    def test_pairwise_connectivity(self):
        # Triangle: 3 nodes → C(3,2) = 3
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        assert g.pairwise_connectivity() == 3

    def test_pairwise_connectivity_disconnected(self):
        # Two components: {0,1} and {2,3,4}
        # C(2,2) + C(3,2) = 1 + 3 = 4
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        assert g.pairwise_connectivity() == 4

    def test_copy_independence(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g2 = g.copy()
        g2.add_edge(1, 2)
        assert g.number_of_edges() == 1
        assert g2.number_of_edges() == 2

    def test_degree(self):
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        assert g.degree(0) == 3
        assert g.degree(1) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  UnionFind
# ─────────────────────────────────────────────────────────────────────────────

class TestUnionFind:
    def test_basic_union(self):
        uf = UnionFind([0, 1, 2, 3])
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)
        assert uf.find(0) != uf.find(2)

    def test_component_size(self):
        uf = UnionFind([0, 1, 2, 3])
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.component_size(0) == 3
        assert uf.component_size(3) == 1

    def test_pairwise_connectivity(self):
        uf = UnionFind([0, 1, 2, 3, 4])
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        # C(3,2) + C(2,2) = 3 + 1 = 4
        assert uf.pairwise_connectivity() == 4

    def test_num_components(self):
        uf = UnionFind([0, 1, 2, 3])
        assert uf.num_components == 4
        uf.union(0, 1)
        assert uf.num_components == 3
        uf.union(0, 1)  # already same component
        assert uf.num_components == 3


# ─────────────────────────────────────────────────────────────────────────────
#  Graph generators
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerators:
    def test_ba_graph_size(self):
        g = generate_barabasi_albert(100, m=3, seed=42)
        assert g.number_of_nodes() == 100
        assert g.number_of_edges() > 0

    def test_ba_connected(self):
        g = generate_barabasi_albert(50, m=2, seed=42)
        comps = g.connected_components()
        assert len(comps) == 1, "BA graph should be connected"
