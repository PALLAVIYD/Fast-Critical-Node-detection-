"""
tests/test_greedy.py — Unit tests for greedy CNDP algorithms.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cndp.graph import LargeGraph, generate_barabasi_albert
from cndp.centrality import compute_all_centralities
from cndp.scoring import score_nodes
from cndp.greedy import (
    lazy_greedy_cndp,
    degree_greedy,
    random_removal,
    betweenness_removal,
    adaptive_betweenness_greedy,
    _uf_pairwise_connectivity,
)


def _make_barbell() -> LargeGraph:
    """
    Barbell graph: two cliques of 5 connected by a single bridge node.
    Nodes 0-4 form clique 1, nodes 6-10 form clique 2, node 5 bridges them.
    """
    g = LargeGraph()
    # Clique 1: 0-4
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(i, j)
    # Clique 2: 6-10
    for i in range(6, 11):
        for j in range(i + 1, 11):
            g.add_edge(i, j)
    # Bridge: 4-5-6
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    return g


# ─────────────────────────────────────────────────────────────────────────────
#  Basic properties
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyBasics:
    def test_connectivity_monotonically_decreases(self):
        """Pairwise connectivity should never increase as nodes are removed."""
        g = generate_barabasi_albert(200, m=3, seed=42)
        centralities = compute_all_centralities(g, betweenness_k=50)
        scores = score_nodes(centralities)
        _, conn_seq, _ = lazy_greedy_cndp(
            g, k=10, initial_scores=scores, verbose=False, precompute_top_n=20
        )
        for i in range(1, len(conn_seq)):
            assert conn_seq[i] <= conn_seq[i - 1], \
                f"Connectivity increased at step {i}: {conn_seq[i - 1]} -> {conn_seq[i]}"

    def test_lcc_monotonically_decreases_or_equal(self):
        """LCC size should generally decrease (may stay same in some steps)."""
        g = generate_barabasi_albert(200, m=3, seed=42)
        initial_lcc = g.largest_connected_component_size()
        centralities = compute_all_centralities(g, betweenness_k=50)
        scores = score_nodes(centralities)
        _, _, lcc_seq = lazy_greedy_cndp(
            g, k=10, initial_scores=scores, verbose=False, precompute_top_n=20
        )
        assert lcc_seq[-1] <= initial_lcc

    def test_removes_correct_count(self):
        g = generate_barabasi_albert(100, m=2, seed=42)
        centralities = compute_all_centralities(g, betweenness_k=50)
        scores = score_nodes(centralities)
        removed, _, _ = lazy_greedy_cndp(
            g, k=5, initial_scores=scores, verbose=False
        )
        assert len(removed) == 5

    def test_k_clamped_to_n(self):
        """Requesting more removals than nodes should be clamped."""
        g = LargeGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        removed, _, _ = degree_greedy(g, k=100)
        assert len(removed) <= 3


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy Greedy beats random
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyVsRandom:
    def test_greedy_beats_random_on_ba(self):
        """Lazy greedy should reduce connectivity more than random removal."""
        g = generate_barabasi_albert(500, m=3, seed=42)
        initial_f = _uf_pairwise_connectivity(g)

        centralities = compute_all_centralities(g, betweenness_k=100)
        scores = score_nodes(centralities)
        _, conn_greedy, _ = lazy_greedy_cndp(
            g, k=10, initial_scores=scores, verbose=False, precompute_top_n=30
        )
        _, conn_random, _ = random_removal(g, k=10, seed=42)

        greedy_reduction = initial_f - conn_greedy[-1]
        random_reduction = initial_f - conn_random[-1]
        assert greedy_reduction >= random_reduction, \
            f"Greedy ({greedy_reduction}) should beat random ({random_reduction})"


# ─────────────────────────────────────────────────────────────────────────────
#  Bridge node detection
# ─────────────────────────────────────────────────────────────────────────────

class TestBridgeDetection:
    def test_barbell_bridge_removed_first(self):
        """In a barbell graph, the bridge node (5) should be removed first."""
        g = _make_barbell()
        centralities = compute_all_centralities(g, betweenness_k=11)
        scores = score_nodes(centralities)
        removed, _, _ = lazy_greedy_cndp(
            g, k=3, initial_scores=scores, verbose=False, precompute_top_n=11
        )
        assert removed[0] == 5, \
            f"Bridge node 5 should be removed first, got {removed[0]}"


# ─────────────────────────────────────────────────────────────────────────────
#  Union-Find consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestUFConsistency:
    def test_uf_matches_bfs(self):
        """UnionFind-based and BFS-based pairwise connectivity should match."""
        g = generate_barabasi_albert(100, m=2, seed=42)
        uf_val = _uf_pairwise_connectivity(g)
        bfs_val = g.pairwise_connectivity()
        assert uf_val == bfs_val
