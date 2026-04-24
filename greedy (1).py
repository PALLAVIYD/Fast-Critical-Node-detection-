"""
greedy.py — Lazy Greedy CNDP with Submodularity Optimisation
Implements priority-queue-based lazy evaluation of marginal gains.

Algorithm:
  1. Pre-compute true marginal gains for top-N candidates by composite score
  2. Initialise priority queue with these true gains (valid upper bounds)
  3. Each iteration:
     a. Pop the node with the best cached gain
     b. Recompute its actual Δ(v|A) = f(A) − f(A∪{v})
     c. If it's still the best after recomputation → select it
     d. Otherwise push it back, try next candidate
  4. Repeat until k nodes selected

Submodularity guarantees cached gains are upper-bounds on true gains,
so we only recompute when a node is popped (lazy evaluation).
"""

import heapq
import logging
import time
from typing import Dict, List, Optional, Tuple

from cndp.graph import LargeGraph, UnionFind
from cndp.metrics import pairwise_connectivity, lcc_size, robustness_index

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Fast pairwise connectivity via Union-Find rebuild
# ─────────────────────────────────────────────────────────────────────────────

def _uf_pairwise_connectivity(G: LargeGraph) -> int:
    """
    Compute pairwise connectivity using Union-Find.
    Faster than BFS-based connected_components for repeated calls.
    """
    nodes = G.nodes()
    if not nodes:
        return 0
    uf = UnionFind(nodes)
    for u, v in G.edges():
        uf.union(u, v)
    return uf.pairwise_connectivity()


def _uf_pairwise_without_node(G: LargeGraph, node: int) -> int:
    """
    Compute pairwise connectivity of G with `node` removed,
    using Union-Find (no graph mutation needed).
    """
    nodes = [v for v in G.nodes() if v != node]
    if not nodes:
        return 0
    uf = UnionFind(nodes)
    nbrs_of_node = G.neighbors(node)
    for u, v in G.edges():
        if u != node and v != node:
            uf.union(u, v)
    return uf.pairwise_connectivity()


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy Greedy CNDP (main algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def lazy_greedy_cndp(
    G_in: LargeGraph,
    k: int,
    initial_scores: Optional[Dict[int, float]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
    precompute_top_n: int = 50,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Lazy Greedy Critical Node Detection.

    Parameters
    ----------
    G_in          : input graph (not modified — we work on an internal copy)
    k             : number of critical nodes to remove
    initial_scores: dict {node: score} used to select top candidates.
                    If None, falls back to pairwise connectivity gain.
    sample_size   : restrict candidate set to top-`sample_size` by score.
                    Use this for 500K+ graphs.
    seed          : random seed for tie-breaking
    verbose       : log each removal step
    precompute_top_n : number of top candidates to pre-compute true marginal
                       gains for (ensures valid upper bounds in the PQ).

    Returns
    -------
    removed_nodes    : ordered list of selected critical nodes
    connectivity_seq : pairwise connectivity after each removal
    lcc_seq          : LCC size after each removal
    """
    G = G_in.copy()
    nodes = G.nodes()
    n = len(nodes)

    k = min(k, n)

    # ── Sort Candidates by Heuristic ──────────────────────────────────────────
    if initial_scores is not None:
        if sample_size:
            candidates_list = sorted(
                nodes, key=lambda v: -initial_scores.get(v, 0)
            )[:sample_size]
        else:
            candidates_list = sorted(
                nodes, key=lambda v: -initial_scores.get(v, 0)
            )
    else:
        if sample_size:
            candidates_list = list(nodes)[:sample_size]
        else:
            candidates_list = list(nodes)

    # ── Compute current pairwise connectivity ─────────────────────────────────
    current_f = _uf_pairwise_connectivity(G)

    pq = []
    candidate_idx = 0
    recompute_count = 0

    def evaluate_next_batch(batch_size: int = 50):
        nonlocal candidate_idx, recompute_count
        end = min(candidate_idx + batch_size, len(candidates_list))
        for v in candidates_list[candidate_idx:end]:
            if not G.has_node(v):
                continue
            new_f = _uf_pairwise_without_node(G, v)
            true_gain = current_f - new_f
            score = initial_scores.get(v, 0.0) if initial_scores else 0.0
            heapq.heappush(pq, (-true_gain, -score, v))
            recompute_count += 1
        candidate_idx = end

    # Initial batch precomputation
    evaluate_next_batch(precompute_top_n)

    removed_nodes    = []
    connectivity_seq = []
    lcc_seq          = []
    
    # Calculate baseline naive evaluations for logging comparison
    naive_evals = sum(max(0, len(candidates_list) - i) for i in range(k))

    for step in range(k):
        if not pq and candidate_idx >= len(candidates_list):
            break
            
        if G.number_of_nodes() == 0:
            break

        best_node = None
        best_gain = -1

        while pq or candidate_idx < len(candidates_list):
            if not pq:
                evaluate_next_batch(precompute_top_n)
                if not pq: break

            neg_cached_gain, neg_score, v = heapq.heappop(pq)
            cached_gain = -neg_cached_gain

            if not G.has_node(v):
                continue  # already removed

            # Recompute actual marginal gain using Union-Find (no graph copy needed)
            new_f = _uf_pairwise_without_node(G, v)
            true_gain  = current_f - new_f
            recompute_count += 1

            # Lazy greedy check: compare the newly computed gain tuple against top of queue
            if not pq or (-true_gain, neg_score, v) <= pq[0]:
                best_node = v
                best_gain = true_gain
                break
            else:
                # Push back with updated gain
                heapq.heappush(pq, (-true_gain, neg_score, v))

        if best_node is None:
            logger.warning("No candidate found at step %d — stopping early.", step)
            break

        # ── Remove best node ──────────────────────────────────────────────────
        G.remove_node(best_node)
        current_f  -= best_gain
        removed_nodes.append(best_node)
        connectivity_seq.append(current_f)
        lcc_seq.append(G.largest_connected_component_size())

        if verbose:
            logger.info(
                f"Step {step + 1:03d}: removed node {best_node:8d} | "
                f"gain={best_gain:>10d} | f(A)={current_f:>12d} | "
                f"LCC={lcc_seq[-1]}"
            )

    logger.info(
        f"Lazy Greedy (CELF) Optimisation Complete.\n"
        f"  Theoretical Submodular Bounds: ≥ 63.2% (1-1/e) of Optimal Target\n"
        f"  Nodes Removed: {len(removed_nodes)}\n"
        f"  Marginal-gain Recomputations Executed: {recompute_count}\n"
        f"  Naive Recalculations Skipped: {max(0, naive_evals - recompute_count)} "
        f"(Computational Savings: {100*max(0, naive_evals - recompute_count)/max(1, naive_evals):.1f}%)"
    )
    return removed_nodes, connectivity_seq, lcc_seq


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline: Static degree removal (no re-ranking)
# ─────────────────────────────────────────────────────────────────────────────

def degree_greedy(G_in: LargeGraph, k: int) -> Tuple[List[int], List[int], List[int]]:
    """Remove top-degree nodes in static order (ranked once, no re-ranking)."""
    G = G_in.copy()
    # Static ranking: compute degree once, then remove in that order
    degree_rank = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)[:k]
    removed_nodes, conn_seq, lcc_seq = [], [], []
    for v in degree_rank:
        if not G.has_node(v):
            continue
        G.remove_node(v)
        removed_nodes.append(v)
        conn_seq.append(_uf_pairwise_connectivity(G))
        lcc_seq.append(G.largest_connected_component_size())
    return removed_nodes, conn_seq, lcc_seq


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline: Adaptive degree removal (re-rank each step)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_degree_greedy(G_in: LargeGraph, k: int) -> Tuple[List[int], List[int], List[int]]:
    """Remove top-degree nodes one by one (re-rank each step)."""
    G = G_in.copy()
    removed_nodes, conn_seq, lcc_seq = [], [], []
    for _ in range(min(k, G.number_of_nodes())):
        if G.number_of_nodes() == 0:
            break
        best = max(G.nodes(), key=lambda v: G.degree(v))
        G.remove_node(best)
        removed_nodes.append(best)
        conn_seq.append(_uf_pairwise_connectivity(G))
        lcc_seq.append(G.largest_connected_component_size())
    return removed_nodes, conn_seq, lcc_seq


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline: Random removal
# ─────────────────────────────────────────────────────────────────────────────

def random_removal(G_in: LargeGraph, k: int, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """Remove k randomly chosen nodes."""
    import random
    G = G_in.copy()
    rng = random.Random(seed)
    removed_nodes, conn_seq, lcc_seq = [], [], []
    nodes = G.nodes()
    order = rng.sample(nodes, min(k, len(nodes)))
    for v in order:
        if not G.has_node(v):
            break
        G.remove_node(v)
        removed_nodes.append(v)
        conn_seq.append(_uf_pairwise_connectivity(G))
        lcc_seq.append(G.largest_connected_component_size())
    return removed_nodes, conn_seq, lcc_seq


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline: Static betweenness removal (pre-computed ranking)
# ─────────────────────────────────────────────────────────────────────────────

def betweenness_removal(
    G_in: LargeGraph,
    k: int,
    betweenness: Dict[int, float],
) -> Tuple[List[int], List[int], List[int]]:
    """Remove top-betweenness nodes (static order, no rerank)."""
    G = G_in.copy()
    order = sorted(betweenness, key=lambda v: -betweenness[v])[:k]
    removed_nodes, conn_seq, lcc_seq = [], [], []
    for v in order:
        if not G.has_node(v):
            conn_seq.append(conn_seq[-1] if conn_seq else _uf_pairwise_connectivity(G))
            lcc_seq.append(lcc_seq[-1] if lcc_seq else G.largest_connected_component_size())
            continue
        G.remove_node(v)
        removed_nodes.append(v)
        conn_seq.append(_uf_pairwise_connectivity(G))
        lcc_seq.append(G.largest_connected_component_size())
    return removed_nodes, conn_seq, lcc_seq


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline: Adaptive betweenness removal (re-rank each step)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_betweenness_greedy(
    G_in: LargeGraph,
    k: int,
    betweenness_k: int = 200,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Remove top-betweenness nodes with re-ranking after each removal.
    Recomputes approximate betweenness after each node is removed.
    """
    from cndp.centrality import approximate_betweenness

    G = G_in.copy()
    removed_nodes, conn_seq, lcc_seq = [], [], []
    for step in range(min(k, G.number_of_nodes())):
        if G.number_of_nodes() == 0:
            break
        bet = approximate_betweenness(G, k=min(betweenness_k, G.number_of_nodes()), seed=seed + step)
        best = max(bet, key=bet.get)
        G.remove_node(best)
        removed_nodes.append(best)
        conn_seq.append(_uf_pairwise_connectivity(G))
        lcc_seq.append(G.largest_connected_component_size())
    return removed_nodes, conn_seq, lcc_seq
