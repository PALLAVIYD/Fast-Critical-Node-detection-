"""
temporal.py — Temporal Graph Modeling
Implements:
  - Graph snapshot management G₁…Gₜ
  - EWMA edge weight update: Wᵗ(u,v) = α·Wᵗ⁻¹(u,v) + (1−α)·W_current(u,v)
  - Temporal node scoring: Sᵛ(t) = α·d̂ + β·b̂ + γ·k̂ + δ·CÎ + ε·Δ̂
  - Temporal smoothness: penalise large changes in selected node sets
"""

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

from cndp.graph import LargeGraph
from cndp.centrality import compute_all_centralities
from cndp.scoring import compute_composite_score, normalise

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Edge Weight Tracker (EWMA)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeWeightTracker:
    """
    Maintains exponentially weighted moving-average edge weights across
    graph snapshots.

    Wᵗ(u,v) = α·Wᵗ⁻¹(u,v) + (1−α)·W_current(u,v)
    """

    def __init__(self, alpha: float = 0.5):
        """
        alpha : smoothing factor in (0, 1).
                High α → more weight on history.
                Low  α → more weight on current snapshot.
        """
        assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"
        self.alpha = alpha
        self._weights: Dict[Tuple[int, int], float] = {}

    def _key(self, u: int, v: int) -> Tuple[int, int]:
        return (min(u, v), max(u, v))

    def update(self, edges: List[Tuple[int, int, float]]) -> None:
        """
        Update weights with new snapshot edges.

        edges : list of (u, v, w) where w is the current-snapshot weight.
        """
        seen = set()
        alpha = self.alpha
        for u, v, w_curr in edges:
            key = self._key(u, v)
            seen.add(key)
            w_prev = self._weights.get(key, w_curr)  # init with current if new
            self._weights[key] = alpha * w_prev + (1 - alpha) * w_curr

        # Decay edges not seen in this snapshot (treat w_curr = 0)
        for key in list(self._weights):
            if key not in seen:
                self._weights[key] = alpha * self._weights[key]
                if self._weights[key] < 1e-6:
                    del self._weights[key]

    def get_weight(self, u: int, v: int) -> float:
        return self._weights.get(self._key(u, v), 0.0)

    def get_all_weights(self) -> Dict[Tuple[int, int], float]:
        return dict(self._weights)


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Score History
# ─────────────────────────────────────────────────────────────────────────────

class TemporalNodeScorer:
    """
    Maintains per-node score history and computes temporal score:

    Sᵛ(t) = α·d̂(v,t) + β·b̂(v,t) + γ·k̂(v,t) + δ·CÎ(v,t) + ε·Δ̂(v,t)

    where Δ̂(v,t) is a velocity term: normalised change in raw degree
    between t−1 and t.  High velocity → node is gaining structural
    importance and should be up-weighted.
    """

    def __init__(
        self,
        alpha: float = 0.20,
        beta:  float = 0.20,
        gamma: float = 0.20,
        delta: float = 0.30,
        eps:   float = 0.10,
        ewma_alpha: float = 0.5,
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.eps   = eps

        self._ewma_alpha = ewma_alpha
        self._prev_degree: Dict[int, float] = {}
        self._history: List[Dict[int, float]] = []   # per-snapshot composite scores

    def update(
        self,
        G: LargeGraph,
        betweenness_k: int = 200,
        seed: int = 42,
    ) -> Dict[int, float]:
        """
        Compute temporal score for all nodes in G at current timestep.

        Returns per-node temporal scores (normalised composite + velocity).
        """
        centralities = compute_all_centralities(G, betweenness_k=betweenness_k, seed=seed)

        # Velocity term Δ(v,t) = |degree(v,t) − degree(v,t-1)|
        curr_degree = centralities["degree"]
        delta_scores: Dict[int, float] = {}
        for v, d in curr_degree.items():
            prev = self._prev_degree.get(v, d)
            delta_scores[v] = abs(d - prev)
        self._prev_degree = dict(curr_degree)

        delta_norm = normalise(delta_scores)

        # Base composite score (without eps·Δ̂)
        base = compute_composite_score(
            degree      = centralities["degree"],
            betweenness = centralities["betweenness"],
            kcore       = centralities["kcore"],
            ci          = centralities["ci"],
            alpha = self.alpha,
            beta  = self.beta,
            gamma = self.gamma,
            delta = self.delta,
        )

        # Add velocity component
        temporal_scores: Dict[int, float] = {}
        for v in base:
            temporal_scores[v] = base[v] + self.eps * delta_norm.get(v, 0.0)

        self._history.append(temporal_scores)
        logger.info(f"Temporal score update: {len(temporal_scores)} nodes, snapshot #{len(self._history)}")
        return temporal_scores

    def history(self) -> List[Dict[int, float]]:
        return self._history


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Smoothness Penalty
# ─────────────────────────────────────────────────────────────────────────────

def temporal_smoothness_penalty(
    selected_prev: Set[int],
    selected_curr: Set[int],
    lambda_: float = 0.5,
) -> float:
    """
    Smoothness penalty = λ · |selected_curr △ selected_prev| / max(|S|)
    Penalise Hamming distance between consecutive node sets.
    """
    if not selected_prev and not selected_curr:
        return 0.0
    sym_diff = selected_prev.symmetric_difference(selected_curr)
    max_size = max(len(selected_prev), len(selected_curr), 1)
    return lambda_ * len(sym_diff) / max_size


def smooth_scores(
    temporal_scores: Dict[int, float],
    selected_prev: Optional[Set[int]],
    lambda_: float = 0.3,
) -> Dict[int, float]:
    """
    Adjust temporal scores by adding a continuity bonus for nodes that
    were previously selected (encourages temporal stability of detection).
    """
    if selected_prev is None:
        return temporal_scores
    bonus = lambda_ / max(len(selected_prev), 1)
    adjusted = dict(temporal_scores)
    for v in selected_prev:
        if v in adjusted:
            adjusted[v] += bonus
    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
#  Snapshot Manager
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGraphManager:
    """
    Orchestrates multi-snapshot CNDP analysis.

    Usage
    -----
    mgr = TemporalGraphManager(alpha=0.5)
    for t, G_t in enumerate(snapshots):
        scores_t = mgr.process_snapshot(G_t, k=20)
        print(f"t={t}: critical nodes = {scores_t['selected']}")
    """

    def __init__(
        self,
        ewma_alpha: float = 0.5,
        score_weights: Optional[Dict[str, float]] = None,
        smoothness_lambda: float = 0.3,
        betweenness_k: int = 200,
    ):
        self.ewma_alpha       = ewma_alpha
        self.score_weights    = score_weights or {}
        self.smoothness_lambda = smoothness_lambda
        self.betweenness_k    = betweenness_k

        self.scorer = TemporalNodeScorer(
            alpha = self.score_weights.get("alpha", 0.2),
            beta  = self.score_weights.get("beta",  0.2),
            gamma = self.score_weights.get("gamma", 0.2),
            delta = self.score_weights.get("delta", 0.3),
            eps   = self.score_weights.get("eps",   0.1),
            ewma_alpha = ewma_alpha,
        )
        self.edge_tracker = EdgeWeightTracker(alpha=ewma_alpha)

        self._selected_prev: Optional[Set[int]] = None
        self._snapshots: List[Tuple[int, List[int]]] = []    # (t, selected_nodes)

    def process_snapshot(
        self,
        G: LargeGraph,
        k: int,
        seed: int = 42,
    ) -> Dict:
        """
        Process one snapshot.  Returns dict with keys:
          'scores', 'selected', 'smoothness_penalty'
        """
        # 1. Update EWMA edge weights
        edges_with_w = [(u, v, 1.0) for u, v in G.edges()]
        self.edge_tracker.update(edges_with_w)

        # 2. Compute temporal scores
        scores = self.scorer.update(G, betweenness_k=self.betweenness_k, seed=seed)

        # 3. Apply smoothness adjustment
        adjusted = smooth_scores(scores, self._selected_prev, self.smoothness_lambda)

        # 4. Select top-k
        k_actual  = min(k, G.number_of_nodes())
        selected  = set(sorted(adjusted, key=lambda v: -adjusted[v])[:k_actual])

        # 5. Compute smoothness penalty vs previous snapshot
        penalty = temporal_smoothness_penalty(
            self._selected_prev or set(),
            selected,
            lambda_ = self.smoothness_lambda,
        )

        self._selected_prev = selected
        t = len(self._snapshots)
        self._snapshots.append((t, list(selected)))

        logger.info(f"Snapshot t={t}: selected {len(selected)} nodes, "
                    f"smoothness_penalty={penalty:.4f}")

        return {
            "scores"             : adjusted,
            "selected"           : list(selected),
            "smoothness_penalty" : penalty,
        }

    def get_snapshot_history(self) -> List[Tuple[int, List[int]]]:
        return self._snapshots
