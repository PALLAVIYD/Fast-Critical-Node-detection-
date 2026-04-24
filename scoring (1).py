"""
scoring.py — Centrality Aggregation & Node Scoring
Implements the composite score:
    F(v) = α·d̂(v) + β·b̂(v) + γ·k̂(v) + δ·CÎ(v)

All inputs are min-max normalised before aggregation.
Default weights: α=0.2, β=0.2, γ=0.2, δ=0.4
"""

import logging
import math
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Min-max normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise(scores: Dict[int, float]) -> Dict[int, float]:
    """
    φ̂(v) = (φ(v) − min) / (max − min)
    Falls back to zeros if max == min.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span < 1e-12:
        return {v: 0.0 for v in scores}
    return {v: (s - lo) / span for v, s in scores.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Composite score F(v)
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_score(
    degree:      Dict[int, float],
    betweenness: Dict[int, float],
    kcore:       Dict[int, float],
    ci:          Dict[int, float],
    alpha: float = 0.2,
    beta:  float = 0.2,
    gamma: float = 0.2,
    delta: float = 0.4,
) -> Dict[int, float]:
    """
    Returns F(v) for all nodes present in all four dicts.

    Parameters
    ----------
    degree, betweenness, kcore, ci : raw (or already normalised) dicts
    alpha, beta, gamma, delta       : mixing weights (must sum to 1)
    """
    # normalise each component
    d_hat  = normalise(degree)
    b_hat  = normalise(betweenness)
    k_hat  = normalise(kcore)
    ci_hat = normalise(ci)

    nodes = set(d_hat) & set(b_hat) & set(k_hat) & set(ci_hat)

    scores: Dict[int, float] = {}
    for v in nodes:
        scores[v] = (
            alpha * d_hat.get(v, 0.0)
            + beta  * b_hat.get(v, 0.0)
            + gamma * k_hat.get(v, 0.0)
            + delta * ci_hat.get(v, 0.0)
        )

    logger.info(f"Composite score computed for {len(scores)} nodes.")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience wrapper that also returns per-component normalised dicts
# ─────────────────────────────────────────────────────────────────────────────

def compute_entropy_weights(centralities: Dict[str, Dict[int, float]]) -> Dict[str, float]:
    """
    Calculates weights dynamically using the Entropy Weight Method (EWM).
    Metrics with higher structural variance carry more information and thus receive higher weights.
    """
    metrics = ["degree", "betweenness", "kcore", "ci"]
    if "degree" not in centralities or not centralities["degree"]:
        return {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}
        
    n_nodes = len(centralities["degree"])
    if n_nodes <= 1:
        return {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}

    entropies = []
    for m in metrics:
        normed = normalise(centralities[m])
        s_sum = sum(normed.values())
        if s_sum == 0:
            entropies.append(1.0) # max entropy = no structural variance
            continue
            
        e = 0.0
        for val in normed.values():
            p = val / s_sum
            if p > 0:
                e -= p * math.log(p)
                
        e /= math.log(n_nodes)
        entropies.append(e)
        
    # Dispersion (1 - e)
    dispersions = [1.0 - e for e in entropies]
    d_sum = sum(dispersions)
    if d_sum == 0:
        return {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}
        
    w = [d / d_sum for d in dispersions]
    logger.info(f"Entropy weights calculated: deg={w[0]:.3f}, bet={w[1]:.3f}, kcore={w[2]:.3f}, ci={w[3]:.3f}")
    return {"alpha": w[0], "beta": w[1], "gamma": w[2], "delta": w[3]}

def score_nodes(
    centralities: Dict[str, Dict[int, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[int, float]:
    """
    High-level wrapper.
    If weights is None, automatically executes Entropy Weight Method (EWM)
    to dynamically identify best topological combinations.
    """
    if weights is None:
        w = compute_entropy_weights(centralities)
    else:
        w = weights
        
    return compute_composite_score(
        degree      = centralities["degree"],
        betweenness = centralities["betweenness"],
        kcore       = centralities["kcore"],
        ci          = centralities["ci"],
        alpha = w.get("alpha", 0.2),
        beta  = w.get("beta",  0.2),
        gamma = w.get("gamma", 0.2),
        delta = w.get("delta", 0.4),
    )
