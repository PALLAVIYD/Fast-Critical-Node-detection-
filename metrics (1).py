"""
metrics.py — Evaluation Metrics for CNDP
Implements:
  - Pairwise connectivity f(A)
  - Largest Connected Component (LCC) size
  - Robustness Index R  (area under the LCC curve)
  - Runtime tracking
"""

import time
import logging
from typing import Dict, List

from cndp.graph import LargeGraph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Core metrics (thin wrappers around LargeGraph methods)
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_connectivity(G: LargeGraph) -> int:
    """
    f(A) = Σ C(|Si|, 2) over connected components Si.
    A submodular, non-increasing function of removed nodes.
    """
    return G.pairwise_connectivity()


def lcc_size(G: LargeGraph) -> int:
    """Size of the largest connected component."""
    return G.largest_connected_component_size()


def robustness_index(lcc_seq: List[int], n: int) -> float:
    """
    R = (1/n) · Σ_{i=1}^{n} lcc_size_after_i_removals / n
    Measures the area under the normalised LCC curve.
    Lower R → more robust disruption.
    """
    if n <= 0:
        return 0.0
    return sum(lcc_seq) / (n * n)


def normalised_connectivity_reduction(
    initial_f: int,
    conn_seq: List[int],
) -> List[float]:
    """
    Returns fraction of pairwise connectivity remaining after each removal.
    """
    if initial_f == 0:
        return [0.0] * len(conn_seq)
    return [v / initial_f for v in conn_seq]


# ─────────────────────────────────────────────────────────────────────────────
#  Full metric evaluation over a single run
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_run(
    G_initial: LargeGraph,
    removed_nodes: List[int],
    conn_seq: List[int],
    lcc_seq: List[int],
    runtime_s: float = 0.0,
) -> Dict:
    """
    Collects all metrics for one experiment run.

    Returns
    -------
    dict with keys:
      initial_f, final_f, f_reduction_pct,
      initial_lcc, final_lcc, lcc_reduction_pct,
      robustness_index, runtime_s, removed_nodes
    """
    n          = G_initial.number_of_nodes()
    initial_f  = G_initial.pairwise_connectivity()
    initial_lcc = G_initial.largest_connected_component_size()

    final_f    = conn_seq[-1] if conn_seq else initial_f
    final_lcc  = lcc_seq[-1]  if lcc_seq  else initial_lcc

    f_reduction_pct   = 100 * (initial_f - final_f) / max(initial_f, 1)
    lcc_reduction_pct = 100 * (initial_lcc - final_lcc) / max(initial_lcc, 1)
    R = robustness_index(lcc_seq, n)

    return {
        "initial_f"          : initial_f,
        "final_f"            : final_f,
        "f_reduction_pct"    : f_reduction_pct,
        "initial_lcc"        : initial_lcc,
        "final_lcc"          : final_lcc,
        "lcc_reduction_pct"  : lcc_reduction_pct,
        "robustness_index"   : R,
        "runtime_s"          : runtime_s,
        "removed_nodes"      : removed_nodes,
        "conn_seq"           : conn_seq,
        "lcc_seq"            : lcc_seq,
        "n_removed"          : len(removed_nodes),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-run statistics (mean ± std)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_runs(run_results: List[Dict]) -> Dict:
    """
    Given a list of evaluate_run dicts, compute mean and std for
    each scalar metric across runs (for ≥5 experimental repeats).
    """
    import numpy as np

    scalar_keys = [
        "f_reduction_pct", "lcc_reduction_pct",
        "robustness_index", "runtime_s",
    ]
    agg = {}
    for key in scalar_keys:
        vals = [r[key] for r in run_results if key in r]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"]  = float(np.std(vals))

    # Also aggregate sequences (for plotting)
    if run_results:
        max_len = max(len(r.get("conn_seq", [])) for r in run_results)
        conn_arr = np.zeros((len(run_results), max_len))
        lcc_arr  = np.zeros((len(run_results), max_len))
        for i, r in enumerate(run_results):
            cs = r.get("conn_seq", [])
            ls = r.get("lcc_seq",  [])
            conn_arr[i, :len(cs)] = cs
            lcc_arr[i,  :len(ls)] = ls
        agg["conn_mean"] = conn_arr.mean(axis=0).tolist()
        agg["conn_std"]  = conn_arr.std(axis=0).tolist()
        agg["lcc_mean"]  = lcc_arr.mean(axis=0).tolist()
        agg["lcc_std"]   = lcc_arr.std(axis=0).tolist()

    return agg
