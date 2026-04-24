"""
experiments/run_experiments.py
Full comparative experiment: Greedy CNDP vs baselines.
Runs ≥5 times per method, reports mean ± std, saves plots.

Usage:
    python -m experiments.run_experiments --dataset facebook --k 20 --runs 5
    python -m experiments.run_experiments --dataset ba --n 10000 --k 30 --runs 5
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ensure project root on path ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cndp.graph import LargeGraph
from cndp.centrality import compute_all_centralities
from cndp.scoring import score_nodes
from cndp.greedy import (
    lazy_greedy_cndp,
    degree_greedy,
    adaptive_degree_greedy,
    random_removal,
    betweenness_removal,
    adaptive_betweenness_greedy,
)
from cndp.metrics import evaluate_run, aggregate_runs
from cndp.utils import setup_logging, MemoryTracker, print_results_table
from datasets.loader import load_dataset

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Single experiment run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(G: LargeGraph, method: str, k: int, centralities: Dict, seed: int, betweenness_k: int = 200) -> Dict:
    mt = MemoryTracker()
    mt.start()
    t0 = time.perf_counter()

    n = G.number_of_nodes()
    # Cap candidate set for greedy to avoid O(n) BFS calls per step
    sample_size = min(n, max(500, n // 10))

    if method == "lazy_greedy":
        scores = score_nodes(centralities)
        removed, conn_seq, lcc_seq = lazy_greedy_cndp(
            G, k=k, initial_scores=scores,
            sample_size=sample_size, seed=seed, verbose=False,
            precompute_top_n=50,
        )
    elif method == "lazy_greedy_no_bet":
        scores = score_nodes(centralities, weights={"alpha": 0.33, "beta": 0.0, "gamma": 0.33, "delta": 0.34})
        removed, conn_seq, lcc_seq = lazy_greedy_cndp(
            G, k=k, initial_scores=scores,
            sample_size=sample_size, seed=seed, verbose=False,
            precompute_top_n=50,
        )
    elif method == "lazy_greedy_no_ci":
        scores = score_nodes(centralities, weights={"alpha": 0.33, "beta": 0.34, "gamma": 0.33, "delta": 0.0})
        removed, conn_seq, lcc_seq = lazy_greedy_cndp(
            G, k=k, initial_scores=scores,
            sample_size=sample_size, seed=seed, verbose=False,
            precompute_top_n=50,
        )
    elif method == "degree":
        removed, conn_seq, lcc_seq = degree_greedy(G, k=k)
    elif method == "adaptive_degree":
        removed, conn_seq, lcc_seq = adaptive_degree_greedy(G, k=k)
    elif method == "random":
        removed, conn_seq, lcc_seq = random_removal(G, k=k, seed=seed)
    elif method == "betweenness":
        removed, conn_seq, lcc_seq = betweenness_removal(
            G, k=k, betweenness=centralities["betweenness"]
        )
    elif method == "adaptive_betweenness":
        removed, conn_seq, lcc_seq = adaptive_betweenness_greedy(
            G, k=k, betweenness_k=betweenness_k, seed=seed
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime = time.perf_counter() - t0
    peak_mb = mt.stop()

    result = evaluate_run(G, removed, conn_seq, lcc_seq, runtime_s=runtime)
    result["peak_memory_mb"] = peak_mb
    result["seed"] = seed
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Full experiment (n_runs per method)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    dataset_name: str,
    k: int,
    n_runs: int,
    dataset_kwargs: Dict,
    betweenness_k: int = 200,
    config_weights: Dict = None,
) -> None:
    logger.info(f"Loading dataset: {dataset_name} {dataset_kwargs}")
    G = load_dataset(dataset_name, **dataset_kwargs)
    n = G.number_of_nodes()
    e = G.number_of_edges()
    logger.info(f"Graph: {n} nodes, {e} edges")

    methods = [
        "lazy_greedy",
        "lazy_greedy_no_bet",
        "lazy_greedy_no_ci",
        "degree",
        "adaptive_degree",
        "betweenness",
        "adaptive_betweenness",
        "random",
    ]
    all_results: Dict[str, List[Dict]] = {m: [] for m in methods}

    for run_idx in range(n_runs):
        seed = 42 + run_idx
        logger.info(f"=== Run {run_idx + 1}/{n_runs} ===")

        # Re-compute centralities each run with different betweenness seed
        # This gives meaningful variance across runs (different pivot samples)
        logger.info("  Computing centralities …")
        centralities = compute_all_centralities(
            G, betweenness_k=betweenness_k, seed=seed
        )

        for method in methods:
            logger.info(f"  Method: {method}")
            res = run_one(G, method, k, centralities, seed, betweenness_k=betweenness_k)
            all_results[method].append(res)
            logger.info(
                f"    f_reduction={res['f_reduction_pct']:.2f}%  "
                f"lcc_reduction={res['lcc_reduction_pct']:.2f}%  "
                f"R={res['robustness_index']:.4f}  "
                f"time={res['runtime_s']:.2f}s"
            )

    # Aggregate and execute T-tests
    aggregated: Dict[str, Dict] = {}
    from scipy.stats import ttest_ind

    for method in methods:
        aggregated[method] = aggregate_runs(all_results[method])
        
        # Calculate p-value of Robustness Index vs Baseline (Degree)
        if method != "degree" and len(all_results[method]) > 1:
            try:
                our_vals = [r["robustness_index"] for r in all_results[method]]
                base_vals = [r["robustness_index"] for r in all_results.get("degree", [])]
                if len(base_vals) > 1:
                    _, p_val = ttest_ind(our_vals, base_vals, equal_var=False)
                    aggregated[method]["p_value_vs_degree"] = float(p_val)
                    logger.info(f"    [Stat] {method} vs Degree p-value = {p_val:.4e}")
            except Exception:
                pass

    print_results_table(aggregated)

    # Save JSON
    tag = f"{dataset_name}_n{n}_k{k}"
    json_path = os.path.join(LOGS_DIR, f"{tag}_results.json")
    with open(json_path, "w") as fh:
        json.dump({m: agg for m, agg in aggregated.items()}, fh, indent=2)
    logger.info(f"Results saved → {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_connectivity(aggregated, methods, k, tag, n)
    _plot_lcc(aggregated, methods, k, tag, n)


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "lazy_greedy"          : "#E63946",
    "lazy_greedy_no_bet"   : "#F4A261",
    "lazy_greedy_no_ci"    : "#E9C46A",
    "degree"               : "#457B9D",
    "adaptive_degree"      : "#1D3557",
    "betweenness"          : "#2A9D8F",
    "adaptive_betweenness" : "#264653",
    "random"               : "#A8DADC",
}
LABELS = {
    "lazy_greedy"          : "Lazy Greedy (Ours)",
    "lazy_greedy_no_bet"   : "LG Ablation (No Betweenness)",
    "lazy_greedy_no_ci"    : "LG Ablation (No CI)",
    "degree"               : "Static Degree",
    "adaptive_degree"      : "Adaptive Degree",
    "betweenness"          : "Static Betweenness",
    "adaptive_betweenness" : "Adaptive Betweenness",
    "random"               : "Random Removal",
}


def _plot_connectivity(agg, methods, k, tag, n):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(1, k + 1))
    for m in methods:
        mean = np.array(agg[m].get("conn_mean", []))[:k]
        std  = np.array(agg[m].get("conn_std",  []))[:k]
        if len(mean) == 0:
            continue
        xp = x[:len(mean)]
        z = 10 if "lazy_greedy" in m else 1
        ax.plot(xp, mean, color=COLORS[m], label=LABELS[m], linewidth=2, zorder=z)
        ax.fill_between(xp, mean - std, mean + std, alpha=0.15, color=COLORS[m], zorder=z-1)

    ax.set_xlabel("Nodes Removed (k)", fontsize=13)
    ax.set_ylabel("Pairwise Connectivity f(A)", fontsize=13)
    ax.set_title(f"Connectivity Reduction — {tag} (n={n})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{tag}_connectivity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Plot saved → {path}")


def _plot_lcc(agg, methods, k, tag, n):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(1, k + 1))
    for m in methods:
        mean = np.array(agg[m].get("lcc_mean", []))[:k]
        std  = np.array(agg[m].get("lcc_std",  []))[:k]
        if len(mean) == 0:
            continue
        xp = x[:len(mean)]
        z = 10 if "lazy_greedy" in m else 1
        ax.plot(xp, mean, color=COLORS[m], label=LABELS[m], linewidth=2, zorder=z)
        ax.fill_between(xp, mean - std, mean + std, alpha=0.15, color=COLORS[m], zorder=z-1)

    ax.set_xlabel("Nodes Removed (k)", fontsize=13)
    ax.set_ylabel("LCC Size", fontsize=13)
    ax.set_title(f"Largest Component Reduction — {tag} (n={n})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{tag}_lcc.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CNDP Comparative Experiment")
    p.add_argument("--dataset",   default="ba",   help="facebook | twitter | ba | ws | <filepath>")
    p.add_argument("--n",         type=int, default=10_000, help="Nodes (synthetic graphs only)")
    p.add_argument("--m",         type=int, default=3,      help="BA parameter m")
    p.add_argument("--k",         type=int, default=20,     help="Nodes to remove")
    p.add_argument("--runs",      type=int, default=5,      help="Repetitions (≥5)")
    p.add_argument("--bet_k",     type=int, default=200,    help="Betweenness sample size")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(
        level=args.log_level,
        log_file=os.path.join(LOGS_DIR, "run_experiments.log"),
    )

    dataset_kwargs = {}
    if args.dataset in ("ba", "ws"):
        dataset_kwargs["n"] = args.n
    if args.dataset == "ba":
        dataset_kwargs["m"] = args.m

    run_experiment(
        dataset_name   = args.dataset,
        k              = args.k,
        n_runs         = args.runs,
        dataset_kwargs = dataset_kwargs,
        betweenness_k  = args.bet_k,
    )
