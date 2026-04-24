"""
experiments/scalability_test.py
Runtime and memory scaling test across graph sizes: 10K, 100K, 500K, 1M nodes.

Usage:
    python -m experiments.scalability_test
    python -m experiments.scalability_test --sizes 10000 100000 --k 20
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cndp.graph import generate_barabasi_albert, generate_watts_strogatz
from cndp.centrality import compute_all_centralities
from cndp.scoring import score_nodes
from cndp.greedy import lazy_greedy_cndp
from cndp.utils import setup_logging, MemoryTracker

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Single scale test
# ─────────────────────────────────────────────────────────────────────────────

def test_one_scale(
    n: int,
    k: int,
    graph_type: str = "ba",
    m: int = 3,
    betweenness_k: int = 100,
    seed: int = 42,
) -> dict:
    logger.info(f"Scale test: n={n:>8,d} | graph={graph_type}")

    # ── Generate graph ────────────────────────────────────────────────────────
    mt = MemoryTracker()
    mt.start()

    t0 = time.perf_counter()
    if graph_type == "ba":
        G = generate_barabasi_albert(n, m=m, seed=seed)
    else:
        G = generate_watts_strogatz(n, seed=seed)
    t_gen = time.perf_counter() - t0
    mem_gen = mt.current_mb()

    # ── Centrality ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    centralities = compute_all_centralities(G, betweenness_k=betweenness_k, seed=seed)
    t_cent = time.perf_counter() - t0

    # ── Scoring + Greedy ──────────────────────────────────────────────────────
    scores = score_nodes(centralities)
    t0 = time.perf_counter()
    _, conn_seq, lcc_seq = lazy_greedy_cndp(
        G, k=k, initial_scores=scores,
        sample_size=min(2000, n),
        seed=seed, verbose=False
    )
    t_greedy = time.perf_counter() - t0
    peak_mb = mt.stop()

    result = {
        "n"             : n,
        "edges"         : G.number_of_edges(),
        "graph_type"    : graph_type,
        "k"             : k,
        "t_gen_s"       : round(t_gen,    3),
        "t_cent_s"      : round(t_cent,   3),
        "t_greedy_s"    : round(t_greedy, 3),
        "t_total_s"     : round(t_gen + t_cent + t_greedy, 3),
        "peak_mem_mb"   : round(peak_mb, 2),
        "final_lcc"     : lcc_seq[-1] if lcc_seq else 0,
    }

    logger.info(
        f"  gen={t_gen:.2f}s  cent={t_cent:.2f}s  greedy={t_greedy:.2f}s  "
        f"peak={peak_mb:.1f}MB"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Run across all sizes
# ─────────────────────────────────────────────────────────────────────────────

def run_scalability_test(
    sizes: list,
    k: int = 20,
    graph_type: str = "ba",
    betweenness_k: int = 100,
):
    records = []
    for n in sizes:
        rec = test_one_scale(
            n=n, k=k, graph_type=graph_type,
            betweenness_k=betweenness_k
        )
        records.append(rec)

    # Print table
    print(f"\n{'n':>10s} | {'Edges':>12s} | {'Gen(s)':>8s} | {'Cent(s)':>9s} | "
          f"{'Greedy(s)':>10s} | {'Total(s)':>9s} | {'Peak(MB)':>9s}")
    print("-" * 90)
    for r in records:
        print(f"{r['n']:>10,d} | {r['edges']:>12,d} | {r['t_gen_s']:>8.2f} | "
              f"{r['t_cent_s']:>9.2f} | {r['t_greedy_s']:>10.2f} | "
              f"{r['t_total_s']:>9.2f} | {r['peak_mem_mb']:>9.1f}")

    # Save JSON
    json_path = os.path.join(LOGS_DIR, f"scalability_{graph_type}.json")
    with open(json_path, "w") as fh:
        json.dump(records, fh, indent=2)
    logger.info(f"Scalability results → {json_path}")

    # Plot runtime scaling
    _plot_runtime_scaling(records, graph_type)
    _plot_memory_scaling(records, graph_type)

    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_runtime_scaling(records, graph_type):
    ns      = [r["n"]          for r in records]
    t_cent  = [r["t_cent_s"]   for r in records]
    t_greed = [r["t_greedy_s"] for r in records]
    t_total = [r["t_total_s"]  for r in records]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, t_cent,  "o-", label="Centrality Computation", color="#457B9D", lw=2)
    ax.plot(ns, t_greed, "s-", label="Lazy Greedy",            color="#E63946", lw=2)
    ax.plot(ns, t_total, "^-", label="Total",                  color="#2A9D8F", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Graph Size (nodes)", fontsize=13)
    ax.set_ylabel("Runtime (seconds)", fontsize=13)
    ax.set_title(f"Runtime Scaling — {graph_type.upper()} Graphs", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"runtime_scaling_{graph_type}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Runtime scaling plot → {path}")


def _plot_memory_scaling(records, graph_type):
    ns  = [r["n"]           for r in records]
    mem = [r["peak_mem_mb"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, mem, "o-", color="#E76F51", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Graph Size (nodes)", fontsize=13)
    ax.set_ylabel("Peak Memory (MB)", fontsize=13)
    ax.set_title(f"Memory Scaling — {graph_type.upper()} Graphs", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"memory_scaling_{graph_type}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Memory scaling plot → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CNDP Scalability Test")
    p.add_argument(
        "--sizes", type=int, nargs="+",
        default=[10_000, 100_000, 500_000, 1_000_000],
        help="List of n values to test"
    )
    p.add_argument("--k",         type=int, default=20,  help="Nodes to remove")
    p.add_argument("--graph",     default="ba",           help="ba | ws")
    p.add_argument("--bet_k",     type=int, default=100, help="Betweenness sample size")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(
        level=args.log_level,
        log_file=os.path.join(LOGS_DIR, "scalability_test.log"),
    )
    run_scalability_test(
        sizes         = args.sizes,
        k             = args.k,
        graph_type    = args.graph,
        betweenness_k = args.bet_k,
    )
