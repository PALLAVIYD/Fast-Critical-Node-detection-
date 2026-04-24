"""
networkx_benchmark.py
Comparative benchmark between custom LargeGraph and networkx.Graph
Measures graph generation, edge removal speed + total memory footprint.
Explicitly justifies the architectural choice to build LargeGraph.
"""

import time
import networkx as nx
import tracemalloc
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cndp.graph import LargeGraph

def build_nx_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def build_large_graph(edges):
    return LargeGraph.from_edge_list(edges)

def benchmark(sizes, m=3):
    our_times = []
    nx_times = []
    our_mem = []
    nx_mem = []

    for n in sizes:
        print(f"Benchmarking n={n}...")
        # Pre-generate edges using simple BA to isolate the actual graph memory structure
        edges = list(nx.barabasi_albert_graph(n, m).edges())
        
        # Benchmark NetworkX
        tracemalloc.start()
        G_nx = build_nx_graph(edges)
        _, peak_nx = tracemalloc.get_traced_memory()
        
        t0 = time.time()
        # Test 100 random node removals (mimicking Greedy algorithm)
        import random
        to_remove = random.sample(list(G_nx.nodes()), min(100, n))
        for v in to_remove:
            G_nx.remove_node(v)
        nx_times.append(time.time() - t0)
        nx_mem.append(peak_nx / 1024 / 1024)
        tracemalloc.stop()

        # Benchmark Ours
        tracemalloc.start()
        G_our = build_large_graph(edges)
        _, peak_our = tracemalloc.get_traced_memory()
        
        t0 = time.time()
        for v in to_remove:
            G_our.remove_node(v)
        our_times.append(time.time() - t0)
        our_mem.append(peak_our / 1024 / 1024)
        tracemalloc.stop()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(sizes, our_times, "o-", label="LargeGraph (Ours)", color="#E63946", lw=2)
    ax1.plot(sizes, nx_times, "s-", label="NetworkX", color="#457B9D", lw=2)
    ax1.set_xlabel("Nodes")
    ax1.set_ylabel("Removal Time for 100 Nodes (s)")
    ax1.set_title("Operational Speed Scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(sizes, our_mem, "o-", label="LargeGraph (Ours)", color="#E63946", lw=2)
    ax2.plot(sizes, nx_mem, "s-", label="NetworkX", color="#457B9D", lw=2)
    ax2.set_xlabel("Nodes")
    ax2.set_ylabel("Peak RAM Memory (MB)")
    ax2.set_title("Memory Overload vs NetworkX")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    plt.savefig(os.path.join("results", "plots", "memory_vs_networkx.png"), dpi=150)
    plt.close()
    
    print("Benchmark complete. Plot saved to results/plots/memory_vs_networkx.png")

if __name__ == "__main__":
    benchmark([10000, 50000, 100000, 200000, 500000])
