# Fast Critical Node Detection in Large-Scale Networks

A **production-grade, research-quality** Python implementation of the Critical Node Detection Problem (CNDP) addressing scalability up to **1 million nodes** and temporal/evolving networks.

---

## Project Structure

```
adv_graph/
│
├── cndp/
│   ├── __init__.py        # Clean package imports
│   ├── graph.py           # LargeGraph (adj-list), Union-Find, generators
│   ├── centrality.py      # Degree, approx betweenness, k-core, CI
│   ├── scoring.py         # Min-max normalisation + F(v) aggregation
│   ├── greedy.py          # Lazy greedy CNDP + baselines
│   ├── temporal.py        # Temporal graph: EWMA, velocity scoring, smoothness
│   ├── metrics.py         # f(A), LCC, Robustness Index, multi-run aggregation
│   └── utils.py           # Logging, Timer, MemoryTracker, parallel_map
│
├── datasets/
│   ├── __init__.py
│   └── loader.py          # SNAP Facebook/Twitter + synthetic loaders
│
├── experiments/
│   ├── __init__.py
│   ├── run_experiments.py    # Main comparative experiment (5+ runs, plots)
│   ├── scalability_test.py   # Runtime/memory scaling: 10K→1M nodes
│   └── temporal_test.py      # Temporal CNDP over evolving snapshots
│
├── results/
│   ├── plots/             # Auto-generated PNG figures
│   └── logs/              # JSON result files & text logs
│
├── config.yaml            # All tunable hyper-parameters
└── requirements.txt
```

---

## Setup

### Local (macOS / Linux)
```bash
cd adv_graph
pip install -r requirements.txt
```

### Google Colab
```python
!pip install networkx numpy matplotlib scipy tqdm pyyaml python-louvain
import sys, os
# Mount Drive or upload files, then:
sys.path.insert(0, "/content/adv_graph")
```

---

## Quick Start

```python
from cndp.graph import generate_barabasi_albert
from cndp.centrality import compute_all_centralities
from cndp.scoring import score_nodes
from cndp.greedy import lazy_greedy_cndp
from cndp.metrics import evaluate_run

# 1. Generate a scale-free graph
G = generate_barabasi_albert(n=10_000, m=3)

# 2. Compute all centralities
centralities = compute_all_centralities(G, betweenness_k=200)

# 3. Aggregate into composite score F(v)
scores = score_nodes(centralities)

# 4. Run Lazy Greedy CNDP
removed, conn_seq, lcc_seq = lazy_greedy_cndp(G, k=20, initial_scores=scores)

# 5. Evaluate
result = evaluate_run(G, removed, conn_seq, lcc_seq)
print(f"Connectivity reduction: {result['f_reduction_pct']:.1f}%")
print(f"LCC reduction:          {result['lcc_reduction_pct']:.1f}%")
print(f"Robustness Index (R):   {result['robustness_index']:.4f}")
```

---

## Experiment Scripts

### 1 — Comparative Experiment (Ours vs. 3 baselines)
```bash
# BA graph, 10K nodes, remove 20, 5 runs
python -m experiments.run_experiments --dataset ba --n 10000 --k 20 --runs 5

# Real-world Facebook ego network
python -m experiments.run_experiments --dataset facebook --k 20 --runs 5
```
**Outputs:** `results/plots/<tag>_connectivity.png`, `<tag>_lcc.png`, `results/logs/<tag>_results.json`

### 2 — Scalability Test (10K → 1M nodes)
```bash
python -m experiments.scalability_test --sizes 10000 100000 500000 1000000 --k 20
```
**Outputs:** `results/plots/runtime_scaling_ba.png`, `memory_scaling_ba.png`

### 3 — Temporal CNDP (Evolving snapshots)
```bash
python -m experiments.temporal_test --n 5000 --T 10 --k 15 --rewire 0.05 --alpha 0.5
```
**Outputs:** `results/plots/temporal_n5000_T10.png`

---

## Algorithm Details

### Composite Score F(v)
```
F(v) = α·d̂(v) + β·b̂(v) + γ·k̂(v) + δ·CÎ(v)
```
| Symbol | Meaning | Default weight |
|--------|---------|----------------|
| d̂(v) | Normalised degree centrality | α = 0.20 |
| b̂(v) | Approx betweenness (k=200 samples) | β = 0.20 |
| k̂(v) | k-core coreness | γ = 0.20 |
| CÎ(v) | Connectivity impact | δ = 0.40 |

All components normalised: φ̂(v) = (φ(v) − min) / (max − min)

### Lazy Greedy (Submodularity)
```
Δ(v|A) = f(A) − f(A ∪ {v})
```
Uses a priority queue with cached upper-bound gains. Recomputes only when a node is popped — exponentially fewer evaluations than naive greedy.

### Temporal Score Sᵛ(t)
```
Sᵛ(t) = α·d̂ + β·b̂ + γ·k̂ + δ·CÎ + ε·Δ̂
```
Where Δ̂(v,t) = normalised |degree(v,t) − degree(v,t−1)| (velocity term).

EWMA edge weights: **Wᵗ(u,v) = α·Wᵗ⁻¹(u,v) + (1−α)·W_current(u,v)**

Temporal smoothness penalty: **λ · |Sₜ △ Sₜ₋₁| / |S|**

---

## Scalability Design

| Nodes | Strategy |
|-------|----------|
| ≤ 50K | Full centrality on all nodes |
| 100K | Sampled betweenness (k≤200), candidate pruning |
| 500K | `sample_size=2000` candidate restriction in lazy greedy |
| 1M | BFS/Union-Find only (no full NX), lazy greedy with sampling |

No NetworkX for large graphs — all algorithms use the custom `LargeGraph` adjacency-list structure.

---

## Metrics

| Metric | Definition |
|--------|-----------|
| f(A) | Pairwise connectivity = Σ C(|Sᵢ|, 2) |
| LCC | Largest connected component size |
| R | Robustness Index = (1/n) Σ LCCᵢ/n |
| Runtime | Wall-clock seconds (mean ± std over ≥5 runs) |

---

## Configuration

Edit `config.yaml` to tune all hyper-parameters without touching code:

```yaml
scoring:
  alpha: 0.20   # degree weight
  beta:  0.20   # betweenness weight
  gamma: 0.20   # k-core weight
  delta: 0.40   # connectivity-impact weight

temporal:
  ewma_alpha: 0.50        # history vs current balance
  smoothness_lambda: 0.30 # temporal stability penalty
```

---

## Edge Cases Handled

- **Disconnected graphs** — BFS/Union-Find naturally handles multiple components
- **k > |V|** — all scripts clamp k to min(k, n)
- **Isolated nodes** — degree=0, score→0, never selected by greedy
- **Empty graph after removals** — early stopping with logged warning
