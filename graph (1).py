"""
graph.py — Core Graph Module
Handles graph loading, adjacency list, BFS/Union-Find connectivity.
Uses NetworkX only for small graphs (<50K nodes); custom impl for large.
"""

import random
import collections
import time
import logging
from typing import Dict, List, Set, Tuple, Optional, Iterator

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom scalable adjacency-list graph
# ─────────────────────────────────────────────────────────────────────────────

class LargeGraph:
    """
    Memory-efficient undirected graph backed by a dict of sets.
    Designed for graphs with 100K–1M nodes.
    """

    def __init__(self):
        self._adj: Dict[int, Set[int]] = {}
        self._num_edges: int = 0

    # ── Construction ──────────────────────────────────────────────────────────

    def add_node(self, u: int):
        if u not in self._adj:
            self._adj[u] = set()

    def add_edge(self, u: int, v: int):
        if u == v:
            return
        if u not in self._adj:
            self._adj[u] = set()
        if v not in self._adj:
            self._adj[v] = set()
        if v not in self._adj[u]:
            self._adj[u].add(v)
            self._adj[v].add(u)
            self._num_edges += 1

    def remove_node(self, u: int):
        if u not in self._adj:
            return
        for nb in list(self._adj[u]):
            self._adj[nb].discard(u)
            self._num_edges -= 1
        del self._adj[u]

    def remove_edge(self, u: int, v: int):
        if u in self._adj and v in self._adj[u]:
            self._adj[u].discard(v)
            self._adj[v].discard(u)
            self._num_edges -= 1

    # ── Properties ───────────────────────────────────────────────────────────

    def nodes(self) -> List[int]:
        return list(self._adj.keys())

    def edges(self) -> Iterator[Tuple[int, int]]:
        seen = set()
        for u, nbrs in self._adj.items():
            for v in nbrs:
                if (v, u) not in seen:
                    seen.add((u, v))
                    yield (u, v)

    def neighbors(self, u: int) -> Set[int]:
        return self._adj.get(u, set())

    def degree(self, u: int) -> int:
        return len(self._adj.get(u, set()))

    def number_of_nodes(self) -> int:
        return len(self._adj)

    def number_of_edges(self) -> int:
        return self._num_edges

    def has_node(self, u: int) -> bool:
        return u in self._adj

    def copy(self) -> "LargeGraph":
        g = LargeGraph()
        for u, nbrs in self._adj.items():
            g._adj[u] = set(nbrs)
        g._num_edges = self._num_edges
        return g

    # ── Connectivity (Union-Find) ─────────────────────────────────────────────

    def connected_components(self) -> List[List[int]]:
        """BFS-based connected components — O(V+E)."""
        visited = set()
        components = []
        for start in self._adj:
            if start in visited:
                continue
            comp = []
            queue = collections.deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in self._adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            components.append(comp)
        return components

    def largest_connected_component_size(self) -> int:
        best = 0
        visited = set()
        for start in self._adj:
            if start in visited:
                continue
            size = 0
            queue = collections.deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                size += 1
                for nb in self._adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            best = max(best, size)
        return best

    def pairwise_connectivity(self) -> int:
        """Sum of C(|Si|, 2) over all components — submodular objective."""
        total = 0
        for comp in self.connected_components():
            s = len(comp)
            total += s * (s - 1) // 2
        return total

    # ── Degree sequence ───────────────────────────────────────────────────────

    def degree_dict(self) -> Dict[int, int]:
        return {u: len(nbrs) for u, nbrs in self._adj.items()}

    # ── Conversion helpers ────────────────────────────────────────────────────

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        for u, nbrs in self._adj.items():
            for v in nbrs:
                G.add_edge(u, v)
        return G

    @staticmethod
    def from_networkx(G: nx.Graph) -> "LargeGraph":
        lg = LargeGraph()
        for u, v in G.edges():
            lg.add_edge(u, v)
        return lg

    @staticmethod
    def from_edge_list(edges: List[Tuple[int, int]]) -> "LargeGraph":
        lg = LargeGraph()
        for u, v in edges:
            lg.add_edge(u, v)
        return lg


# ─────────────────────────────────────────────────────────────────────────────
#  Union-Find for incremental connectivity
# ─────────────────────────────────────────────────────────────────────────────

class UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self, nodes: List[int]):
        self.parent = {n: n for n in nodes}
        self.rank   = {n: 0  for n in nodes}
        self.size   = {n: 1  for n in nodes}
        self.num_components = len(nodes)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.num_components -= 1
        return True

    def component_size(self, x: int) -> int:
        return self.size[self.find(x)]

    def pairwise_connectivity(self) -> int:
        comp_sizes: Dict[int, int] = {}
        for n in self.parent:
            root = self.find(n)
            comp_sizes[root] = comp_sizes.get(root, 0) + 1
        return sum(s * (s - 1) // 2 for s in comp_sizes.values())


# ─────────────────────────────────────────────────────────────────────────────
#  Graph generators (synthetic benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

def generate_barabasi_albert(n: int, m: int = 3, seed: int = 42) -> LargeGraph:
    """
    Scale-free graph via Barabási–Albert preferential attachment.

    Uses NetworkX for n ≤ 500K (fast C-backed random.choices approach),
    and a numpy vectorised repeat-array trick for n > 500K.

    Parameters
    ----------
    n : number of nodes
    m : edges added per new node
    """
    import numpy as np

    logger.info(f"Generating BA graph: n={n}, m={m}")
    t0 = time.time()

    SMALL_THRESHOLD = 500_000

    if n <= SMALL_THRESHOLD:
        # NetworkX's implementation uses Python random.choices — very fast
        G_nx = nx.barabasi_albert_graph(n, m, seed=seed)
        lg = LargeGraph.from_networkx(G_nx)
    else:
        # For very large graphs: numpy repeat-array trick (O(E) memory, O(1) per draw)
        rng = np.random.default_rng(seed)

        # Start with a small seed clique
        seed_nodes = list(range(m + 1))
        edges: list = []
        repeated: list = []  # flat list: each node appears degree(v) times

        for u in seed_nodes:
            for v in seed_nodes:
                if u < v:
                    edges.append((u, v))
                    repeated.extend([u, v])

        import random as pyrand

        for new_node in range(m + 1, n):
            # Draw m targets with probability ∝ degree (with replacement, then dedup)
            if not repeated:
                break
            
            # Using Python's native fast sampling to pull indices
            over  = min(m * 3, len(repeated))
            drawn_indices = pyrand.sample(range(len(repeated)), over)
            
            seen  = set()
            targets = []
            for idx in drawn_indices:
                ti = int(repeated[idx])
                if ti != new_node and ti not in seen:
                    seen.add(ti)
                    targets.append(ti)
                    if len(targets) == m:
                        break
            if not targets:
                targets = [int(repeated[pyrand.randrange(len(repeated))])]

            for t in targets:
                edges.append((new_node, t))
                repeated.append(new_node)
                repeated.append(t)

            if new_node % 50_000 == 0:
                logger.info(f"  BA progress: {new_node:,}/{n:,} nodes")

        lg = LargeGraph.from_edge_list(edges)

    elapsed = time.time() - t0
    logger.info(f"BA graph built in {elapsed:.2f}s — {lg.number_of_nodes()} nodes, {lg.number_of_edges()} edges")
    return lg


def generate_watts_strogatz(n: int, k: int = 6, p: float = 0.1, seed: int = 42) -> LargeGraph:
    """
    Small-world graph via Watts–Strogatz rewiring.
    Uses NetworkX for correctness then converts.
    """
    logger.info(f"Generating WS graph: n={n}, k={k}, p={p}")
    t0 = time.time()
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    lg = LargeGraph.from_networkx(G)
    elapsed = time.time() - t0
    logger.info(f"WS graph built in {elapsed:.2f}s — {lg.number_of_nodes()} nodes, {lg.number_of_edges()} edges")
    return lg


def generate_erdos_renyi(n: int, p: float = 0.0001, seed: int = 42) -> LargeGraph:
    """Random Erdős–Rényi graph. Uses fast O(n+m) generation."""
    logger.info(f"Generating ER graph: n={n}, p={p}")
    G_nx = nx.fast_gnp_random_graph(n, p, seed=seed)
    return LargeGraph.from_networkx(G_nx)


# ─────────────────────────────────────────────────────────────────────────────
#  Thin wrappers that return either LargeGraph or nx.Graph depending on size
# ─────────────────────────────────────────────────────────────────────────────

SMALL_GRAPH_THRESHOLD = 50_000  # below this, use networkx


def smart_graph(n: int, graph_type: str = "ba", **kwargs):
    """Return the right graph type based on n."""
    if graph_type == "ba":
        return generate_barabasi_albert(n, **kwargs)
    elif graph_type == "ws":
        return generate_watts_strogatz(n, **kwargs)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
