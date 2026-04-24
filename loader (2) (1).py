"""
datasets/loader.py — Dataset loading for real-world and synthetic graphs.
Handles:
  - SNAP Facebook ego network
  - SNAP Twitter (sampled)
  - Synthetic BA / WS generation
  - Edge-list files (generic)
"""

import gzip
import logging
import os
import urllib.request
from typing import Optional, Tuple

from cndp.graph import LargeGraph, generate_barabasi_albert, generate_watts_strogatz

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Generic edge-list loader
# ─────────────────────────────────────────────────────────────────────────────

def load_edge_list(
    filepath: str,
    comment: str = "#",
    delimiter: Optional[str] = None,
    limit_edges: Optional[int] = None,
) -> LargeGraph:
    """
    Load an undirected graph from a plain or gzipped edge-list file.
    Lines starting with `comment` are skipped.
    """
    G = LargeGraph()
    opener = gzip.open if filepath.endswith(".gz") else open
    count  = 0
    with opener(filepath, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            parts = line.split(delimiter)
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            G.add_edge(u, v)
            count += 1
            if limit_edges and count >= limit_edges:
                break
    logger.info(f"Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges from {filepath}")
    return G


# ─────────────────────────────────────────────────────────────────────────────
#  SNAP Facebook ego network (~4K nodes, ~88K edges)
# ─────────────────────────────────────────────────────────────────────────────

FACEBOOK_URL  = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
FACEBOOK_FILE = os.path.join(DATA_DIR, "facebook_combined.txt.gz")


def load_facebook(use_cache: bool = True, limit_edges: Optional[int] = None) -> LargeGraph:
    """Download (once) and load the SNAP Facebook ego graph."""
    if not os.path.exists(FACEBOOK_FILE) or not use_cache:
        logger.info(f"Downloading Facebook dataset → {FACEBOOK_FILE}")
        urllib.request.urlretrieve(FACEBOOK_URL, FACEBOOK_FILE)
    else:
        logger.info(f"Using cached Facebook dataset: {FACEBOOK_FILE}")
    return load_edge_list(FACEBOOK_FILE, limit_edges=limit_edges)


# ─────────────────────────────────────────────────────────────────────────────
#  SNAP Twitter (large — sampled subset by default)
# ─────────────────────────────────────────────────────────────────────────────

TWITTER_URL  = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
TWITTER_FILE = os.path.join(DATA_DIR, "twitter_combined.txt.gz")


def load_twitter(
    use_cache: bool = True,
    limit_edges: int = 500_000,
) -> LargeGraph:
    """
    Download (once) and load the SNAP Twitter network.
    By default loads only the first `limit_edges` edges for tractability.
    """
    if not os.path.exists(TWITTER_FILE) or not use_cache:
        logger.info(f"Downloading Twitter dataset → {TWITTER_FILE}")
        urllib.request.urlretrieve(TWITTER_URL, TWITTER_FILE)
    else:
        logger.info(f"Using cached Twitter dataset: {TWITTER_FILE}")
    return load_edge_list(TWITTER_FILE, limit_edges=limit_edges)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic generators (wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def load_barabasi_albert(n: int, m: int = 3, seed: int = 42) -> LargeGraph:
    return generate_barabasi_albert(n, m=m, seed=seed)


def load_watts_strogatz(n: int, k: int = 6, p: float = 0.1, seed: int = 42) -> LargeGraph:
    return generate_watts_strogatz(n, k=k, p=p, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Named loader dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str, **kwargs) -> LargeGraph:
    """
    name: one of 'facebook', 'twitter', 'ba', 'ws', or a file path.
    Extra kwargs forwarded to the specific loader.
    """
    name_lower = name.lower()
    if name_lower == "facebook":
        return load_facebook(**kwargs)
    if name_lower == "twitter":
        return load_twitter(**kwargs)
    if name_lower == "ba":
        return load_barabasi_albert(**kwargs)
    if name_lower == "ws":
        return load_watts_strogatz(**kwargs)
    if os.path.isfile(name):
        return load_edge_list(name, **kwargs)
    raise ValueError(f"Unknown dataset: {name}. "
                     f"Choose from 'facebook', 'twitter', 'ba', 'ws', or a file path.")
