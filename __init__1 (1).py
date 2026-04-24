"""cndp package — Fast Critical Node Detection in Large-Scale Networks"""
from cndp.graph import LargeGraph, generate_barabasi_albert, generate_watts_strogatz
from cndp.centrality import compute_all_centralities
from cndp.scoring import score_nodes
from cndp.greedy import lazy_greedy_cndp, degree_greedy, random_removal, betweenness_removal
from cndp.metrics import evaluate_run, aggregate_runs
from cndp.temporal import TemporalGraphManager
