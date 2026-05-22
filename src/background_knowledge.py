"""
background_knowledge.py — Background-knowledge sampling and initialisation.

All functions in this module are part of the **b-LOAD** algorithm introduced
in this work.  They handle the construction of prior causal constraints
(required and forbidden directed edges) from a ground-truth DAG, and their
translation into the adjacency-matrix format consumed by the b-LOAD and b-PC
learners.

Background knowledge encoding
------------------------------
Throughout this codebase, background knowledge is represented as an
``(n, n)`` integer adjacency matrix ``g`` where:

* ``g[i, j] == -1``  and  ``g[j, i] == 1``  →  required edge  i → j
* ``g[i, j] ==  0``                           →  no constraint on edge (i, j)

Forbidden edges are currently represented only as Python sets of tuples
(not encoded into the matrix directly); the matrix stores only *required*
edges.
"""

import math
import random
from typing import Any

import networkx as nx
import numpy as np


def sample_background_knowledge_v2(
    true_dag: nx.DiGraph,
    fraction_required: float = 0.1,
    fraction_forbidden: float = 0.1,
    seed: int = 42,
) -> dict[str, set]:
    """
    Sample required and forbidden edge sets from the ground-truth DAG,
    uniformly at random from the global edge and non-edge pools.

    Required edges are drawn from edges that **exist** in ``true_dag``.
    Forbidden edges are drawn from directed pairs that do **not** exist in
    ``true_dag`` (including reverse directions of true edges).

    .. note::
        Part of the **b-LOAD** algorithm introduced in this work.

    Args:
        true_dag (nx.DiGraph): Ground-truth directed acyclic graph.  Nodes
            are expected to be integer indices.
        fraction_required (float): Fraction of true edges to include as
            required knowledge.  E.g. ``0.3`` means 30 % of true edges are
            revealed.
        fraction_forbidden (float): Fraction of non-edges to mark as
            forbidden.
        seed (int): Random seed for reproducibility.

    Returns:
        dict with keys:

        * ``"required"`` (set of (int, int)): Required directed edges i → j.
        * ``"forbidden"`` (set of (int, int)): Forbidden directed edges i → j.
    """
    random.seed(seed)
    nodes = list(true_dag.nodes())
    all_possible_edges = [(u, v) for u in nodes for v in nodes if u != v]
    true_edges = set(true_dag.edges())

    num_required = int(len(true_edges) * fraction_required)
    required = set(random.sample(list(true_edges), num_required))

    non_edges = [edge for edge in all_possible_edges if edge not in true_edges]
    num_forbidden = int(len(non_edges) * fraction_forbidden)
    forbidden = set(random.sample(non_edges, num_forbidden))

    return {"forbidden": forbidden, "required": required}


def sample_local_background_knowledge(
    true_dag: nx.DiGraph,
    targets: np.ndarray,
    fraction_required: float = 0.1,
    fraction_forbidden: float = 0.1,
    seed: int = 42,
) -> dict[str, set]:
    """
    Sample required and forbidden edges restricted to the *local neighbourhood*
    of the target nodes (edges incident on at least one target).

    This is the **local** background-knowledge sampling strategy used in the
    b-LOAD experiments: instead of revealing global graph structure, only edges
    that touch the treatment or outcome are made available.

    .. note::
        Part of the **b-LOAD** algorithm introduced in this work.

    Args:
        true_dag (nx.DiGraph): Ground-truth DAG.
        targets (np.ndarray): Array of target node indices (typically the
            treatment–outcome pair).
        fraction_required (float): Fraction of local true edges to include as
            required knowledge.  A minimum of 1 edge is sampled if the
            fraction is positive and the pool is non-empty.
        fraction_forbidden (float): Fraction of local non-edges to mark as
            forbidden.
        seed (int): Random seed for reproducibility.

    Returns:
        dict with keys:

        * ``"required"`` (set of (int, int)): Required directed edges.
        * ``"forbidden"`` (set of (int, int)): Forbidden directed edges.
    """
    random.seed(seed)
    nodes = list(true_dag.nodes())
    target_set = set(targets)
    true_edges = list(true_dag.edges())

    local_true_edges = [
        (u, v) for u, v in true_edges
        if u in target_set or v in target_set
    ]

    all_possible_local = [
        (u, v) for u in nodes for v in nodes
        if u != v and (u in target_set or v in target_set)
    ]
    local_non_edges = [e for e in all_possible_local if e not in set(true_edges)]

    def get_sample_count(pool: list, fraction: float) -> int:
        if fraction <= 0 or not pool:
            return 0
        return max(1, math.ceil(len(pool) * fraction))

    num_req = get_sample_count(local_true_edges, fraction_required)
    num_forb = get_sample_count(local_non_edges, fraction_forbidden)

    required = set(random.sample(local_true_edges, num_req)) if num_req > 0 else set()
    forbidden = set(random.sample(local_non_edges, num_forb)) if num_forb > 0 else set()

    return {"forbidden": forbidden, "required": required}


def initialize_background_knowledge(num_nodes: int, bk_dict: dict) -> np.ndarray:
    """
    Convert a background-knowledge dictionary into an MPDAG adjacency matrix.

    Required edges from ``bk_dict["required"]`` are encoded as directed edges
    in the output matrix.  Forbidden edges (``bk_dict["forbidden"]``) are
    **not** encoded in the matrix — their enforcement is handled externally
    (e.g. by preventing the skeleton search from drawing those edges).

    The returned matrix is used to initialise ``g`` in
    :func:`~src.mb_by_mb.mb_by_mb_in_mpdag_alg` and as the ``background_knowledge``
    mask in :func:`~src.mb_by_mb.update_graph_mpdag`.

    .. note::
        Part of the **b-LOAD** algorithm introduced in this work.

    Args:
        num_nodes (int): Total number of variables / nodes in the dataset.
        bk_dict (dict): Background-knowledge dictionary with optional keys:

            * ``"required"`` (set of (int, int)): Each tuple ``(i, j)``
              encodes the required directed edge i → j.
            * ``"forbidden"`` (set of (int, int)): Currently unused in the
              matrix encoding.

    Returns:
        np.ndarray: Integer matrix of shape ``(num_nodes, num_nodes)`` where
            ``g[i,j] == -1`` and ``g[j,i] == 1`` for every required edge i→j,
            and all other entries are ``0``.
    """
    g = np.zeros((num_nodes, num_nodes), dtype=int)

    if "required" in bk_dict:
        for i, j in bk_dict["required"]:
            g[i, j] = -1  # tail at i
            g[j, i] = 1   # arrowhead at j

    return g


def sample_local_background_knowledge_noised(
    true_dag: nx.DiGraph,
    targets: np.ndarray,
    fraction_required: float = 0.1,
    fraction_forbidden: float = 0.1,
    noise_mu: float = 0.2,
    seed: int = 42,
) -> dict[str, set]:
    """
    Sample local background knowledge with Gaussian noise applied to the
    required edge set.

    This is the **noised** variant of :func:`sample_local_background_knowledge`,
    used to study the robustness of b-LOAD to incorrect prior information.
    After sampling the required and forbidden sets as usual, a fraction of the
    required edges are replaced by randomly selected non-edges (false positives
    in the required-edge set).

    The noise rate is itself drawn from a Gaussian distribution
    ``N(noise_mu, (noise_mu/3)²)`` clipped to ``[0, 1]``, so that the
    experiments reflect real-world uncertainty about how reliable the prior
    information is.

    .. note::
        Part of the **b-LOAD** algorithm introduced in this work.

    Args:
        true_dag (nx.DiGraph): Ground-truth DAG.
        targets (np.ndarray): Target node indices.
        fraction_required (float): Fraction of local true edges to sample as
            required knowledge (before corruption).
        fraction_forbidden (float): Fraction of local non-edges to sample as
            forbidden.
        noise_mu (float): Mean of the Gaussian noise distribution controlling
            what fraction of required edges are corrupted with false positives.
            ``0.0`` means no noise; ``1.0`` means all required edges are
            replaced.
        seed (int): Random seed for reproducibility (both ``random`` and
            ``numpy.random``).

    Returns:
        dict with keys:

        * ``"required"`` (set of (int, int)): Required edges (possibly
          containing some incorrect entries due to noise).
        * ``"forbidden"`` (set of (int, int)): Forbidden edges (unaffected by
          noise).
    """
    random.seed(seed)
    np.random.seed(seed)

    noise_sigma = noise_mu / 3

    nodes = list(true_dag.nodes())
    target_set = set(targets)
    true_edges = set(true_dag.edges())

    local_true_edges = [
        (u, v) for u, v in true_edges
        if u in target_set or v in target_set
    ]

    all_possible_local = [
        (u, v) for u in nodes for v in nodes
        if u != v and (u in target_set or v in target_set)
    ]
    local_non_edges = [e for e in all_possible_local if e not in true_edges]

    def get_sample_count(pool: list, fraction: float) -> int:
        if fraction <= 0 or not pool:
            return 0
        return max(1, math.ceil(len(pool) * fraction))

    num_req = get_sample_count(local_true_edges, fraction_required)
    num_forb = get_sample_count(local_non_edges, fraction_forbidden)

    required = set(random.sample(local_true_edges, num_req)) if num_req > 0 else set()
    forbidden = set(random.sample(local_non_edges, num_forb)) if num_forb > 0 else set()

    # Draw noise rate from Gaussian, clipped to [0, 1]
    noise_rate = float(
        np.clip(np.random.normal(loc=noise_mu, scale=noise_sigma), 0.0, 1.0)
    )

    # Corrupt required edges: replace k edges with random non-edges
    n = len(required)
    if n == 0 or noise_rate == 0.0 or not local_non_edges:
        return {"required": required, "forbidden": forbidden}

    if n == 1:
        if random.random() < noise_rate:
            required = set(random.sample(local_non_edges, 1))
    else:
        k = min(round(noise_rate * n), len(local_non_edges))
        if k > 0:
            to_corrupt = set(random.sample(list(required), k))
            false_edges = set(random.sample(local_non_edges, k))
            required = (required - to_corrupt) | false_edges

    return {"required": required, "forbidden": forbidden}
