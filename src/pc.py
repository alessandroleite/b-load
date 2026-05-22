"""
pc.py — PC and b-PC global causal structure learning.

Two algorithms are provided:

1. **PC** — standard PC algorithm producing a CPDAG via skeleton search,
   v-structure orientation, and three-rule Meek propagation.
   Functions: ``_meek``, ``pc``.

2. **b-PC** — PC extended with background knowledge, producing an MPDAG.
   Required edges from the background-knowledge matrix are injected into the
   CPDAG after the standard PC run, and a four-rule Meek procedure (R1–R3
   structural + R4 using the sep-set) propagates the induced orientations.
   Part of the b-LOAD algorithm introduced in this work.
   Functions: ``_meek4``, ``pc_with_bk``.

Graph encoding
--------------
``g[i, j] == -1``  →  tail at i  (i — j or i → j)
``g[i, j] ==  1``  →  head at i  (j → i)
``g[i, j] ==  0``  →  no edge

An undirected edge i — j satisfies  g[i,j] == g[j,i] == -1.
A directed edge  i → j satisfies    g[i,j] == -1  and  g[j,i] == 1.

The final ``"amat"`` output follows a different convention (used by the
evaluation code): ``amat[i][j] == 1`` indicates a tail at i (i.e., an edge
from i toward j, directed or undirected).
"""

from collections import defaultdict
from typing import Callable

import numpy as np

from src.mb_by_mb import skeleton_step, v_struc_pc


# ---------------------------------------------------------------------------
# PC (base)
# ---------------------------------------------------------------------------


def _meek(pdag: np.ndarray) -> np.ndarray:
    """
    Apply the standard three-rule Meek orientation rules (R1–R3) to a PDAG.

    Rules operate on graph *structure alone* (no sep-set conditioning), which
    matches the original PC algorithm specification.  All matching patterns are
    oriented simultaneously per rule before re-checking (vectorised).

    * **R1** — i → j — k, i not adjacent to k  ⟹  j → k
    * **R2** — i → j → k — i  ⟹  i → k  (acyclicity)
    * **R3** — i — k -/- j, i — l -/- j, k → i, l → i, i — j  ⟹  i → j

    Args:
        pdag (np.ndarray): Partially directed graph (copied internally; input
            is not mutated).

    Returns:
        np.ndarray: CPDAG with all Meek-propagatable orientations resolved.
    """

    def rule1(pdag: np.ndarray) -> bool:
        """R1: i → j — k, i -/- k  ⟹  j → k."""
        i, j = np.where((pdag == -1) & (pdag.T == 1))  # i -> j
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == -1))  # j - k
            & (pdag[i] == 0)                         # i -/- k
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        pdag[k, j] = 1
        return len(i) > 0

    def rule2(pdag: np.ndarray) -> bool:
        """R2: i → j → k — i  ⟹  i → k  (closing a directed cycle)."""
        i, j = np.where((pdag == -1) & (pdag.T == 1))  # i -> j
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == 1))    # j -> k
            & ((pdag[i] == -1) & (pdag.T[i] == -1)) # i - k
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        pdag[k, i] = 1
        return len(i) > 0

    def rule3(pdag: np.ndarray) -> bool:
        """R3: i — k -/- j, i — l -/- j, k → i, l → i, i — j  ⟹  i → j."""
        i, j = np.where((pdag == -1) & (pdag.T == -1))  # i - j
        k = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))  # k - i
            & (pdag[j] == 0)                         # k -/- j
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        mask = j < k
        i, j, k = i[mask], j[mask], k[mask]
        l = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))  # i - l
            & ((pdag[j] == -1) & (pdag.T[j] == 1)) # j -> l
            & ((pdag[k] == -1) & (pdag.T[k] == 1)) # k -> l
        )
        i, j, k, l = i[l[0]], j[l[0]], k[l[0]], l[1]
        pdag[l, i] = 1
        return len(i) > 0

    pdag = pdag.copy()
    while rule1(pdag) or rule2(pdag) or rule3(pdag):
        continue
    return pdag


def pc(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    **kwargs,
) -> dict:
    """
    PC algorithm for global causal structure learning.

    Learns a CPDAG (completed partially directed acyclic graph) representing
    the Markov equivalence class of the underlying DAG.

    Pipeline:

    1. **Skeleton search**: iteratively remove edges using CI tests (growing
       conditioning-set size).
    2. **V-structure orientation**: orient unshielded colliders from the
       skeleton and separating sets.
    3. **Meek propagation**: apply three-rule Meek (R1–R3) to orient as many
       remaining edges as possible.

    Args:
        data (np.ndarray): Data matrix of shape (n_samples, n_nodes).
        ci_test (Callable[[int, int, list[int]], float]): CI test returning a
            p-value given (X, Y, conditioning_set).
        alpha (float): Significance level — edges whose CI test p-value ≥
            alpha are removed from the skeleton.
        **kwargs: Additional keyword arguments are silently ignored (for
            interface compatibility).

    Returns:
        dict with key ``"amat"``: CPDAG adjacency matrix as a nested list.
            ``amat[i][j] == 1`` means there is an edge with tail at i
            (i → j for a directed edge, or i — j for undirected).
            Undirected edges satisfy ``amat[i][j] == amat[j][i] == 1``.
    """
    n = data.shape[1]
    skeleton = np.full((n, n), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    order = 0
    while np.amax(np.sum(skeleton != 0, axis=1)) > order:
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set)
        order += 1

    pdag = v_struc_pc(skeleton, sep_set)
    cpdag = _meek(pdag)

    # Convert internal encoding (tail at i = -1) to output encoding (tail at i = 1)
    amat = np.zeros((n, n), dtype=int)
    amat[cpdag == -1] = 1
    return {"amat": amat.tolist()}


# ---------------------------------------------------------------------------
# b-PC — background-knowledge extension (part of the b-LOAD algorithm)
# ---------------------------------------------------------------------------


def _meek4(pdag: np.ndarray, sep_set: dict) -> np.ndarray:
    """
    Four-rule Meek for b-PC: R1–R3 structural rules plus R4 using the sep-set.

    R1–R3 are the same purely structural rules as :func:`_meek`.
    R4 additionally consults the sep-set from the PC skeleton search:

    * **R4** — undirected a — b; a — c (undir), b → c; a — d (undir), d → c;
      b not adjacent to d; and a in every Sep(b, d)  ⟹  orient a → b.

    R4 propagates orientations introduced by injected background-knowledge
    edges that R1–R3 alone cannot deduce from graph topology.

    .. note::
        Part of the **b-LOAD / b-PC** algorithm introduced in this work.

    Args:
        pdag (np.ndarray): Partially directed graph (copied internally).
        sep_set (dict): Separating sets from the skeleton search, keyed by
            ``frozenset({i, k})``.  Each value is a list of conditioning sets
            that d-separated i and k.

    Returns:
        np.ndarray: Graph with all four-rule Meek propagatable orientations
            resolved.
    """

    def rule1(pdag: np.ndarray) -> bool:
        """R1: i → j — k, i -/- k  ⟹  j → k."""
        i, j = np.where((pdag == -1) & (pdag.T == 1))
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == -1))
            & (pdag[i] == 0)
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        pdag[k, j] = 1
        return len(i) > 0

    def rule2(pdag: np.ndarray) -> bool:
        """R2: i → j → k — i  ⟹  i → k."""
        i, j = np.where((pdag == -1) & (pdag.T == 1))
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == 1))
            & ((pdag[i] == -1) & (pdag.T[i] == -1))
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        pdag[k, i] = 1
        return len(i) > 0

    def rule3(pdag: np.ndarray) -> bool:
        """R3: i — k -/- j, i — l -/- j, k → i, l → i, i — j  ⟹  i → j."""
        i, j = np.where((pdag == -1) & (pdag.T == -1))
        k = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))
            & (pdag[j] == 0)
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        mask = j < k
        i, j, k = i[mask], j[mask], k[mask]
        l = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))
            & ((pdag[j] == -1) & (pdag.T[j] == 1))
            & ((pdag[k] == -1) & (pdag.T[k] == 1))
        )
        i, j, k, l = i[l[0]], j[l[0]], k[l[0]], l[1]
        pdag[l, i] = 1
        return len(i) > 0

    def rule4(pdag: np.ndarray, sep_set: dict) -> bool:
        """
        R4: a — b (undir); a — c (undir), b → c; a — d (undir), d → c;
            b -/- d; a ∈ Sep(b, d)  ⟹  a → b.

        a being in every Sep(b, d) certifies it is a non-collider on
        b — a — d, so the only consistent orientation of a — b is a → b.
        This handles BK-induced directed edges that R1–R3 cannot propagate.
        """
        a, b = np.where((pdag == -1) & (pdag.T == -1))
        changed = False
        for a_, b_ in zip(a, b):
            c_list = np.where(
                ((pdag[a_] == -1) & (pdag.T[a_] == -1))   # a - c
                & ((pdag[b_] == -1) & (pdag.T[b_] == 1))  # b -> c
            )[0]
            for c_ in c_list:
                d_list = np.where(
                    ((pdag[a_] == -1) & (pdag.T[a_] == -1))   # a - d
                    & ((pdag[c_] == -1) & (pdag.T[c_] == 1))  # d -> c
                    & (pdag[b_] == 0)                           # b -/- d
                )[0]
                for d_ in d_list:
                    if frozenset((b_, d_)) in sep_set and all(
                        a_ in S for S in sep_set[frozenset((b_, d_))]
                    ):
                        pdag[b_, a_] = 1  # a -> b
                        changed = True
                        break
                if changed:
                    break
        return changed

    pdag = pdag.copy()
    while rule1(pdag) or rule2(pdag) or rule3(pdag) or rule4(pdag, sep_set):
        continue
    return pdag


def pc_with_bk(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    background_knowledge: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """
    b-PC: PC algorithm with background-knowledge injection and four-rule Meek.

    Extends the standard :func:`pc` pipeline with two additional steps:

    1. After the CPDAG is built by standard PC, **required edges** from
       ``background_knowledge`` are overlaid directly onto the CPDAG.
    2. A **four-rule Meek** sweep (R1–R3 structural + R4 with sep-set) is run
       to propagate orientations induced by the injected edges.

    Forbidden edges in ``background_knowledge`` are ignored; only required
    edges (non-zero entries in the matrix) are injected.

    .. note::
        Part of the **b-LOAD / b-PC** algorithm introduced in this work.

    Args:
        data (np.ndarray): Data matrix of shape (n_samples, n_nodes).
        ci_test (Callable[[int, int, list[int]], float]): CI test returning a
            p-value given (X, Y, conditioning_set).
        alpha (float): Significance level.
        background_knowledge (np.ndarray | None): Adjacency matrix where
            ``bk[i,j] == -1`` and ``bk[j,i] == 1`` encodes the required
            directed edge i → j.  If ``None``, the function behaves
            identically to :func:`pc`.
        **kwargs: Additional keyword arguments are silently ignored.

    Returns:
        dict with key ``"amat"``: MPDAG adjacency matrix as a nested list
            (same convention as :func:`pc`).
    """
    n = data.shape[1]
    skeleton = np.full((n, n), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    order = 0
    while np.amax(np.sum(skeleton != 0, axis=1)) > order:
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set)
        order += 1

    pdag = v_struc_pc(skeleton, sep_set)

    # Standard three-rule Meek (PC step)
    cpdag = _meek(pdag)

    # Inject required background-knowledge edges
    if background_knowledge is not None:
        mask = background_knowledge != 0
        cpdag[mask] = background_knowledge[mask]

    # Four-rule Meek to propagate BK-induced orientations
    cpdag = _meek4(cpdag, sep_set)

    amat = np.zeros((n, n), dtype=int)
    amat[cpdag == -1] = 1
    return {"amat": amat.tolist()}
