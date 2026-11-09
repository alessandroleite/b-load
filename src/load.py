"""
load.py — LOAD and b-LOAD causal effect estimation via local adjustment sets.

This module contains two groups of functions:

1. **LOAD (base)** — adapted from Shubert et al.'s implementation of the
   Local Optimal Adjustment-set Discovery (LOAD) algorithm:
       https://github.com/[shubert-et-al-repo]
   Functions: ``Neighbors``, ``get_neighbors``, ``is_explicit_ancestor``,
   ``is_possible_ancestor``, ``is_amenable``, ``get_locally_valid_parent_sets``,
   ``load``.

2. **b-LOAD** — extension of LOAD that incorporates background knowledge so
   that the graph used for adjustment-set identification is an MPDAG consistent
   with prior causal constraints.  Part of the b-LOAD algorithm introduced in
   this work:
   ``load_in_mpdag``.

Graph encoding (shared throughout this module)
----------------------------------------------
``g[i, j] == -1``  →  tail at i  (i — j or i → j)
``g[i, j] ==  1``  →  head at i  (j → i)
``g[i, j] ==  0``  →  no edge

An undirected edge i — j satisfies  g[i,j] == g[j,i] == -1.
A directed edge  i → j satisfies    g[i,j] == -1  and  g[j,i] == 1.
"""

from collections import defaultdict
from itertools import chain, combinations
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

from src.mb_by_mb import mb_by_mb_alg, mb_by_mb_in_mpdag_alg


# ---------------------------------------------------------------------------
# LOAD (base) — adapted from Shubert et al.
# ---------------------------------------------------------------------------


class Neighbors(NamedTuple):
    """
    Typed container for the three classes of neighbours a node can have in a
    partially directed graph (PDAG / MPDAG).

    .. note::
        Part of the **LOAD** base implementation (Shubert et al.).

    Attributes:
        parents (set[int]): Nodes with a directed edge *into* the target
            (parent → target).
        children (set[int]): Nodes with a directed edge *out of* the target
            (target → child).
        unoriented (set[int]): Nodes connected by an undirected edge
            (target — neighbour).
    """

    parents: set[int]
    children: set[int]
    unoriented: set[int]


def get_neighbors(g: np.ndarray, x: int) -> Neighbors:
    """
    Extract the parents, children, and undirected neighbours of node ``x``
    from the local PDAG / MPDAG ``g``.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        g (np.ndarray): Local graph adjacency matrix (full node-index space).
        x (int): Index of the node whose neighbourhood is to be retrieved.

    Returns:
        Neighbors: Named tuple with three sets — ``parents``, ``children``,
            and ``unoriented``.
    """
    parents = set(np.where((g[x] == 1) & (g[:, x] == -1))[0])
    children = set(np.where((g[x] == -1) & (g[:, x] == 1))[0])
    unoriented = set(np.where((g[x] == -1) & (g[:, x] == -1))[0])
    return Neighbors(parents, children, unoriented)


def is_explicit_ancestor(
    x: int,
    y: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Test whether ``x`` is an *explicit* ancestor of ``y`` in the local graph
    ``g``.

    A node ``x`` is an explicit ancestor of ``y`` if:

    * ``y`` is a *child* of ``x`` in ``g`` (direct directed edge x → y), or
    * ``y`` is not adjacent to ``x`` in ``g`` AND the CI test rejects
      independence of ``x`` and ``y`` given all parents and undirected
      neighbours of ``x`` (i.e., the parents/unoriented set is not a
      d-separating set, suggesting a directed path from x to y).

    If ``y`` is a parent or undirected neighbour of ``x``, the function
    returns ``False`` immediately (no directed path from x to y is implied).

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        x (int): Candidate ancestor node.
        y (int): Candidate descendant node.
        g (np.ndarray): Local PDAG / MPDAG around ``x``.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test returning
            a p-value given (X, Y, conditioning_set).
        alpha (float): Significance level.

    Returns:
        bool: ``True`` if ``x`` is identified as an explicit ancestor of ``y``.
    """
    neighbors = get_neighbors(g, x)
    if y in neighbors.children:
        return True
    elif y in neighbors.parents | neighbors.unoriented:
        return False
    return ci_test(x, y, neighbors.parents | neighbors.unoriented) < alpha


def is_possible_ancestor(
    x: int,
    y: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Test whether ``x`` is a *possible* ancestor of ``y`` in the local graph
    ``g``.

    A node ``x`` is a possible ancestor of ``y`` if:

    * ``y`` is a child or undirected neighbour of ``x`` (a directed path
      x → y is possible), or
    * ``y`` is not adjacent to ``x`` AND the CI test rejects independence of
      ``x`` and ``y`` given only the *parents* of ``x`` (the more conservative
      conditioning set, compared to :func:`is_explicit_ancestor`).

    If ``y`` is a parent of ``x``, the function returns ``False`` (``x``
    cannot be an ancestor of its own parent in a DAG).

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        x (int): Candidate ancestor node.
        y (int): Candidate descendant node.
        g (np.ndarray): Local PDAG / MPDAG around ``x``.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test returning
            a p-value.
        alpha (float): Significance level.

    Returns:
        bool: ``True`` if ``x`` cannot be ruled out as an ancestor of ``y``.
    """
    neighbors = get_neighbors(g, x)
    if y in neighbors.children | neighbors.unoriented:
        return True
    elif y in neighbors.parents:
        return False
    return ci_test(x, y, neighbors.parents) < alpha


def is_amenable(
    x: int,
    y: int,
    v: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Test the *local amenability* condition for an undirected neighbour ``v``
    of the treatment node ``x`` with respect to outcome ``y``.

    A graph is amenable for the adjustment formula (x → y) if no undirected
    neighbour of ``x`` has an undirected or directed path to ``y`` that does
    not pass through ``x``.  This function checks this condition locally for a
    single candidate ``v``:

    * If ``y`` is adjacent to ``v`` in ``g[v]``, the condition fails (``v``
      has a direct path to ``y`` not through ``x``).
    * Otherwise, the CI test checks whether ``v`` and ``y`` are independent
      given parents(v) ∪ {x}.  If they are independent (p-value ≥ alpha),
      ``v`` does not contradict amenability.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        x (int): Treatment node index.
        y (int): Outcome node index.
        v (int): An undirected neighbour of ``x`` to be checked.
        g (np.ndarray): Local PDAG / MPDAG around ``v``.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test returning
            a p-value.
        alpha (float): Significance level.

    Returns:
        bool: ``True`` if ``v`` does *not* violate the amenability condition
            (i.e., ``v`` is "safe" for adjustment through ``x``).
    """
    neighbors = get_neighbors(g, v)
    if y in neighbors.children | neighbors.unoriented | neighbors.parents:
        return False
    return ci_test(v, y, neighbors.parents | {x}) >= alpha


def get_locally_valid_parent_sets(g: np.ndarray, x: int) -> Sequence[set[int]]:
    """
    Enumerate all locally valid parent sets of node ``x`` in the local PDAG
    ``g``, as required by the local IDA algorithm.

    A parent set is *locally valid* if it does not introduce a new v-structure:
    every undirected neighbour promoted to a parent must be adjacent to every
    other candidate parent.

    Implements Algorithm 3 from:
        Maathuis, Kalisch & Bühlmann (2009). "Estimating high-dimensional
        intervention effects from observational data." *Annals of Statistics*.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        g (np.ndarray): Local PDAG / MPDAG containing node ``x``.
        x (int): Index of the node whose valid parent sets are enumerated.

    Returns:
        Sequence[set[int]]: List of valid parent sets (each a set of indices).
    """
    neighbors = get_neighbors(g, x)
    skeleton = g != 0
    np.fill_diagonal(skeleton, True)

    valid_sets = []
    for new_parents in chain.from_iterable(
        combinations(neighbors.unoriented, r)
        for r in range(len(neighbors.unoriented) + 1)
    ):
        candidate_set = neighbors.parents.union(new_parents)
        if np.all(skeleton[new_parents, :][:, list(candidate_set)]):
            valid_sets.append(candidate_set)
    return valid_sets


def load(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    targets: Sequence[int],
    mb_algorithm: str = "grow_shrink",
    ignore: Sequence[int] = [],
    logging: bool = False,
    **kwargs,
) -> dict:
    """
    LOAD: Local Optimal Adjustment-set Discovery for a pair of target nodes.

    Estimates the optimal adjustment set (o-set) for the causal effect of a
    treatment on an outcome using only *local* CI tests and MB-by-MB structure
    learning, without reconstructing the full global causal graph.

    Pipeline:

    1. Learn local MDAGs around both targets using :func:`~src.mb_by_mb.mb_by_mb_alg`.
    2. Determine the causal direction (treatment → outcome) via
       :func:`is_explicit_ancestor`.  If direction is ambiguous, fall back to
       :func:`is_possible_ancestor` and return all locally valid parent sets
       with ``identifiable=False``.
    3. Verify *amenability*: check that no undirected neighbour of the
       treatment has an undirected path to the outcome outside the treatment.
    4. Identify explicit mediators between treatment and outcome.
    5. Compute the o-set as the union of parent sets of all mediators and the
       outcome, minus mediators and the treatment itself.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        data (np.ndarray): Data matrix of shape (n_samples, n_nodes).
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test returning
            a p-value given (X, Y, conditioning_set).
        alpha (float): Significance level for CI tests.
        targets (Sequence[int]): Pair of node indices ``[t1, t2]`` whose causal
            effect is to be estimated.
        mb_algorithm (str): MB discovery algorithm — ``"grow_shrink"``
            (default) or ``"total_conditioning"``.
        ignore (Sequence[int]): Node indices to exclude from MB searches.
        logging (bool): If ``True``, print verbose progress information.
        **kwargs: Additional keyword arguments are silently ignored (for
            interface compatibility with other algorithms).

    Returns:
        dict with keys:

        * ``"adj_sets"`` (str): String representation of a dict mapping
          ``(treatment, outcome)`` to a list of adjustment sets.
        * ``"identifiable"`` (bool): ``True`` if a unique o-set was found.
        * ``"id_tests"`` (list): CI test counts per conditioning-set order.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)
    adj_sets = dict()

    t1, t2 = targets

    if logging:
        print("========================")
        print(f"Running mb-by-mb for {t1}")
    G[t1] = mb_by_mb_alg(
        data, ci_test, alpha, t1, mb_algorithm, MB, L, sep_set, ignore=ignore
    )

    if logging:
        print("========================")
        print(f"Running mb-by-mb for {t2}")
    G[t2] = mb_by_mb_alg(
        data, ci_test, alpha, t2, mb_algorithm, MB, L, sep_set, ignore=ignore
    )

    # Determine causal direction
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        treatment, outcome = t1, t2
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        treatment, outcome = t2, t1
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
        return {
            "adj_sets": str(adj_sets),
            "identifiable": False,
            "id_tests": ci_test.get_tests_per_order().tolist(),
        }

    # Check amenability of undirected neighbours of treatment
    unoriented = get_neighbors(G[treatment], treatment).unoriented
    for v in unoriented:
        G[v] = mb_by_mb_alg(data, ci_test, alpha, v, mb_algorithm, MB, L, sep_set)
        if not is_amenable(treatment, outcome, v, G[v], ci_test, alpha):
            adj_sets[(treatment, outcome)] = get_locally_valid_parent_sets(
                G[treatment], treatment
            )
            return {
                "adj_sets": str(adj_sets),
                "identifiable": False,
                "id_tests": ci_test.get_tests_per_order().tolist(),
            }
    id_tests = ci_test.get_tests_per_order().tolist()

    # Identify explicit descendants of treatment (candidates for mediators)
    desc = set()
    for v in set(range(data.shape[1])) - {treatment, outcome}:
        if is_explicit_ancestor(treatment, v, G[treatment], ci_test, alpha):
            if logging:
                print(f"Adding {v} to descendants of treatment")
            desc.add(v)
            G[v] = mb_by_mb_alg(data, ci_test, alpha, v, mb_algorithm, MB, L, sep_set)

    # Identify explicit mediators (descendants that are ancestors of the outcome)
    meds = set()
    for v in desc:
        if is_explicit_ancestor(v, outcome, G[v], ci_test, alpha):
            meds.add(v)
            if logging:
                print(f"Adding {v} to mediators")

    # Compute the optimal adjustment set (o-set)
    oset = set()
    for med in meds | {outcome}:
        oset |= get_neighbors(G[med], med).parents
        if logging:
            print(f"Parents of {med}: {get_neighbors(G[med], med).parents}")

    oset -= meds | {treatment}

    if logging:
        print(f"Final o-set: {oset}")

    adj_sets[(treatment, outcome)] = [oset]
    return {"adj_sets": str(adj_sets), "identifiable": True, "id_tests": id_tests}


# ---------------------------------------------------------------------------
# b-LOAD — background-knowledge extension (part of the b-LOAD algorithm)
# ---------------------------------------------------------------------------


def load_in_mpdag(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    targets: Sequence[int],
    background_knowledge: Optional[np.ndarray] = None,
    mb_algorithm: str = "grow_shrink",
    ignore: Sequence[int] = [],
    logging: bool = False,
    **kwargs,
) -> dict:
    """
    b-LOAD: Background-knowledge-augmented LOAD for causal effect estimation
    in MPDAGs.

    This is the **b-LOAD** counterpart of :func:`load`.  The pipeline is
    identical, with one key difference: all calls to the MB-by-MB structure
    learner use :func:`~src.mb_by_mb.mb_by_mb_in_mpdag_alg` instead of
    :func:`~src.mb_by_mb.mb_by_mb_alg`.  That variant:

    * Seeds the growing graph ``g`` with the provided ``background_knowledge``
      so that known edges are present from the first iteration.
    * Re-enforces required edges after every local structure merge
      (preventing CI-test errors from erasing prior constraints).
    * Uses the four-rule MPDAG Meek procedure to propagate orientations
      introduced by background-knowledge edges.

    Everything downstream (ancestor tests, amenability checks, o-set
    computation) is identical to :func:`load` and operates on the resulting
    MPDAG.

    .. note::
        Part of the **b-LOAD** algorithm introduced in this work.

    Args:
        data (np.ndarray): Data matrix of shape (n_samples, n_nodes).
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test returning
            a p-value given (X, Y, conditioning_set).
        alpha (float): Significance level for CI tests.
        targets (Sequence[int]): Pair of node indices ``[t1, t2]`` whose causal
            effect is to be estimated.
        background_knowledge (np.Optional[ndarray]): Prior edge constraints in the
            standard adjacency format (``bk[i,j] == -1`` and ``bk[j,i] == 1``
            ↔ required edge i→j).  If ``None``, the function behaves
            identically to :func:`load`.
        mb_algorithm (str): MB discovery algorithm — ``"grow_shrink"``
            (default) or ``"total_conditioning"``.
        ignore (Sequence[int]): Node indices excluded from all MB searches.
        logging (bool): If ``True``, print verbose progress information.
        **kwargs: Additional keyword arguments are silently ignored.

    Returns:
        dict with keys:

        * ``"adj_sets"`` (str): String representation of a dict mapping
          ``(treatment, outcome)`` to a list of adjustment sets.
        * ``"identifiable"`` (bool): ``True`` if a unique o-set was found.
        * ``"id_tests"`` (list): CI test counts per conditioning-set order.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)
    adj_sets = dict()

    t1, t2 = targets

    if logging:
        print("========================")
        print(f"Running b-LOAD mb-by-mb for {t1}")
    G[t1] = mb_by_mb_in_mpdag_alg(
        data, ci_test, alpha, t1, background_knowledge, mb_algorithm, MB, L, sep_set,
        ignore=ignore,
    )

    if logging:
        print("========================")
        print(f"Running b-LOAD mb-by-mb for {t2}")
    G[t2] = mb_by_mb_in_mpdag_alg(
        data, ci_test, alpha, t2, background_knowledge, mb_algorithm, MB, L, sep_set,
        ignore=ignore,
    )

    # Determine causal direction
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        treatment, outcome = t1, t2
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        treatment, outcome = t2, t1
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
        return {
            "adj_sets": str(adj_sets),
            "identifiable": False,
            "id_tests": ci_test.get_tests_per_order().tolist(),
        }

    # Check amenability of undirected neighbours of treatment
    unoriented = get_neighbors(G[treatment], treatment).unoriented
    for v in unoriented:
        G[v] = mb_by_mb_in_mpdag_alg(
            data, ci_test, alpha, v, background_knowledge, mb_algorithm, MB, L, sep_set
        )
        if not is_amenable(treatment, outcome, v, G[v], ci_test, alpha):
            adj_sets[(treatment, outcome)] = get_locally_valid_parent_sets(
                G[treatment], treatment
            )
            return {
                "adj_sets": str(adj_sets),
                "identifiable": False,
                "id_tests": ci_test.get_tests_per_order().tolist(),
            }
    id_tests = ci_test.get_tests_per_order().tolist()

    # Identify explicit descendants of treatment (candidates for mediators)
    desc = set()
    for v in set(range(data.shape[1])) - {treatment, outcome}:
        if is_explicit_ancestor(treatment, v, G[treatment], ci_test, alpha):
            if logging:
                print(f"Adding {v} to descendants of treatment")
            desc.add(v)
            G[v] = mb_by_mb_in_mpdag_alg(
                data, ci_test, alpha, v, background_knowledge, mb_algorithm, MB, L,
                sep_set,
            )

    # Identify explicit mediators (descendants that are ancestors of the outcome)
    meds = set()
    for v in desc:
        if is_explicit_ancestor(v, outcome, G[v], ci_test, alpha):
            meds.add(v)
            if logging:
                print(f"Adding {v} to mediators")

    # Compute the optimal adjustment set (o-set)
    oset = set()
    for med in meds | {outcome}:
        oset |= get_neighbors(G[med], med).parents
        if logging:
            print(f"Parents of {med}: {get_neighbors(G[med], med).parents}")

    oset -= meds | {treatment}

    if logging:
        print(f"Final o-set: {oset}")

    adj_sets[(treatment, outcome)] = [oset]
    return {"adj_sets": str(adj_sets), "identifiable": True, "id_tests": id_tests}
