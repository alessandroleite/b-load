"""
mb_by_mb.py — MB-by-MB local causal discovery, base LOAD and b-LOAD variants.

This module contains two groups of functions:

1. **LOAD (base)** — adapted from Shubert et al.'s public implementation of the
   MB-by-MB algorithm for local causal structure learning:
       https://github.com/[shubert-et-al-repo]
   Functions: ``find_unshielded_triples``, ``v_struc_pc``, ``skeleton_step``,
   ``get_locally_valid_parent_sets``, ``grow_shrink``, ``total_conditioning``,
   ``learn_local_structure``, ``copy_local_structure``, ``update_graph``,
   ``meek``, ``reach_with_undirected``, ``mb_by_mb_alg``.

2. **b-LOAD** — extensions of the above that incorporate background knowledge
   into the MB-by-MB loop so that the resulting graph is an MPDAG consistent
   with the provided prior edge constraints.  These functions are part of the
   b-LOAD algorithm introduced in this work:
   ``update_graph_mpdag``, ``meek_mpdag``, ``mb_by_mb_in_mpdag_alg``.

Graph encoding (shared throughout this module)
----------------------------------------------
``g[i, j] == -1``  →  tail of an edge at i  (i — j or i → j)
``g[i, j] ==  1``  →  head of an edge at i  (j — i or j → i)
``g[i, j] ==  0``  →  no edge between i and j

An undirected edge i — j satisfies  g[i,j] == g[j,i] == -1.
A directed edge  i → j satisfies    g[i,j] == -1  and  g[j,i] == 1.
"""

from collections import defaultdict
from itertools import chain, combinations
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# LOAD (base) — adapted from Shubert et al.
# ---------------------------------------------------------------------------


def find_unshielded_triples(g: np.ndarray) -> np.ndarray:
    """
    Find all unshielded triples (i, j, k) in the skeleton in a vectorised way.

    An unshielded triple is a path i — j — k where i and k are *not* adjacent.
    These are the candidates for v-structure (collider) orientation.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        g (np.ndarray): Skeleton adjacency matrix  (non-zero ↔ edge present).

    Returns:
        np.ndarray: Array of shape (M, 3) whose rows are (i, j, k) triples with
            i < k (each unshielded triple appears exactly once).
    """
    i, j = np.where(g != 0)  # i - j
    k = np.where((g[j] != 0) & (g[i] == 0))  # j - k -/- i
    i, j, k = i[k[0]], j[k[0]], k[1]
    mask = i < k
    return np.column_stack((i[mask], j[mask], k[mask]))


def v_struc_pc(
    skeleton: np.ndarray, sep_set: dict[frozenset, list], bidirected: bool = False
) -> np.ndarray:
    """
    Orient v-structures (unshielded colliders) in the skeleton following the
    PC-algorithm criterion: for each unshielded triple i — j — k, orient
    i → j ← k if and only if j is *not* in any separating set of i and k.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        skeleton (np.ndarray): Undirected skeleton adjacency matrix.
        sep_set (dict[frozenset, list]): Separating sets recorded during the
            skeleton search; ``sep_set[frozenset({i, k})]`` is a list of
            conditioning sets that d-separated i and k.
        bidirected (bool): If ``True``, conflicting orientations (two colliders
            sharing an edge) are left as bidirected rather than overwritten.
            Default is ``False`` (last-writer wins).

    Returns:
        np.ndarray: PDAG with v-structures oriented; all remaining edges are
            still encoded as undirected (g[i,j] == g[j,i] == -1).
    """
    pdag = skeleton.copy()
    for x, y, z in find_unshielded_triples(skeleton):
        if all(y not in S for S in sep_set[frozenset((x, z))]):
            pdag[y, x] = pdag[y, z] = 1  # arrowhead into y
            if not bidirected:
                pdag[x, y] = pdag[z, y] = -1  # tails out of x and z
    return pdag


def skeleton_step(
    order: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    sep_set: dict[frozenset, list],
) -> np.ndarray:
    """
    Perform one round of skeleton pruning at a given conditioning-set size.

    For every adjacent pair (x, y) in ``g``, iterate over all subsets of
    size ``order`` drawn from x's current neighbours (excluding y) and remove
    the edge x — y as soon as a d-separating set is found.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        order (int): Size of the conditioning sets to test (0, 1, 2, …).
        g (np.ndarray): Current skeleton (modified **in place** and returned).
        ci_test (Callable[[int, int, list[int]], float]): Conditional
            independence test; returns a p-value given (X, Y, conditioning_set).
        alpha (float): Significance level — edges whose p-value ≥ alpha are
            removed from the skeleton.
        sep_set (dict[frozenset, list]): Accumulates the separating sets found;
            updated in place when an edge is removed.

    Returns:
        np.ndarray: Updated skeleton (same object as the input ``g``).
    """
    for x in range(len(g)):
        Neigh_x = np.where(g[x, :] != 0)[0]
        if len(Neigh_x) < order - 1:
            continue
        for y in Neigh_x:
            curr_neigh = np.where(g[x, :] != 0)[0]
            Neigh_x_noy = np.delete(curr_neigh, np.where(curr_neigh == y))
            for S in combinations(Neigh_x_noy, order):
                if ci_test(x, y, S) >= alpha:
                    g[x, y] = g[y, x] = 0
                    sep_set[frozenset((x, y))].append(S)
                    break
    return g


def get_locally_valid_parent_sets(g: np.ndarray, x: int) -> Sequence[set[int]]:
    """
    Enumerate all locally valid parent sets of node ``x`` in the local CPDAG ``g``.

    A parent set is *locally valid* if it does not introduce a new v-structure
    relative to the skeleton — i.e., every undirected neighbour promoted to a
    parent must itself be adjacent to every other candidate parent.

    This implements Algorithm 3 from:
        Maathuis, Kalisch & Bühlmann (2009). "Estimating high-dimensional
        intervention effects from observational data." *Annals of Statistics*.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        g (np.ndarray): Local graph (adjacency matrix) containing node ``x``
            and its neighbourhood.
        x (int): Index of the target node within ``g``.

    Returns:
        Sequence[set[int]]: List of valid parent sets (each a set of node
            indices).  Every combination of undirected neighbours of ``x``
            that can be oriented as parents without creating a new collider is
            included.
    """
    parents = set(np.where((g[x] == 1) & (g[:, x] == -1))[0])
    unoriented = set(np.where((g[x] == -1) & (g[:, x] == -1))[0])
    skeleton = g != 0
    np.fill_diagonal(skeleton, True)

    valid_sets = []
    for new_parents in chain.from_iterable(
        combinations(unoriented, r) for r in range(len(unoriented) + 1)
    ):
        candidate_set = parents.union(new_parents)
        if np.all(skeleton[new_parents, :][:, list(candidate_set)]):
            valid_sets.append(candidate_set)
    return valid_sets


def grow_shrink(
    nodes: set,
    target: int,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
) -> set:
    """
    Estimate the Markov blanket of ``target`` using the Grow-Shrink (GS)
    algorithm.

    The GS algorithm operates in two passes:

    * **Grow**: iteratively add nodes that are conditionally dependent on
      ``target`` given the current blanket.
    * **Shrink**: remove nodes that become conditionally independent of
      ``target`` given the full blanket.

    Adapted from: https://github.com/acmi-lab/local-causal-discovery

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        nodes (set): Full node set to search over.
        target (int): Node whose Markov blanket is to be estimated.
        ci_test (Callable[[int, int, list[int]], float]): CI test returning a
            p-value given (X, Y, conditioning_set).
        alpha (float): Significance level for CI tests.

    Returns:
        set: Estimated Markov blanket of ``target`` (does not include
            ``target`` itself).
    """
    mb = set()

    # Grow phase
    cont = True
    while cont:
        cont = False
        mb_copy = set(mb)
        nodes_to_check = nodes - {target} - mb_copy
        for n in nodes_to_check:
            if ci_test(n, target, mb - {n}) < alpha:
                mb.add(n)
                cont = True

    # Shrink phase
    mb_copy = set(mb)
    for n in mb_copy:
        if ci_test(n, target, mb - {n}) >= alpha:
            mb.remove(n)

    return mb


def total_conditioning(
    nodes: set,
    target: int,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
) -> set:
    """
    Estimate the Markov blanket of ``target`` using the Total Conditioning (TC)
    algorithm.

    Every node in ``nodes`` is tested for independence of ``target`` given
    *all other* nodes simultaneously.  Nodes that remain dependent are
    included in the blanket.

    Reference:
        Pellet & Elisseeff (2008). "Using Markov blankets for causal structure
        learning." *Journal of Machine Learning Research*.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        nodes (set): Full node set to search over.
        target (int): Node whose Markov blanket is to be estimated.
        ci_test (Callable[[int, int, list[int]], float]): CI test returning a
            p-value given (X, Y, conditioning_set).
        alpha (float): Significance level for CI tests.

    Returns:
        set: Estimated Markov blanket of ``target`` (does not include
            ``target`` itself).
    """
    mb = set()
    for n in nodes - {target}:
        if ci_test(n, target, nodes - {target, n}) < alpha:
            mb.add(n)
    return mb


def learn_local_structure(
    nodes: set,
    observed: set,
    sep_set: dict[frozenset, list],
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
) -> np.ndarray:
    """
    Learn the local PDAG over ``observed`` nodes using a PC-style skeleton
    search followed by v-structure orientation.

    Only nodes in ``observed`` participate in the skeleton search; all other
    nodes in ``nodes`` are treated as absent (their rows/columns are zeroed
    out at initialisation).  The resulting PDAG is bidirected-safe: when two
    v-structure orientations conflict on the same edge, the edge is left
    bidirected rather than overwritten.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        nodes (set): Complete variable set (used for index space).
        observed (set): Subset of nodes over which to learn the structure
            (typically mb⁺(x) for some target x).
        sep_set (dict[frozenset, list]): Separating sets — updated in place.
        ci_test (Callable[[int, int, list[int]], float]): CI test returning a
            p-value.
        alpha (float): Significance level.

    Returns:
        np.ndarray: PDAG adjacency matrix of size ``len(nodes) × len(nodes)``.
            Only entries corresponding to ``observed`` nodes are meaningful;
            all others are zero.
    """
    g = np.full((len(nodes), len(nodes)), -1, dtype=int)
    np.fill_diagonal(g, 0)
    ignore = list(nodes.difference(observed))
    g[ignore, :] = g[:, ignore] = 0

    order = 0
    while np.amax(np.sum(g != 0, axis=1)) > order:
        g = skeleton_step(order, g, ci_test, alpha, sep_set)
        order += 1

    g = v_struc_pc(g, sep_set, bidirected=True)
    return g


def copy_local_structure(g: np.ndarray, observed: set) -> np.ndarray:
    """
    Extract the sub-graph of ``g`` induced by ``observed`` into a zero-padded
    array of the same shape.

    Used in the MB-by-MB algorithm when the local structure of a new node can
    be derived from an already-computed local structure (because its
    neighbourhood is a subset of one that was already fully learned).

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        g (np.ndarray): Source graph (full node-index space).
        observed (set): Node indices whose mutual edges should be copied.

    Returns:
        np.ndarray: Array of the same shape as ``g`` with only the
            ``observed × observed`` block copied; all other entries are zero.
    """
    local = np.zeros_like(g, dtype=int)
    observed = list(observed)
    local[np.ix_(observed, observed)] = g[np.ix_(observed, observed)]
    return local


def update_graph(g: np.ndarray, target: int, local: np.ndarray) -> np.ndarray:
    """
    Merge the local structure ``local`` for ``target`` into the global graph
    ``g``.

    Two classes of information are written into ``g``:

    1. **All edges incident on ``target``**: the entire row and column of
       ``target`` in ``g`` are replaced by those from ``local``.
    2. **V-structures where ``target`` is a parent**: for each collider
       ``coll`` into which ``target`` points, the edge ``spouse → coll``
       (for every other parent ``spouse`` of ``coll`` in ``local``) is also
       propagated.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        g (np.ndarray): Global graph to update (modified in place).
        target (int): Node whose local neighbourhood is being merged.
        local (np.ndarray): Local PDAG over ``mb⁺(target)`` (same index
            space as ``g``).

    Returns:
        np.ndarray: Updated ``g`` (same object).
    """
    g[target, :] = local[target, :]
    g[:, target] = local[:, target]

    coll = np.where(local.T[target] == 1)[0]  # target *-> coll
    spouse = np.where((local[coll] == 1) & (local[target] == 0))
    coll, spouse = coll[spouse[0]], spouse[1]
    mask = spouse != target
    coll, spouse = coll[mask], spouse[mask]
    g[target, coll] = local[target, coll]
    g[coll, target] = local[coll, target]
    g[spouse, coll] = local[spouse, coll]
    g[coll, spouse] = local[coll, spouse]

    return g


def meek(g: np.ndarray, sep_set: dict[frozenset, list]) -> np.ndarray:
    """
    Propagate edge orientations in ``g`` using a sep-set-aware revision of
    Meek's rules, as specified in the MB-by-MB algorithm.

    Three rules are applied exhaustively (until no further change occurs):

    * **R1** — ``a → b — c`` and b ∈ Sep(a, c)  ⟹  orient ``b → c``
      (b is a non-collider between a and c, so it cannot point into c).
    * **R2** — ``a → b → c — a``  ⟹  orient ``a → c``
      (acyclicity: orienting ``c → a`` would create a directed cycle).
    * **R3** — ``a — c — b``, ``a — d — b``, c ↛ d, a ∈ Sep(c, d),
      ``c → b``, ``d → b``  ⟹  orient ``a → b``
      (unique non-collider path).

    Unlike the standard Meek rules (which operate on graph structure alone),
    R1 and R3 here additionally consult the ``sep_set`` recorded during the
    skeleton search.  This is the variant used in the original LOAD /
    MB-by-MB paper.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        g (np.ndarray): Partially directed graph to orient (copied
            internally; the input is not mutated).
        sep_set (dict[frozenset, list]): Separating sets from the skeleton
            search; ``sep_set[frozenset({i, k})]`` is the list of
            conditioning sets that d-separated i and k.

    Returns:
        np.ndarray: Graph with all propagatable orientations resolved.
    """

    def rule1(g: np.ndarray, sep_set: dict[frozenset, list]) -> bool:
        """
        R1: a → b — c, b ∈ Sep(a, c)  ⟹  b → c.
        b being in the sep-set certifies it is a non-collider, so it must
        have a tail toward c.
        """
        a, b = np.where((g == -1) & (g.T == 1))  # a -> b
        c = np.where(((g[b] == -1) & (g.T[b] == -1)) & (g[a] == 0))  # b - c, a -/- c
        a, b, c = a[c[0]], b[c[0]], c[1]
        changed = False
        for a_, b_, c_ in zip(a, b, c):
            if frozenset((a_, c_)) in sep_set and all(
                b_ in S for S in sep_set[frozenset((a_, c_))]
            ):
                g[c_, b_] = 1  # b -> c
                changed = True
        return changed

    def rule2(g: np.ndarray) -> bool:
        """
        R2: a → b → c — a  ⟹  a → c.
        Orienting c → a would close a directed cycle a → b → c → a.
        """
        a, b = np.where((g == -1) & (g.T == 1))  # a -> b
        c = np.where(
            ((g[b] == -1) & (g.T[b] == 1))    # b -> c
            & ((g[a] == -1) & (g.T[a] == -1)) # a - c
        )
        a, b, c = a[c[0]], b[c[0]], c[1]
        g[c, a] = 1  # a -> c
        return len(a) > 0

    def rule3(g: np.ndarray, sep_set: dict[frozenset, list]) -> bool:
        """
        R3: a — c, a — d, c -/- d, a ∈ Sep(c, d), c → b, d → b, a — b
            ⟹  a → b.
        a being in Sep(c,d) means it is a non-collider between c and d,
        so the only consistent orientation of a — b is a → b.
        """
        a, c = np.where((g == -1) & (g.T == -1))  # a - c
        d = np.where(((g[a] == -1) & (g.T[a] == -1)) & (g[c] == 0))  # a - d, c -/- d
        a, c, d = a[d[0]], c[d[0]], d[1]
        mask = c < d
        a, c, d = a[mask], c[mask], d[mask]
        b = np.where(
            ((g[a] == -1) & (g.T[a] == -1))   # a - b
            & ((g[c] == -1) & (g.T[c] == 1))  # c -> b
            & ((g[d] == -1) & (g.T[d] == 1))  # d -> b
        )
        a, c, d, b = a[b[0]], c[b[0]], d[b[0]], b[1]
        changed = False
        for a_, b_, c_, d_ in zip(a, b, c, d):
            if frozenset((c_, d_)) in sep_set and all(
                a_ in S for S in sep_set[frozenset((c_, d_))]
            ):
                g[b_, a_] = 1  # a -> b
                changed = True
        return changed

    g = g.copy()
    while rule1(g, sep_set) or rule2(g) or rule3(g, sep_set):
        continue
    return g


def reach_with_undirected(g: np.ndarray, target: int) -> np.ndarray:
    """
    Return all nodes that are connected to ``target`` via a path consisting
    entirely of **undirected** edges.

    This is used in the MB-by-MB loop to prune the wait-list: once a node
    can only reach ``target`` through a chain of directed edges, its local
    structure cannot change the orientation of edges incident on ``target``,
    so it is safe to drop from further processing.

    Implementation: the undirected adjacency matrix is raised to the
    (n − 1)-th power; any non-zero entry in ``target``'s row indicates
    reachability.

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        g (np.ndarray): Partially directed graph.
        target (int): The node to measure reachability toward.

    Returns:
        np.ndarray: 1-D array of node indices that can reach ``target``
            via undirected edges (``target`` itself is excluded).
    """
    undir = (g == -1) & (g.T == -1)
    np.fill_diagonal(undir, 1)
    reach = np.linalg.matrix_power(undir, undir.shape[0] - 1)
    mask = reach[target] != 0
    mask[target] = False
    return np.where(mask)[0]


def mb_by_mb_alg(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    target: int,
    mb_algorithm: str = "grow_shrink",
    mb: dict[int, set] | None = None,
    L: dict[int, np.ndarray] | None = None,
    sep_set: dict[frozenset, list] | None = None,
    ignore: Sequence[int] = [],
) -> np.ndarray:
    """
    MB-by-MB algorithm for learning the local causal network around ``target``.

    The algorithm expands outward from ``target`` one Markov blanket at a time,
    learning a local PDAG for each node encountered, merging it into a growing
    global graph ``g``, propagating orientations with Meek rules, and pruning
    the work-list of nodes whose connection to ``target`` is already fully
    directed.

    High-level steps (repeated until the wait-list is empty):

    1. Pop node ``x`` from the wait-list.
    2. Compute mb(x) with the chosen MB algorithm.
    3. Learn ``L[x]``, the local PDAG over mb⁺(x), reusing previously
       computed structures where possible.
    4. Merge ``L[x]`` into ``g`` via :func:`update_graph`.
    5. Propagate orientations with :func:`meek`.
    6. Prune the wait-list to nodes still reachable from ``target`` via
       undirected paths (:func:`reach_with_undirected`).

    .. note::
        Adapted from Shubert et al.'s LOAD implementation.

    Args:
        data (np.ndarray): Data matrix of shape (n_samples, n_nodes).
        ci_test (Callable[[int, int, list[int]], float]): CI test returning a
            p-value given (X, Y, conditioning_set).
        alpha (float): Significance level for CI tests.
        target (int): Index of the node whose local network is to be learned.
        mb_algorithm (str): MB discovery algorithm — ``"grow_shrink"`` (default)
            or ``"total_conditioning"``.
        mb (dict[int, set] | None): Pre-computed Markov blankets; updated in
            place and reused across calls.
        L (dict[int, np.ndarray] | None): Pre-computed local structures;
            updated in place.
        sep_set (dict[frozenset, list] | None): Pre-computed separating sets;
            updated in place.
        ignore (Sequence[int]): Node indices to exclude from all MB searches
            (e.g. intervention targets).

    Returns:
        np.ndarray: Learned local PDAG ``g`` of shape (n_nodes, n_nodes),
            focused on the neighbourhood of ``target``.
    """
    if mb_algorithm == "grow_shrink":
        mb_algorithm = grow_shrink
    elif mb_algorithm == "total_conditioning":
        mb_algorithm = total_conditioning
    else:
        raise ValueError(f"Unknown mb_algorithm: {mb_algorithm}")

    mb = mb or dict()
    L = L or dict()
    sep_set = sep_set or defaultdict(list)
    V = set(range(data.shape[1]))

    done_list = set()
    wait_list = [target]
    g = np.zeros((len(V), len(V)), dtype=int)

    while len(wait_list) > 0:
        x = wait_list.pop(0)

        if x not in mb:
            mb[x] = mb_algorithm(V.difference(set(ignore)), x, ci_test, alpha)
        mb_x = mb[x].union({x})

        wait_list.extend([v for v in mb[x] if v not in list(done_list) + wait_list])
        done_list.add(x)

        if x in L:
            pass
        elif (
            n := next((n for n in done_list if n != x and mb_x < mb[n] | {n}), None)
        ) is not None:
            L[x] = copy_local_structure(L[n], mb_x)
        elif mb[x] < done_list:
            L[x] = copy_local_structure(g, mb_x)
        else:
            L[x] = learn_local_structure(V, mb_x, sep_set, ci_test, alpha)

        g = update_graph(g, x, L[x])
        g = meek(g, sep_set)

        connected = reach_with_undirected(g, target)
        wait_list = [w for w in wait_list if w in connected]

    return g


# ---------------------------------------------------------------------------
# b-LOAD — background-knowledge extensions (part of the b-LOAD algorithm)
# ---------------------------------------------------------------------------


def update_graph_mpdag(
    g: np.ndarray,
    target: int,
    local: np.ndarray,
    background_knowledge: np.ndarray | None = None,
) -> np.ndarray:
    """
    Merge the local structure ``local`` for ``target`` into the global MPDAG
    ``g``, then re-stamp any required background-knowledge edges that the
    merge may have overwritten.

    This is the **b-LOAD** counterpart of :func:`update_graph`.  The only
    addition is step 3: after writing all local information into ``g``, the
    background-knowledge matrix is applied as a mask so that required edges
    are never erased by statistical CI-test errors.

    The merge itself follows the same two-phase logic as :func:`update_graph`:

    1. Overwrite the row and column of ``target`` in ``g`` with the
       corresponding entries from ``local`` (atomic overwrite — this prevents
       stale-edge cycles within mb⁺(target)).
    2. Propagate v-structures in which ``target`` acts as a parent.
    3. Re-enforce background knowledge: any position where
       ``background_knowledge[i, j] ≠ 0`` is forced back to its prescribed
       value.

    .. note::
        Part of the **b-LOAD** algorithm introduced in this work.

    Args:
        g (np.ndarray): Global MPDAG to update (modified in place).
        target (int): Node whose local neighbourhood is being merged.
        local (np.ndarray): Local PDAG over mb⁺(target) (same index space
            as ``g``).
        background_knowledge (np.ndarray | None): Prior edge constraints
            encoded in the standard adjacency format
            (``bk[i,j] == -1`` and ``bk[j,i] == 1``  ↔  required edge i→j).
            If ``None``, the function behaves identically to
            :func:`update_graph`.

    Returns:
        np.ndarray: Updated ``g`` (same object).
    """
    # 1. Atomic row + column overwrite for target
    g[target, :] = local[target, :]
    g[:, target] = local[:, target]

    # 2. Propagate v-structures where target is a parent
    coll = np.where(local.T[target] == 1)[0]  # target *-> coll
    spouse = np.where((local[coll] == 1) & (local[target] == 0))
    coll, spouse = coll[spouse[0]], spouse[1]
    mask = spouse != target
    coll, spouse = coll[mask], spouse[mask]
    g[target, coll] = local[target, coll]
    g[coll, target] = local[coll, target]
    g[spouse, coll] = local[spouse, coll]
    g[coll, spouse] = local[coll, spouse]

    # 3. Re-enforce background knowledge
    if background_knowledge is not None:
        mask = background_knowledge != 0
        g[mask] = background_knowledge[mask]

    return g


def meek_mpdag(g: np.ndarray, sep_set: dict[frozenset, list]) -> np.ndarray:
    """
    Propagate edge orientations in an MPDAG using a four-rule sep-set-aware
    revision of Meek's rules.

    Rules R1–R3 are identical to those in :func:`meek` (the LOAD base version).
    Rule R4 is an additional orientation rule needed when background knowledge
    has introduced directed edges that cannot be explained by v-structures
    alone:

    * **R4** — ``a — b``, ``a — c`` (undirected), ``b → c``,
      ``a — d`` (undirected), ``d → c``, b -/- d,
      and a ∈ Sep(b, d)  ⟹  orient ``a → b``.
      (a being in Sep(b,d) certifies it is not a collider on b — a — d,
      so a → b is the only orientation that avoids a new unshielded collider.)

    This four-rule variant is required for the b-LOAD / b-PC setting where
    injected background-knowledge edges may create orientations that the
    three structural rules alone cannot propagate.

    .. note::
        Part of the **b-LOAD** algorithm introduced in this work.

    Args:
        g (np.ndarray): Partially directed MPDAG to orient (copied internally;
            the input is not mutated).
        sep_set (dict[frozenset, list]): Separating sets from the skeleton
            search, used by R1, R3, and R4.

    Returns:
        np.ndarray: Graph with all propagatable orientations resolved under
            the four rules.
    """

    def rule1(g, sep_set):
        """R1: a → b — c, b ∈ Sep(a,c)  ⟹  b → c."""
        a, b = np.where((g == -1) & (g.T == 1))
        c = np.where(((g[b] == -1) & (g.T[b] == -1)) & (g[a] == 0))
        a, b, c = a[c[0]], b[c[0]], c[1]
        changed = False
        for a_, b_, c_ in zip(a, b, c):
            if frozenset((a_, c_)) in sep_set and all(
                b_ in S for S in sep_set[frozenset((a_, c_))]
            ):
                g[c_, b_] = 1  # b -> c
                changed = True
        return changed

    def rule2(g):
        """R2: a → b → c — a  ⟹  a → c  (acyclicity)."""
        a, b = np.where((g == -1) & (g.T == 1))
        c = np.where(
            ((g[b] == -1) & (g.T[b] == 1))
            & ((g[a] == -1) & (g.T[a] == -1))
        )
        a, b, c = a[c[0]], b[c[0]], c[1]
        g[c, a] = 1  # a -> c
        return len(a) > 0

    def rule3(g, sep_set):
        """R3: a — c, a — d, c -/- d, a ∈ Sep(c,d), c → b, d → b, a — b  ⟹  a → b."""
        a, c = np.where((g == -1) & (g.T == -1))
        d = np.where(((g[a] == -1) & (g.T[a] == -1)) & (g[c] == 0))
        a, c, d = a[d[0]], c[d[0]], d[1]
        mask = c < d
        a, c, d = a[mask], c[mask], d[mask]
        b = np.where(
            ((g[a] == -1) & (g.T[a] == -1))
            & ((g[c] == -1) & (g.T[c] == 1))
            & ((g[d] == -1) & (g.T[d] == 1))
        )
        a, c, d, b = a[b[0]], c[b[0]], d[b[0]], b[1]
        changed = False
        for a_, b_, c_, d_ in zip(a, b, c, d):
            if frozenset((c_, d_)) in sep_set and all(
                a_ in S for S in sep_set[frozenset((c_, d_))]
            ):
                g[b_, a_] = 1  # a -> b
                changed = True
        return changed

    def rule4(g, sep_set):
        """
        R4: a — b, a — c (undir), b → c, a — d (undir), d → c, b -/- d,
            a ∈ Sep(b,d)  ⟹  a → b.
        This rule handles orientations introduced by background-knowledge
        edges that are not discoverable from graph structure alone.
        """
        a, b = np.where((g == -1) & (g.T == -1))
        changed = False
        for a_, b_ in zip(a, b):
            c_list = np.where(
                ((g[a_] == -1) & (g.T[a_] == -1))   # a - c
                & ((g[b_] == -1) & (g.T[b_] == 1))  # b -> c
            )[0]
            for c_ in c_list:
                d_list = np.where(
                    ((g[a_] == -1) & (g.T[a_] == -1))   # a - d
                    & ((g[c_] == -1) & (g.T[c_] == 1))  # d -> c
                    & (g[b_] == 0)                        # b -/- d
                )[0]
                for d_ in d_list:
                    if frozenset((b_, d_)) in sep_set and all(
                        a_ in S for S in sep_set[frozenset((b_, d_))]
                    ):
                        g[b_, a_] = 1  # a -> b
                        changed = True
                        break
                if changed:
                    break
        return changed

    g = g.copy()
    while rule1(g, sep_set) or rule2(g) or rule3(g, sep_set) or rule4(g, sep_set):
        continue
    return g


def mb_by_mb_in_mpdag_alg(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    target: int,
    background_knowledge: np.ndarray | None = None,
    mb_algorithm: str = "grow_shrink",
    mb: dict[int, set] | None = None,
    L: dict[int, np.ndarray] | None = None,
    sep_set: dict[frozenset, list] | None = None,
    ignore: Sequence[int] = [],
) -> np.ndarray:
    """
    b-LOAD variant of the MB-by-MB algorithm that incorporates background
    knowledge to produce a local MPDAG around ``target``.

    This function extends :func:`mb_by_mb_alg` with three adaptations:

    1. **BK initialisation** — ``g`` is seeded with ``background_knowledge``
       rather than an empty zero matrix, so known edges are present from the
       start and influence the wait-list pruning immediately.
    2. **BK-aware graph update** — :func:`update_graph_mpdag` is used instead
       of :func:`update_graph`; it re-enforces required edges after each local
       structure is merged (preventing CI-test errors from erasing known edges).
    3. **Four-rule Meek** — :func:`meek_mpdag` is used instead of :func:`meek`;
       the additional Rule 4 propagates orientations introduced by background-
       knowledge edges that the three structural rules alone cannot deduce.

    The wait-list pruning via :func:`reach_with_undirected` also benefits
    from the BK initialisation: directed BK edges block undirected paths
    earlier, reducing the number of MBs that need to be learned.

    .. note::
        Part of the **b-LOAD** algorithm introduced in this work.

    Args:
        data (np.ndarray): Data matrix of shape (n_samples, n_nodes).
        ci_test (Callable[[int, int, list[int]], float]): CI test returning a
            p-value given (X, Y, conditioning_set).
        alpha (float): Significance level for CI tests.
        target (int): Index of the node whose local network is to be learned.
        background_knowledge (np.ndarray | None): Prior edge constraints in the
            standard adjacency format (``bk[i,j] == -1`` and ``bk[j,i] == 1``
            ↔ required edge i→j).  If ``None``, the function behaves
            identically to :func:`mb_by_mb_alg`.
        mb_algorithm (str): MB discovery algorithm — ``"grow_shrink"`` (default)
            or ``"total_conditioning"``.
        mb (dict[int, set] | None): Pre-computed Markov blankets; updated in
            place.
        L (dict[int, np.ndarray] | None): Pre-computed local structures;
            updated in place.
        sep_set (dict[frozenset, list] | None): Pre-computed separating sets;
            updated in place.
        ignore (Sequence[int]): Node indices excluded from all MB searches.

    Returns:
        np.ndarray: Learned local MPDAG ``g`` of shape (n_nodes, n_nodes),
            consistent with ``background_knowledge`` and the CI-test evidence,
            focused on the neighbourhood of ``target``.
    """
    if mb_algorithm == "grow_shrink":
        mb_algorithm = grow_shrink
    elif mb_algorithm == "total_conditioning":
        mb_algorithm = total_conditioning
    else:
        raise ValueError(f"Unknown mb_algorithm: {mb_algorithm}")

    mb = mb or dict()
    L = L or dict()
    sep_set = sep_set or defaultdict(list)
    V = set(range(data.shape[1]))

    done_list = set()
    wait_list = [target]

    # b-LOAD adaptation 1: seed g with background knowledge
    if background_knowledge is not None:
        g = background_knowledge.copy()
    else:
        g = np.zeros((len(V), len(V)), dtype=int)

    while len(wait_list) > 0:
        x = wait_list.pop(0)

        if x not in mb:
            mb[x] = mb_algorithm(V.difference(set(ignore)), x, ci_test, alpha)
        mb_x = mb[x].union({x})

        wait_list.extend([v for v in mb[x] if v not in list(done_list) + wait_list])
        done_list.add(x)

        if x in L:
            pass
        elif (
            n := next((n for n in done_list if n != x and mb_x < mb[n] | {n}), None)
        ) is not None:
            L[x] = copy_local_structure(L[n], mb_x)
        elif mb[x] < done_list:
            L[x] = copy_local_structure(g, mb_x)
        else:
            L[x] = learn_local_structure(V, mb_x, sep_set, ci_test, alpha)

        # b-LOAD adaptation 2: BK-protected graph update
        g = update_graph_mpdag(g, x, L[x], background_knowledge)

        # b-LOAD adaptation 3: four-rule MPDAG Meek
        g = meek_mpdag(g, sep_set)

        # Wait-list pruning: BK edges block undirected paths earlier, so fewer
        # MBs need to be learned when strong background knowledge is available.
        connected = reach_with_undirected(g, target)
        wait_list = [w for w in wait_list if w in connected]

    return g
