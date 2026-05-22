"""
sachs.py — Sachs protein-signalling real-data helpers.

Provides the ground-truth DAG for the Sachs et al. (2005) protein-signalling
dataset (11 proteins, 20 directed edges), a helper to map integer node indices
back to protein names, and a utility for fitting oracle linear SEM weights from
observational data onto a binary DAG topology.

The 11-protein ordering in :data:`SACHS_PROTEIN_ORDER` matches the column
order of the Sachs dataset as loaded by ``causal-learn``.

Reference:
    Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P.
    (2005). Causal protein-signaling networks derived from multiparameter
    single-cell data. *Science*, 308(5721), 523–529.
"""

import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SACHS_PROTEIN_ORDER = [
    "Raf", "Mek", "Plcg", "PIP2", "PIP3",
    "Erk", "Akt", "PKA", "PKC", "P38", "Jnk",
]
"""
Ordered list of the 11 protein names corresponding to dataset columns.
Index ``i`` in this list equals node index ``i`` throughout the codebase.
"""

# Name → index mapping (module-level convenience)
name_to_idx = {name: idx for idx, name in enumerate(SACHS_PROTEIN_ORDER)}

# True ancestor–descendant pairs, as (treatment, outcome) string pairs
forward_pairs = [
    # From Plcg
    ["Plcg", "PIP3"], ["Plcg", "PIP2"], ["Plcg", "PKC"], ["Plcg", "PKA"],
    ["Plcg", "Raf"], ["Plcg", "Mek"], ["Plcg", "Erk"], ["Plcg", "Akt"],
    ["Plcg", "P38"], ["Plcg", "Jnk"],
    # From PIP3
    ["PIP3", "PIP2"], ["PIP3", "PKC"], ["PIP3", "PKA"], ["PIP3", "Raf"],
    ["PIP3", "Mek"], ["PIP3", "Erk"], ["PIP3", "Akt"], ["PIP3", "P38"],
    ["PIP3", "Jnk"],
    # From PIP2
    ["PIP2", "PKC"], ["PIP2", "PKA"], ["PIP2", "Raf"], ["PIP2", "Mek"],
    ["PIP2", "Erk"], ["PIP2", "Akt"], ["PIP2", "P38"], ["PIP2", "Jnk"],
    # From PKC
    ["PKC", "PKA"], ["PKC", "Raf"], ["PKC", "Mek"], ["PKC", "Erk"],
    ["PKC", "Akt"], ["PKC", "P38"], ["PKC", "Jnk"],
    # From PKA
    ["PKA", "Raf"], ["PKA", "Mek"], ["PKA", "Erk"], ["PKA", "Akt"],
    ["PKA", "P38"], ["PKA", "Jnk"],
    # From Raf
    ["Raf", "Mek"], ["Raf", "Erk"], ["Raf", "Akt"],
    # From Mek
    ["Mek", "Erk"], ["Mek", "Akt"],
    # From Erk
    ["Erk", "Akt"],
]

# Inverse direction (outcome, treatment) string pairs
inverse_pairs = [
    ["PIP3", "Plcg"], ["PIP2", "Plcg"], ["PKC", "Plcg"], ["PKA", "Plcg"],
    ["Raf", "Plcg"], ["Mek", "Plcg"], ["Erk", "Plcg"], ["Akt", "Plcg"],
    ["P38", "Plcg"], ["Jnk", "Plcg"],
    ["PIP2", "PIP3"], ["PKC", "PIP3"], ["PKA", "PIP3"], ["Raf", "PIP3"],
    ["Mek", "PIP3"], ["Erk", "PIP3"], ["Akt", "PIP3"], ["P38", "PIP3"],
    ["Jnk", "PIP3"],
    ["PKC", "PIP2"], ["PKA", "PIP2"], ["Raf", "PIP2"], ["Mek", "PIP2"],
    ["Erk", "PIP2"], ["Akt", "PIP2"], ["P38", "PIP2"], ["Jnk", "PIP2"],
    ["PKA", "PKC"], ["Raf", "PKC"], ["Mek", "PKC"], ["Erk", "PKC"],
    ["Akt", "PKC"], ["P38", "PKC"], ["Jnk", "PKC"],
    ["Raf", "PKA"], ["Mek", "PKA"], ["Erk", "PKA"], ["Akt", "PKA"],
    ["P38", "PKA"], ["Jnk", "PKA"],
    ["Mek", "Raf"], ["Erk", "Raf"], ["Akt", "Raf"],
    ["Erk", "Mek"], ["Akt", "Mek"],
    ["Akt", "Erk"],
]

# Integer-index versions of the pair lists
forward_pairs_idx = [[name_to_idx[t], name_to_idx[o]] for t, o in forward_pairs]
inverse_pairs_idx = [[name_to_idx[o], name_to_idx[t]] for o, t in inverse_pairs]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_protein_name(index: int) -> str:
    """
    Map a node index back to its protein name.

    Args:
        index (int): 0-based node index (must be in range 0–10).

    Returns:
        str: Protein name corresponding to ``index`` in
            :data:`SACHS_PROTEIN_ORDER`.

    Raises:
        IndexError: If ``index`` is out of range.
    """
    if 0 <= index < len(SACHS_PROTEIN_ORDER):
        return SACHS_PROTEIN_ORDER[index]
    raise IndexError(f"Index {index} is out of bounds for the Sachs protein list.")


def get_sachs_ground_truth_dag() -> nx.DiGraph:
    """
    Construct the ground-truth DAG for the Sachs protein-signalling network.

    The topology follows the consensus network from Sachs et al. (2005),
    comprising 11 proteins and 20 directed causal edges.  Nodes are integer
    indices (0–10) matching :data:`SACHS_PROTEIN_ORDER`.

    Returns:
        nx.DiGraph: Unweighted ground-truth DAG.  Edge weights are added
            separately by :func:`fit_oracle_weights`.
    """
    _name_to_idx = {name: idx for idx, name in enumerate(SACHS_PROTEIN_ORDER)}

    true_dag = nx.DiGraph()
    true_dag.add_nodes_from(range(len(SACHS_PROTEIN_ORDER)))

    edges_by_name = [
        # From Plcg
        ("Plcg", "PIP3"),
        ("Plcg", "PIP2"),
        ("Plcg", "PKC"),
        # From PIP3
        ("PIP3", "PIP2"),
        ("PIP3", "Akt"),
        # From PIP2
        ("PIP2", "PKC"),
        # From PKC
        ("PKC", "PKA"),
        ("PKC", "Raf"),
        ("PKC", "Mek"),
        ("PKC", "Jnk"),
        ("PKC", "P38"),
        # From PKA
        ("PKA", "Raf"),
        ("PKA", "Mek"),
        ("PKA", "Erk"),
        ("PKA", "Akt"),
        ("PKA", "Jnk"),
        ("PKA", "P38"),
        # Downstream cascade
        ("Raf", "Mek"),
        ("Mek", "Erk"),
        ("Erk", "Akt"),
    ]

    edges_by_index = [(_name_to_idx[u], _name_to_idx[v]) for u, v in edges_by_name]
    true_dag.add_edges_from(edges_by_index)
    return true_dag


def fit_oracle_weights(true_dag: nx.DiGraph, real_samples: np.ndarray) -> nx.DiGraph:
    """
    Fit linear SEM coefficients onto a binary DAG topology using observational
    data, producing an *oracle* weighted DAG for intervention-distance
    evaluation.

    For each node ``v`` with parents ``pa(v)``, an OLS regression
    ``v ~ pa(v)`` is fit on ``real_samples``.  The fitted coefficients are
    stored as ``weight`` attributes on the corresponding edges, matching the
    edge-weight convention expected by
    :func:`~src.evaluation.true_linear_gaussian_effect`.

    Args:
        true_dag (nx.DiGraph): Binary ground-truth DAG.  Node indices must
            correspond to columns of ``real_samples``.
        real_samples (np.ndarray): Observational data matrix of shape
            (n_samples, n_nodes).

    Returns:
        nx.DiGraph: Copy of ``true_dag`` with ``weight`` attributes on every
            edge set to the OLS-fitted linear coefficient.
    """
    oracle_dag = true_dag.copy()

    for node in oracle_dag.nodes():
        parents = list(oracle_dag.predecessors(node))
        if not parents:
            continue

        X = real_samples[:, parents]
        y = real_samples[:, node]

        reg = LinearRegression(fit_intercept=True).fit(X, y)

        for i, parent in enumerate(parents):
            oracle_dag[parent][node]["weight"] = reg.coef_[i]

    return oracle_dag
