"""
dream.py — DREAM4 and DREAM5 real-data experiment runners.

Provides helpers for loading the DREAM4 In Silico Size-100 benchmark datasets
and the DREAM5 network inference challenge data, constructing ground-truth DAGs
from the accompanying gold-standard files, fitting oracle linear SEM weights,
and running b-LOAD experiments via :func:`~src.running.run_experiment_mpdag_real`.

Usage (DREAM4)::

    result_dict, details_dict = run_experiment_dream4(
        subset_number=1, sampling_strategy="local", frac_req=0.3
    )

Usage (DREAM5)::

    result_dict, details_dict = run_experiment_dream5(
        subset_number=2, sampling_strategy="global", frac_req=0.2
    )

Expected file layout::

    experiments/dream4/
        insilico_size100_<N>_multifactorial.tsv
        DREAM4_GoldStandard_InSilico_Size100_multifactorial_<N>.tsv

    experiments/dream5/
        net<N>_expression_data.tsv
        DREAM5_NetworkInference_GoldStandard_Network<N>.tsv
"""

import networkx as nx
import numpy as np
import pandas as pd

from src.real_data_applications.sachs import fit_oracle_weights
from src.running import run_experiment_mpdag_real


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def sanitize_to_dag(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Iteratively break cycles in a directed graph until it is a strict DAG.

    Cycles are broken by removing the first edge reported by
    :func:`networkx.find_cycle` at each iteration.  A warning is printed for
    each removed edge so the user can assess the magnitude of the correction.

    Args:
        graph (nx.DiGraph): Input graph, potentially containing directed cycles
            (e.g. because the biological ground-truth network has feedback loops
            that are not representable as a DAG).

    Returns:
        nx.DiGraph: A copy of ``graph`` with enough edges removed to guarantee
            acyclicity.
    """
    safe_dag = graph.copy()
    cycle_count = 0
    while True:
        try:
            cycle = nx.find_cycle(safe_dag)
            edge_to_remove = cycle[0]
            safe_dag.remove_edge(*edge_to_remove)
            cycle_count += 1
            print(f"Broke cycle by removing edge: {edge_to_remove}")
        except nx.NetworkXNoCycle:
            break
    print(f"Sanitization complete. Removed {cycle_count} edges to ensure acyclicity.")
    return safe_dag


# ---------------------------------------------------------------------------
# DREAM4 In Silico Size-100
# ---------------------------------------------------------------------------


def get_dream4_ground_truth_dag(
    gs_dataframe: pd.DataFrame, node_order: list
) -> nx.DiGraph:
    """
    Build the ground-truth DAG for a DREAM4 In Silico Size-100 network.

    Nodes in the returned graph are integer indices (0-based) corresponding to
    the column order of the multifactorial expression dataset.

    Args:
        gs_dataframe (pd.DataFrame): Gold-standard file loaded with columns
            ``['cause', 'outcome', 'edge']``.  Rows where ``edge == 1`` encode
            true directed edges; rows where ``edge == 0`` are ignored.
        node_order (list[str]): Ordered list of gene names matching the columns
            of the multifactorial data matrix (e.g. ``list(df.columns)``).

    Returns:
        nx.DiGraph: DAG whose nodes are integer indices 0…(n−1) and whose
            edges correspond to the true causal connections in the gold
            standard.  Disconnected genes are included as isolated nodes.
    """
    name_to_idx = {name: idx for idx, name in enumerate(node_order)}

    true_dag = nx.DiGraph()
    true_dag.add_nodes_from(range(len(node_order)))

    edges_by_index = []
    for _, row in gs_dataframe.iterrows():
        cause_name = row["cause"]
        outcome_name = row["outcome"]
        edge_exists = row["edge"]

        if edge_exists == 1:
            if cause_name in name_to_idx and outcome_name in name_to_idx:
                u = name_to_idx[cause_name]
                v = name_to_idx[outcome_name]
                edges_by_index.append((u, v))

    true_dag.add_edges_from(edges_by_index)
    return true_dag


def run_experiment_dream4(
    subset_number: int,
    sampling_strategy: str = "local",
    frac_req: float = 0.3,
) -> tuple[dict, dict]:
    """
    Run a b-LOAD experiment on a DREAM4 In Silico Size-100 network.

    Pipeline:

    1. Load multifactorial expression data and gold-standard edge list.
    2. Construct the binary ground-truth DAG and sanitize it to a strict DAG.
    3. Fit oracle linear SEM weights onto the binary DAG structure.
    4. Enumerate all ancestor–descendant pairs as candidate targets.
    5. Run :func:`~src.running.run_experiment_mpdag_real` with the specified
       background-knowledge fraction and sampling strategy.

    Data files are expected at::

        experiments/dream4/insilico_size100_<subset_number>_multifactorial.tsv
        experiments/dream4/DREAM4_GoldStandard_InSilico_Size100_multifactorial_<subset_number>.tsv

    Args:
        subset_number (int): DREAM4 network index (1–5).
        sampling_strategy (str): Background-knowledge sampling strategy —
            ``"local"`` (edges incident on targets) or ``"global"`` (all edges).
        frac_req (float): Fraction of true edges to reveal as required
            background knowledge.

    Returns:
        tuple[dict, dict]: ``(result_dict, details_dict)`` as returned by
            :func:`~src.running.run_experiment_mpdag_real`.
    """
    multifactorial_df = pd.read_csv(
        f"experiments/dream4/insilico_size100_{subset_number}_multifactorial.tsv",
        sep="\t",
    )
    node_order = list(multifactorial_df.columns)

    dream4_gs = pd.read_csv(
        f"experiments/dream4/DREAM4_GoldStandard_InSilico_Size100_multifactorial_{subset_number}.tsv",
        sep="\t",
        header=None,
        names=["cause", "outcome", "edge"],
    )

    binary_dag = sanitize_to_dag(get_dream4_ground_truth_dag(dream4_gs, node_order))

    dream4_data_array = multifactorial_df.to_numpy()
    oracle_dag = fit_oracle_weights(binary_dag, dream4_data_array)

    forward_pairs_idx = [
        [node, descendant]
        for node in binary_dag.nodes()
        for descendant in nx.descendants(binary_dag, node)
    ]

    base_args = {
        "data_name": f"dream4_net{subset_number}",
        "real_data": dream4_data_array,
        "true_dag": oracle_dag,
        "observed": dream4_data_array.shape[1],
        "alpha": 0.01,
        "ci_test": "fisherz",
        "sampling_strategy": sampling_strategy,
        "fraction_forbidden": 0,
        "fraction_required": frac_req,
    }

    result_dict, details_dict = run_experiment_mpdag_real(base_args, 42, forward_pairs_idx)
    return result_dict, details_dict


# ---------------------------------------------------------------------------
# DREAM5 Network Inference
# ---------------------------------------------------------------------------


def get_dream5_tf_ground_truth_dag(
    gs_dataframe: pd.DataFrame, tf_order: list
) -> nx.DiGraph:
    """
    Build the ground-truth DAG for a DREAM5 sub-network restricted to a list
    of transcription factors (TFs).

    Only edges where *both* the cause and the outcome are in ``tf_order`` are
    included; all other gold-standard edges are ignored.

    Args:
        gs_dataframe (pd.DataFrame): Gold-standard file with at least columns
            ``['cause', 'outcome']``.  All rows are assumed to be true edges
            (unlike DREAM4, there is no ``edge`` column).
        tf_order (list[str]): Ordered list of TF gene names matching the
            columns of the filtered expression dataset.

    Returns:
        nx.DiGraph: DAG whose nodes are integer indices 0…(len(tf_order)−1).
            Disconnected TFs are included as isolated nodes.
    """
    name_to_idx = {name: idx for idx, name in enumerate(tf_order)}

    true_dag = nx.DiGraph()
    true_dag.add_nodes_from(range(len(tf_order)))

    edges_by_index = []
    for _, row in gs_dataframe.iterrows():
        cause_name = row["cause"]
        outcome_name = row["outcome"]
        if cause_name in name_to_idx and outcome_name in name_to_idx:
            u = name_to_idx[cause_name]
            v = name_to_idx[outcome_name]
            edges_by_index.append((u, v))

    true_dag.add_edges_from(edges_by_index)
    return true_dag


def run_experiment_dream5(
    subset_number: int,
    sampling_strategy: str = "local",
    frac_req: float = 0.3,
) -> tuple[dict, dict]:
    """
    Run a b-LOAD experiment on a DREAM5 network inference sub-network.

    Pipeline:

    1. Load the filtered expression dataset and gold-standard edge list for
       the specified network.
    2. Construct the ground-truth DAG over TFs and sanitize it.
    3. Fit oracle linear SEM weights.
    4. Enumerate ancestor–descendant pairs as candidate targets.
    5. Run :func:`~src.running.run_experiment_mpdag_real`.

    Data files are expected at::

        experiments/dream5/net<subset_number>_expression_data.tsv
        experiments/dream5/DREAM5_NetworkInference_GoldStandard_Network<subset_number>.tsv

    Args:
        subset_number (int): DREAM5 network index (1–4).
        sampling_strategy (str): ``"local"`` or ``"global"``.
        frac_req (float): Fraction of true edges to reveal as required BK.

    Returns:
        tuple[dict, dict]: ``(result_dict, details_dict)``.
    """
    expression_df = pd.read_csv(
        f"experiments/dream5/net{subset_number}_expression_data.tsv", sep="\t"
    )
    tf_order = list(expression_df.columns)
    expression_array = expression_df.to_numpy()

    dream5_gs = pd.read_csv(
        f"experiments/dream5/DREAM5_NetworkInference_GoldStandard_Network{subset_number}.tsv",
        sep="\t",
        header=None,
        names=["cause", "outcome"],
    )

    binary_dag = sanitize_to_dag(get_dream5_tf_ground_truth_dag(dream5_gs, tf_order))
    oracle_dag = fit_oracle_weights(binary_dag, expression_array)

    forward_pairs_idx = [
        [node, descendant]
        for node in binary_dag.nodes()
        for descendant in nx.descendants(binary_dag, node)
    ]

    base_args = {
        "data_name": f"dream5_net{subset_number}",
        "real_data": expression_array,
        "true_dag": oracle_dag,
        "observed": expression_array.shape[1],
        "alpha": 0.01,
        "ci_test": "fisherz",
        "sampling_strategy": sampling_strategy,
        "fraction_forbidden": 0,
        "fraction_required": frac_req,
    }

    result_dict, details_dict = run_experiment_mpdag_real(base_args, 42, forward_pairs_idx)
    return result_dict, details_dict
