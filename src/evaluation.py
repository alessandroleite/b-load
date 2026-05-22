"""
evaluation.py — Ground-truth computation and algorithm evaluation metrics.

This module provides two groups of functionality:

1. **Ground-truth helpers**: convert DAGs to CPDAGs, enumerate true optimal
   adjustment sets, and compute true average treatment effects (ATEs) under
   linear Gaussian and binary SEMs.

2. **Evaluation metrics**: precision, recall, F1 and intervention distance for
   comparing estimated adjustment sets or ATEs to ground truth.

These functions are shared by both the LOAD (base) and b-LOAD pipelines.

Notes
-----
Several functions call R via ``rpy2`` (specifically ``pcalg``'s
``dag2cpdag`` and ``optAdjSet``, and the binary ATE helper).  The R
evaluation script ``evaluate.R`` is sourced automatically at module import.
"""

from pathlib import Path
from itertools import permutations

import numpy as np
import networkx as nx
from rpy2.robjects import default_converter, globalenv, numpy2ri, r
from rpy2 import robjects

# Source the R evaluation functions from the project root
file_path = Path(__file__).resolve().parent.parent / "evaluate.R"

# Source the file using the R 'source' function via robjects
if file_path.exists():
    robjects.r.source(str(file_path))
    print(f"Successfully sourced: {file_path}")
else:
    print(f"Error: File not found at {file_path}")
    raise FileNotFoundError(f"R evaluation script not found at {file_path}")


def dag2cpdag(dag: np.ndarray) -> np.ndarray:
    """
    Convert a DAG adjacency matrix to its CPDAG (Markov equivalence class)
    using the ``pcalg`` R package.

    Args:
        dag (np.ndarray): Binary DAG adjacency matrix (non-zero entry = edge).

    Returns:
        np.ndarray: CPDAG adjacency matrix as an ``int8`` numpy array.
            ``cpdag[i, j] == 1`` indicates an edge from i to j (directed or
            undirected in the pcalg convention).
    """
    with (default_converter + numpy2ri.converter).context():
        cpdag = r["dag2cpdag"](r["as"](dag != 0, "graphNEL"))
        cpdag = np.array(r["as"](cpdag, "matrix")).astype(np.int8)
    return cpdag


def get_test_nums(results: list) -> np.ndarray:
    """Return the total number of CI tests performed for each result dict."""
    test_sum = [np.sum(h["tests"]) for h in results]
    return np.array(test_sum)


def get_times(results: list) -> np.ndarray:
    """Return the wall-clock time (seconds) recorded for each result dict."""
    return np.array([h["time"] for h in results])


# ADJ-SETS
def get_optimal_adj_set(amat: np.ndarray, treatment: int, outcome: int):
    """
    Compute the optimal adjustment set (o-set) for the effect of ``treatment``
    on ``outcome`` in the CPDAG / DAG represented by ``amat``, using
    ``pcalg``'s ``optAdjSet`` R function.

    Returns ``None`` if ``treatment`` is not an ancestor of ``outcome`` in
    ``amat``, or if ``pcalg`` raises an exception (e.g. the effect is not
    identifiable from the given graph).

    Args:
        amat (np.ndarray): CPDAG or DAG adjacency matrix.  A non-zero entry
            at ``[i, j]`` indicates an edge from i toward j (pcalg convention).
        treatment (int): 0-based index of the treatment variable.
        outcome (int): 0-based index of the outcome variable.

    Returns:
        set[int] | None: The optimal adjustment set as a set of 0-based node
            indices, or ``None`` if the effect is not identifiable.
    """
    if not is_ancestor(treatment, outcome, amat):
        return None
    try:
        with (default_converter + numpy2ri.converter).context():
            S = r["optAdjSet"](
                (r["as"](amat != 0, "graphNEL")), treatment + 1, outcome + 1
            )
        return set(np.array(S) - 1)
    except Exception:
        return None


def get_true_osets(experiments: dict) -> dict:
    """
    Compute the true optimal adjustment sets for a dictionary of synthetic
    experiments.

    For each experiment, the true DAG is first converted to its CPDAG (Markov
    equivalence class), and the optimal adjustment set is then retrieved from
    that CPDAG via ``pcalg``'s ``optAdjSet``.

    Args:
        experiments (dict): Keyed by experiment ID.  Each value must contain:

            * ``"true_dag"`` (nx.DiGraph): Ground-truth DAG.
            * ``"targets"`` (sequence): Pair of 0-based node indices
              ``[t1, t2]``; the ancestor direction is inferred from the DAG.

    Returns:
        dict: Keyed by experiment ID.  Each value is a dict with:

            * ``"treatment"`` (int): 0-based treatment index.
            * ``"outcome"`` (int): 0-based outcome index.
            * ``"oset"`` (set[int] | None): True optimal adjustment set, or
              ``None`` if not identifiable from the CPDAG.
    """
    true_osets = {}

    for exp_id, exp in experiments.items():
        # 1. Convert the true DAG to a CPDAG
        # Using a helper (dag2cpdag) to represent the Markov Equivalence Class
        true_dag_matrix = nx.to_numpy_array(exp["true_dag"])
        cpdag = dag2cpdag(true_dag_matrix)

        # 2. Identify treatment and outcome from the true DAG
        # Based on explicit ancestral relations
        targets = exp["targets"]
        if nx.has_path(exp["true_dag"], targets[0], targets[1]):
            treatment, outcome = targets[0], targets[1]
        else:
            treatment, outcome = targets[1], targets[0]

        # 3. Retrieve the true optimal adjustment set relative to the CPDAG
        true_oset = get_optimal_adj_set(cpdag, treatment, outcome)

        # 4. Store the ground truth result
        true_osets[exp_id] = {
            "treatment": treatment,
            "outcome": outcome,
            "oset": true_oset,
        }

        # print(f"Processed Experiment {exp_id}: Treatment={treatment}, Outcome={outcome}")

    return true_osets

def get_precision(true: set, pred: set) -> float:
    """
    Compute precision of a predicted adjustment set against the ground truth.

    Args:
        true (set): True optimal adjustment set.
        pred (set): Predicted adjustment set.

    Returns:
        float: Precision = TP / (TP + FP).  Returns 1.0 if ``true == pred``,
            0.0 if ``pred`` is empty and ``true`` is not.
    """
    if true == pred:
        return 1.0
    tp = len(true.intersection(pred))
    fp = len(pred - true)
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def get_recall(true: set, pred: set) -> float:
    """
    Compute recall of a predicted adjustment set against the ground truth.

    Args:
        true (set): True optimal adjustment set.
        pred (set): Predicted adjustment set.

    Returns:
        float: Recall = TP / (TP + FN).  Returns 1.0 if ``true == pred``.
    """
    if true == pred:
        return 1.0
    tp = len(true.intersection(pred))
    fn = len(true - pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def get_f1(precision: float, recall: float) -> float:
    """
    Compute the F1 score from precision and recall.

    Args:
        precision (float): Precision score in [0, 1].
        recall (float): Recall score in [0, 1].

    Returns:
        float: Harmonic mean of precision and recall; 0.0 if both are zero.
    """
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


def get_oset(project: str, h: dict, treatment: int, outcome: int):
    if project in ["pc", "fges", "marvel", "snap"]:
        est_oset = get_optimal_adj_set(np.array(h["amat"]), treatment, outcome)
        if est_oset is not None:
            est_osets = [est_oset]
        else:
            est_osets = None
    elif project in ["mb_by_mb", "ldecc", "mb_by_mb_plus", "ldecc_plus"]:
        adj_sets = eval(h["adj_sets"])
        if (treatment, outcome) in adj_sets:
            est_osets = adj_sets[(treatment, outcome)]
        else:
            est_osets = None
    elif project in ["ldp", "ldp_plus"]:
        results = eval(h["results"])
        if (treatment, outcome) in results and results[(treatment, outcome)][
            "vas_exists"
        ]:
            parts = results[(treatment, outcome)]
            est_osets = [
                set(parts["Z1"]),
                set(parts["Z1"] + parts["Z4"]),
                set(parts["Z1"] + parts["Z4"] + parts["Z5"]),
            ]
        else:
            est_osets = None
    elif project in ["load", "load_oracle"]:
        adj_sets = eval(h["adj_sets"])
        if h["identifiable"] and (treatment, outcome) in adj_sets:
            est_osets = adj_sets[(treatment, outcome)]
        else:
            est_osets = None
    else:
        raise ValueError("Invalid project")
    return est_osets


def evaluate_oset(algorithm, results, true_osets: dict):
    """
    Sequentially evaluates discovered adjustment sets against ground truth.
    Returns trimmed arrays (removing outliers) for precision, recall, and F1.
    """
    prec_scores = []
    rec_scores = []
    f1_scores = []

    for h in results:
        # 1. Handle algorithm failures
        if "failed" in h and h["failed"]:
            prec_scores.append(0.0)
            rec_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        exp_id = h["id"]
        treatment = true_osets[exp_id]["treatment"]
        outcome = true_osets[exp_id]["outcome"]

        # 2. Get estimated optimal adjustment sets
        # In a sequential run, we call the function directly
        est_osets = get_oset(algorithm, h, treatment, outcome)
        true_oset = true_osets[exp_id]["oset"]

        # 3. Calculate Scores (Precision, Recall, F1)
        # Based on the paper's metrics for adjustment set quality
        if true_oset is not None and est_osets is not None:
            precs = [get_precision(true_oset, s) for s in est_osets]
            recs = [get_recall(true_oset, s) for s in est_osets]
            f1s = [get_f1(p, r) for p, r in zip(precs, recs)]

            # Record the best match among returned sets
            prec_scores.append(max(precs))
            rec_scores.append(max(recs))
            f1_scores.append(max(f1s))

        elif true_oset == est_osets:  # Both are None (case for non-existent Oset)
            prec_scores.append(1.0)
            rec_scores.append(1.0)
            f1_scores.append(1.0)
        else:
            # Case where one exists and the other doesn't
            prec_scores.append(0.0)
            rec_scores.append(0.0)
            f1_scores.append(0.0)

    # 4. Return results
    return (
        (prec_scores),
        (rec_scores),
        (f1_scores),
    )


# Modified functions to evaluate on real data


def get_true_osets_real_data(experiments: dict) -> dict:
    """
    Compute the true optimal adjustment sets for a dictionary of real-data
    experiments.

    Unlike :func:`get_true_osets`, this function queries the o-set directly
    from the **true DAG matrix** (not from the CPDAG), which is appropriate
    for real datasets where the ground-truth DAG is known (e.g. Sachs protein
    network, DREAM benchmark) but no CPDAG conversion is available or needed.

    Args:
        experiments (dict): Keyed by experiment ID.  Each value must contain:

            * ``"true_dag"`` (nx.DiGraph): Ground-truth DAG with node indices
              matching the columns of the data matrix.
            * ``"targets"`` (sequence): Pair of 0-based node indices
              ``[t1, t2]``; the ancestor direction is inferred from the DAG.

    Returns:
        dict: Keyed by experiment ID.  Each value is a dict with:

            * ``"treatment"`` (int): 0-based treatment index.
            * ``"outcome"`` (int): 0-based outcome index.
            * ``"oset"`` (set[int] | None): True optimal adjustment set from
              the true DAG, or ``None`` if not identifiable.
    """
    true_osets = {}
    for exp_id, exp in experiments.items():
        true_dag_matrix = nx.to_numpy_array(exp["true_dag"])

        targets = exp["targets"]
        if nx.has_path(exp["true_dag"], targets[0], targets[1]):
            treatment, outcome = targets[0], targets[1]
        else:
            treatment, outcome = targets[1], targets[0]

        true_oset = get_optimal_adj_set(true_dag_matrix, treatment, outcome)

        true_osets[exp_id] = {
            "treatment": treatment,
            "outcome": outcome,
            "oset": true_oset,
        }

    return true_osets


# INTERVENTION DISTANCE
def is_ancestor(t1: int, t2: int, amat: np.ndarray) -> bool:
    """
    Check whether ``t1`` is an ancestor of ``t2`` in the directed graph
    represented by ``amat``.

    Uses matrix exponentiation on the boolean reachability matrix.

    Args:
        t1 (int): Candidate ancestor (0-based index).
        t2 (int): Candidate descendant (0-based index).
        amat (np.ndarray): Adjacency matrix where ``amat[i,j] != 0`` encodes
            a directed edge i → j.

    Returns:
        bool: ``True`` if there exists a directed path t1 → … → t2.
    """
    reach = amat.copy().astype(bool)
    np.fill_diagonal(reach, True)
    reach = np.linalg.matrix_power(reach, reach.shape[0] - 1)
    return reach[t1, t2]


def true_linear_gaussian_effect(
    treatment: int, outcome: int, true_dag: nx.DiGraph, **kwargs
) -> float:
    """
    Compute the true total causal effect of ``treatment`` on ``outcome`` in a
    linear Gaussian SEM by summing path products.

    The edge weights in ``true_dag`` are the linear SEM coefficients (set by
    :func:`~src.data_generation.generate_data`).  The total effect equals the
    sum over all directed paths of the products of edge coefficients along each
    path.

    Args:
        treatment (int): 0-based treatment index.
        outcome (int): 0-based outcome index.
        true_dag (nx.DiGraph): Weighted ground-truth DAG.
        **kwargs: Silently ignored (for interface compatibility with
            ``true_binary_effect``).

    Returns:
        float: True total causal effect; 0.0 if ``treatment`` is not an
            ancestor of ``outcome``.
    """
    if treatment not in nx.ancestors(true_dag, outcome):
        return 0.0
    amat = nx.to_numpy_array(true_dag)
    return sum(
        np.prod([amat[path[i], path[i + 1]] for i in range(len(path) - 1)])
        for path in nx.all_simple_paths(true_dag, treatment, outcome)
    )


def true_binary_effect(
    treatment: int, outcome: int, true_dag: nx.DiGraph, cpt: object, **kwargs
) -> float:
    """
    Compute the true total causal effect of ``treatment`` on ``outcome`` in a
    discrete binary SEM, using the R helper ``true_binary_effect``.

    Args:
        treatment (int): 0-based treatment index.
        outcome (int): 0-based outcome index.
        true_dag (nx.DiGraph): Ground-truth DAG.
        cpt: R conditional-probability table object produced by
            :func:`~src.data_generation.generate_data`.
        **kwargs: Silently ignored.

    Returns:
        float: True binary causal effect E[Y | do(X=1)] − E[Y | do(X=0)];
            0.0 if ``treatment`` is not an ancestor of ``outcome``.
    """
    if treatment not in nx.ancestors(true_dag, outcome):
        return 0.0
    with (default_converter + numpy2ri.converter).context():
        return globalenv["true_binary_effect"](
            int(treatment) + 1, int(outcome) + 1, nx.to_numpy_array(true_dag), cpt
        )[0]


def true_causal_effects(experiments: dict, family: str = "gaussian") -> dict:
    """
    Compute the true causal effects for all target pairs across all experiments.

    Args:
        experiments (dict): Keyed by experiment ID.  Each value must contain
            ``"targets"`` (pair of 0-based node indices), ``"true_dag"``
            (``nx.DiGraph``), and ``"cpt"`` (R CPT object, only used for
            ``family="binary"``).
        family (str): SEM family — ``"gaussian"`` (uses
            :func:`true_linear_gaussian_effect`) or ``"binary"`` (uses
            :func:`true_binary_effect`).

    Returns:
        dict: Nested dict ``effects[exp_id][(t1, t2)]`` = true total causal
            effect of ``t1`` on ``t2``.
    """
    if family == "gaussian":
        get_effect = true_linear_gaussian_effect
    elif family == "binary":
        get_effect = true_binary_effect
    else:
        raise ValueError("Invalid family")
    effects = {}
    for exp_id, exp in experiments.items():
        effects[exp_id] = {}
        for t1, t2 in permutations(exp["targets"], 2):
            effects[exp_id][(t1, t2)] = get_effect(
                treatment=t1, outcome=t2, true_dag=exp["true_dag"], cpt=exp["cpt"]
            )
    return effects


def estimate_binary(samples, val, outcome, treatment, adj_set):
    # Filter rows based on the treatment value
    samples_val = samples[samples[:, treatment] == val]
    # Create the design matrix X
    X = np.column_stack(
        [np.ones(len(samples_val))] + [samples_val[:, idx] for idx in adj_set]
    )
    # Outcome variable (filtered by val)
    y = samples_val[:, outcome]
    # Solve the least squares problem X * beta = y
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # Predict the outcome for all rows in the original samples using the estimated beta
    X_full = np.column_stack(
        [np.ones(len(samples))] + [samples[:, idx] for idx in adj_set]
    )
    y_pred = X_full @ beta
    return y_pred.mean()


def estimate_ate(
    treatment: int,
    outcome: int,
    adj_sets: list[list[int]],
    samples: np.ndarray,
    family: str = "gaussian",
) -> list[float]:
    effects = []
    for adj_set in adj_sets:
        adj_set = list(adj_set)
        if family == "gaussian":
            x = samples[:, [treatment] + adj_set]
            y = samples[:, outcome]
            A = np.vstack((x.T, np.ones(len(x))))
            effect = np.linalg.lstsq(A.T, y, rcond=None)[0][0]
            effects.append(effect)
        elif family == "binary":
            do_1 = estimate_binary(samples, 1, outcome, treatment, adj_set)
            do_0 = estimate_binary(samples, 0, outcome, treatment, adj_set)
            effects.append(do_1 - do_0)
    return effects


def get_adj_sets(project: str, h: dict, t1: int, t2: int):
    if "failed" in h and h["failed"]:
        return [[]]
    if project in ["pc", "fges", "marvel", "snap"]:
        amat = np.array(h["amat"], dtype=np.int8)
        if is_ancestor(t1, t2, amat):
            oset = get_optimal_adj_set(amat, t1, t2)
            if oset != None:  # identifiable
                return [oset]
            else:  # unidentifiable, fall back to local IDA
                amat = -amat.copy()
                amat[np.logical_and(amat == 0, amat.T == -1)] = 1
                # return get_locally_valid_parent_sets(amat, t1, t2)
                return [[]]  # Placeholder
        else:
            return None
    elif project in ["mb_by_mb", "ldecc", "mb_by_mb_plus", "ldecc_plus"]:
        adj_sets: dict = eval(h["adj_sets"])
        return adj_sets.get((t1, t2), None)
    elif project in ["ldp", "ldp_plus"]:
        results: dict = eval(h["results"])
        if (t1, t2) in results:
            parts = results[(t1, t2)]
            return [
                parts["Z1"],
                parts["Z1"] + parts["Z4"],
                parts["Z1"] + parts["Z4"] + parts["Z5"],
            ]
        else:
            return None
    elif project == "local_optimal" or project.startswith("load"):
        adj_sets: dict = eval(h["adj_sets"])
        return adj_sets.get((t1, t2), None)
    else:
        raise ValueError("Invalid project")


def estimate_ates(
    results,
    algorithm: str,
    samples: dict,
    family: str = "gaussian",
) -> dict:
    ates = {}
    for h in results:
        exp_id = h["id"]
        ates[exp_id] = {}
        for t1, t2 in permutations(h["targets"], 2):
            adj_sets = get_adj_sets(algorithm, h, t1, t2)
            if adj_sets is None:
                ates[exp_id][(t1, t2)] = [0.0]
            else:
                ates[exp_id][(t1, t2)] = estimate_ate(
                    treatment=t1,
                    outcome=t2,
                    adj_sets=adj_sets,
                    samples=samples[exp_id],
                    family=family,
                )
    return ates


def intervention_distance(
    est_ates: dict, true_ates: dict, aggr: str = "abs"
) -> np.ndarray:
    """
    Compute the intervention distance between estimated and true ATEs.

    For each experiment and each treatment–outcome pair, the distance is the
    mean absolute error (or mean squared error) across all candidate adjustment
    sets returned by the algorithm.  The per-experiment distance is the mean
    over all target pairs.

    Args:
        est_ates (dict): Estimated ATEs as produced by :func:`estimate_ates`.
            ``est_ates[exp_id][(t1, t2)]`` is a list of ATE estimates (one per
            candidate adjustment set).
        true_ates (dict): True ATEs as produced by :func:`true_causal_effects`.
        aggr (str): Aggregation — ``"abs"`` (mean absolute error, default) or
            ``"mse"`` (mean squared error).

    Returns:
        np.ndarray: 1-D array of per-experiment distances.
    """
    distances = []
    for exp in est_ates:
        exp_dist = []
        for pair in est_ates[exp]:
            true_ate = true_ates[exp][pair]
            if aggr == "mse":
                dist = [(true_ate - est_ate) ** 2 for est_ate in est_ates[exp][pair]]
            elif aggr == "abs":
                dist = [np.abs(true_ate - est_ate) for est_ate in est_ates[exp][pair]]
            else:
                raise ValueError("Invalid aggregation")
            exp_dist.append(np.mean(dist))
        distances.append(np.mean(exp_dist))
    return np.array(distances)


def evaluate_intervention(algorithm: str, results: list, true_effects: dict, family: str = "gaussian") -> np.ndarray:
    """
    Evaluate the interventional performance of an algorithm by comparing its
    estimated ATEs to the ground-truth ATEs.

    Pipeline:

    1. Build a ``samples_dict`` mapping each experiment ID to its data matrix.
    2. Estimate ATEs from each result's discovered adjustment sets using
       :func:`estimate_ates`.
    3. Compute the intervention distance (mean absolute error) via
       :func:`intervention_distance`.

    Args:
        algorithm (str): Algorithm name used to parse adjustment sets from
            each result dict (passed to :func:`get_adj_sets`).
        results (list): List of result dicts, each containing at minimum
            ``"id"``, ``"data"``, and the adjustment-set key appropriate for
            ``algorithm``.
        true_effects (dict): Ground-truth ATEs, keyed by experiment ID then
            (treatment, outcome) pair.  Produced by :func:`true_causal_effects`.
        family (str): SEM family — ``"gaussian"`` (default) or ``"binary"``.

    Returns:
        np.ndarray: 1-D array of intervention distances (one per experiment).
    """
    samples_dict = {h["id"]: h["data"] for h in results}
    est_ates = estimate_ates(results, algorithm, samples_dict, family=family)
    distances = intervention_distance(est_ates, true_effects)
    return distances
