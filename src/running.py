"""
running.py — Experiment runners for synthetic and real-data benchmarks.

This module orchestrates the full experimental pipeline: data generation,
background-knowledge sampling, algorithm execution, and evaluation.  Each
runner encapsulates one experimental condition and returns a summary dict
compatible with the aggregation utilities in ``src/analysis.py``.

Two families of runners are provided:

**b-LOAD / LOAD runners** (local and global BK variants):
    ``run_algorithm_mpdag``, ``get_experiment_mpdag``, ``run_experiment_mpdag_light``
    ``get_experiment_mpdag_v2``, ``run_experiment_mpdag_v2_light``
    ``get_experiment_mpdag_real``, ``run_experiment_mpdag_real``
    ``get_experiment_mpdag_noised``, ``run_experiment_mpdag_noised_light``

**PC / b-PC runners** (baselines):
    ``run_algorithm_pc``, ``get_experiment_pc``, ``run_experiment_pc_light``
    ``run_algorithm_bpc``, ``get_experiment_bpc``, ``run_experiment_bpc_light``
    ``get_experiment_bpc_v2``, ``run_experiment_bpc_v2_light``

All ``run_experiment_*_light`` functions iterate over a list of experiment
seeds and call their corresponding ``get_experiment_*`` function in a loop,
collecting per-iteration results into summary statistics.
"""

from time import perf_counter

import numpy as np
import networkx as nx
from tqdm import tqdm

from src.counting_ci_tests import CountingTest
from src.load import load, load_in_mpdag
from src.pc import pc, pc_with_bk
from src.data_generation import generate_data
from src.background_knowledge import (
    sample_local_background_knowledge,
    sample_background_knowledge_v2,
    initialize_background_knowledge,
    sample_local_background_knowledge_noised,
)
from src.evaluation import (
    get_true_osets,
    evaluate_oset,
    true_causal_effects,
    evaluate_intervention,
    get_true_osets_real_data,
)


# ---------------------------------------------------------------------------
# b-LOAD / LOAD — local BK sampling (edges incident on targets)
# ---------------------------------------------------------------------------

def run_algorithm_mpdag(
    id: str,
    data: np.ndarray,
    targets: list[int],
    ci_test: str,
    alpha: float,
    with_b: bool = False,
    bg_knowledge: Optional[dict] = None,
    true_dag: nx.Optional[DiGraph] = None,
    cpt: Optional[dict] = None,
    logging: bool = False,
    **kwargs,
) -> dict:
    """
    Run a single LOAD or b-LOAD trial on one set of targets and return a result
    dict augmented with timing, CI test counts, and metadata.

    When ``with_b=True`` the b-LOAD algorithm (:func:`~src.load.load_in_mpdag`)
    is used with the provided ``bg_knowledge`` matrix.  When ``with_b=False``
    the base LOAD algorithm (:func:`~src.load.load`) is used.

    Part of the **b-LOAD** experimental pipeline.

    Args:
        id (str): Unique experiment identifier stored in the result dict.
        data (np.ndarray): Data matrix (n_samples × n_nodes).
        targets (list[int]): ``[treatment, outcome]`` 0-based indices.
        ci_test (str): CI test name for :class:`~src.counting_ci_tests.CountingTest`
            (e.g. ``"fisherz"``).
        alpha (float): Significance level.
        with_b (bool): If ``True``, run b-LOAD; otherwise run base LOAD.
        bg_knowledge (np.Optional[ndarray]): Background-knowledge adjacency matrix
            (only used when ``with_b=True``).
        true_dag (nx.Optional[DiGraph]): Ground-truth DAG stored in the result
            (for downstream evaluation).
        cpt: R CPT object stored in the result (for binary data evaluation).
        logging (bool): Verbose progress logging.
        **kwargs: Forwarded to the CI test constructor and algorithm.

    Returns:
        dict: Algorithm result dict extended with ``"time"``, ``"id"``,
            ``"targets"``, ``"tests"``, ``"cpt"``, ``"true_dag"``, ``"data"``.
    """
    ci_test = CountingTest(data, ci_test, **kwargs)

    start = perf_counter()
    if with_b:
        result = load_in_mpdag(
            data=data,
            ci_test=ci_test,
            alpha=alpha,
            targets=targets,
            background_knowledge=bg_knowledge,
            mb_algorithm="grow_shrink",
            logging=logging,
            **kwargs,
        )
    else:
        result = load(
            data=data,
            ci_test=ci_test,
            alpha=alpha,
            targets=targets,
            mb_algorithm="grow_shrink",
            logging=logging,
            **kwargs,
        )

    result["time"] = perf_counter() - start
    result["id"] = id
    result["targets"] = targets
    result["tests"] = ci_test.get_tests_per_order().tolist()
    result["cpt"] = cpt
    result["true_dag"] = true_dag
    result["data"] = data

    return result


def get_experiment_mpdag(
    seed: int,
    observed: int,
    exp_degree: float,
    max_degree: int,
    targets: int,
    expl_anc: bool,
    identifiable: bool,
    min_adj_size: int,
    samples_num: int,
    discrete: bool,
    save: bool,
    with_b: bool,
    logging: bool,
    fraction_forbidden: float = 0.1,
    fraction_required: float = 0.1,
    alpha: float = 0.01,
    ci_test: str = "fisherz",
) -> dict:
    """
    Generate synthetic data for one seed and run LOAD or b-LOAD with local
    background knowledge.

    Background knowledge is sampled locally — only edges incident on the
    treatment or outcome nodes are revealed — using
    :func:`~src.background_knowledge.sample_local_background_knowledge`.

    Part of the **b-LOAD** experimental pipeline.

    Args:
        seed (int): Random seed for data generation and BK sampling.
        observed (int): Number of observed nodes.
        exp_degree (float): Expected graph degree.
        max_degree (int): Maximum node degree.
        targets (int): Number of target pairs to sample.
        expl_anc (bool): Require explicit ancestor–descendant target pair.
        identifiable (bool): Require identifiable causal effect.
        min_adj_size (int): Minimum true adjustment-set size.
        samples_num (int): Number of i.i.d. observations.
        discrete (bool): Use discrete (binary) data.
        save (bool): Cache generated data to disk.
        with_b (bool): Run b-LOAD (``True``) or base LOAD (``False``).
        logging (bool): Verbose progress output.
        fraction_forbidden (float): Fraction of non-edges to mark forbidden.
        fraction_required (float): Fraction of true edges to reveal as required.
        alpha (float): CI test significance level.
        ci_test (str): CI test name (e.g. ``"fisherz"``).

    Returns:
        dict: Result dict from :func:`run_algorithm_mpdag`.
    """

    # Generate the synthetic data
    data = generate_data(
        seed=seed,
        observed=observed,        # Slightly more nodes to ensure a richer MB
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,   # More samples for more reliable Fisher-Z tests
        expl_anc=expl_anc,
        discrete=discrete,
        save=save
    )


    # Extract components
    true_dag = data["true_dag"]
    cpt = data["cpt"]
    targets = data["targets"]


    # Sample background knowledge (e.g., 20% of edges)
    bg_knowledge = sample_local_background_knowledge(true_dag, targets, fraction_required, fraction_forbidden, seed)

    initial_G = initialize_background_knowledge(num_nodes=observed, bk_dict=bg_knowledge)

    return run_algorithm_mpdag(
        ci_test=ci_test,
        alpha=alpha,
        with_b=with_b,
        bg_knowledge=initial_G,
        logging=logging,
        **data
    )


def run_experiment_mpdag(base_args, base_seed, n_exp):

  experiments_with_b = {}
  experiments_without_b = {}

  for s in tqdm(range(n_exp)):
    exp_with_b = get_experiment_mpdag(**base_args, seed=base_seed+s, with_b=True)
    experiments_with_b[exp_with_b["id"]] = exp_with_b

    exp_without_b = get_experiment_mpdag(**base_args, seed=base_seed+s, with_b=False)
    experiments_without_b[exp_without_b["id"]] = exp_without_b

  # Global metrics
  true_osets_with_b = get_true_osets(experiments_with_b)
  f1_with_b = np.sort(evaluate_oset("load", list(experiments_with_b.values()), true_osets_with_b)[2])

  true_osets_without_b = get_true_osets(experiments_without_b)
  f1_without_b = np.sort(evaluate_oset("load", list(experiments_without_b.values()), true_osets_without_b)[2])

  # Intervention metrics
  family = "binary" if base_args.get("discrete", False) else "gaussian"
  true_effects = true_causal_effects(experiments_with_b, family=family)

  int_dist_with_b = evaluate_intervention("load", list(experiments_with_b.values()), true_effects, family=family)
  int_dist_without_b = evaluate_intervention("load", list(experiments_without_b.values()), true_effects, family=family)

  f1_dict = {
      "f1_with_b": {"mean":f1_with_b.mean(), "std": f1_with_b.std()},
      "f1_without_b": {"mean":f1_without_b.mean(), "std": f1_without_b.std()},
      "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
      "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
  }

  return f1_dict


def run_experiment_mpdag_light(base_args, base_seed, n_exp):
    """
    Run ``n_exp`` paired LOAD / b-LOAD trials with local BK and return
    aggregate summary statistics.

    "Light" means only scalar metrics are kept in memory; full experiment
    dicts are discarded after each iteration to reduce peak memory usage.

    Part of the **b-LOAD** experimental pipeline.

    Args:
        base_args (dict): Keyword arguments forwarded to
            :func:`get_experiment_mpdag` (excluding ``seed`` and ``with_b``).
        base_seed (int): Starting seed; iteration ``s`` uses ``base_seed + s``.
        n_exp (int): Number of independent trials.

    Returns:
        dict: Summary statistics with keys ``"f1_with_b"``, ``"f1_without_b"``,
            ``"int_dist_with_b"``, ``"int_dist_without_b"``, each mapping to
            a sub-dict with ``"mean"`` and ``"std"``.
    """

    f1_with_b_list = []
    f1_without_b_list = []
    int_dist_with_b_list = []
    int_dist_without_b_list = []
    family = "binary" if base_args.get("discrete", False) else "gaussian"

    for s in tqdm(range(n_exp)):
        exp_with_b = get_experiment_mpdag(**base_args, seed=base_seed+s, with_b=True)
        exp_without_b = get_experiment_mpdag(**base_args, seed=base_seed+s, with_b=False)

        # Evaluate immediately, store only scalars
        true_osets_with_b = get_true_osets({exp_with_b["id"]: exp_with_b})
        f1_with_b_list.append(evaluate_oset("load", [exp_with_b], true_osets_with_b)[2])

        true_osets_without_b = get_true_osets({exp_without_b["id"]: exp_without_b})
        f1_without_b_list.append(evaluate_oset("load", [exp_without_b], true_osets_without_b)[2])

        true_effects = true_causal_effects({exp_with_b["id"]: exp_with_b}, family=family)
        int_dist_with_b_list.append(evaluate_intervention("load", [exp_with_b], true_effects, family=family))
        int_dist_without_b_list.append(evaluate_intervention("load", [exp_without_b], true_effects, family=family))

        # Explicitly discard large objects
        del exp_with_b, exp_without_b

    f1_with_b = np.sort(np.concatenate(f1_with_b_list))
    f1_without_b = np.sort(np.concatenate(f1_without_b_list))
    int_dist_with_b = np.concatenate(int_dist_with_b_list)
    int_dist_without_b = np.concatenate(int_dist_without_b_list)

    return {
        "f1_with_b": {"mean": f1_with_b.mean(), "std": f1_with_b.std()},
        "f1_without_b": {"mean": f1_without_b.mean(), "std": f1_without_b.std()},
        "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
        "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
    }


# ---------------------------------------------------------------------------
# b-LOAD / LOAD — global BK sampling (all edges in the graph)
# ---------------------------------------------------------------------------


def get_experiment_mpdag_v2(
    seed: int,
    observed: int,
    exp_degree: float,
    max_degree: int,
    targets: int,
    expl_anc: bool,
    identifiable: bool,
    min_adj_size: int,
    samples_num: int,
    discrete: bool,
    save: bool,
    with_b: bool,
    logging: bool,
    fraction_forbidden: float = 0.1,
    fraction_required: float = 0.1,
    alpha: float = 0.01,
    ci_test: str = "fisherz",
) -> dict:
    """
    Generate synthetic data and run LOAD or b-LOAD with **global** background
    knowledge (edges sampled from the entire graph, not just around targets).

    Background knowledge is sampled using
    :func:`~src.background_knowledge.sample_background_knowledge_v2`, which
    draws required edges uniformly from all true edges in the DAG.

    Part of the **b-LOAD** experimental pipeline.

    Args:
        seed (int): Random seed.
        observed (int): Number of observed nodes.
        exp_degree (float): Expected graph degree.
        max_degree (int): Maximum node degree.
        targets (int): Number of target pairs.
        expl_anc (bool): Require explicit ancestor–descendant pair.
        identifiable (bool): Require identifiable effect.
        min_adj_size (int): Minimum adjustment-set size.
        samples_num (int): Number of observations.
        discrete (bool): Use discrete data.
        save (bool): Cache data to disk.
        with_b (bool): Run b-LOAD if ``True``, base LOAD if ``False``.
        logging (bool): Verbose output.
        fraction_forbidden (float): Fraction of non-edges to forbid.
        fraction_required (float): Fraction of true edges to require.
        alpha (float): CI test significance level.
        ci_test (str): CI test name.

    Returns:
        dict: Result dict from :func:`run_algorithm_mpdag`.
    """

    # Generate the synthetic data
    data = generate_data(
        seed=seed,
        observed=observed,        # Slightly more nodes to ensure a richer MB
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,   # More samples for more reliable Fisher-Z tests
        expl_anc=expl_anc,
        discrete=discrete,
        save=save
    )


    # Extract components
    true_dag = data["true_dag"]
    cpt = data["cpt"]
    targets = data["targets"]


    # Sample background knowledge (e.g., 20% of edges)
    bg_knowledge = sample_background_knowledge_v2(true_dag, fraction_required, fraction_forbidden, seed)

    initial_G = initialize_background_knowledge(num_nodes=observed, bk_dict=bg_knowledge)

    return run_algorithm_mpdag(
        ci_test=ci_test,
        alpha=alpha,
        with_b=with_b,
        bg_knowledge=initial_G,
        logging=logging,
        **data
    )


def run_experiment_mpdag_v2(base_args, base_seed, n_exp):

  experiments_with_b = {}
  experiments_without_b = {}

  for s in tqdm(range(n_exp)):
    exp_with_b = get_experiment_mpdag_v2(**base_args, seed=base_seed+s, with_b=True)
    experiments_with_b[exp_with_b["id"]] = exp_with_b

    exp_without_b = get_experiment_mpdag_v2(**base_args, seed=base_seed+s, with_b=False)
    experiments_without_b[exp_without_b["id"]] = exp_without_b

  # Global metrics
  true_osets_with_b = get_true_osets(experiments_with_b)
  f1_with_b = np.sort(evaluate_oset("load", list(experiments_with_b.values()), true_osets_with_b)[2])

  true_osets_without_b = get_true_osets(experiments_without_b)
  f1_without_b = np.sort(evaluate_oset("load", list(experiments_without_b.values()), true_osets_without_b)[2])

  # Intervention metrics
  family = "binary" if base_args.get("discrete", False) else "gaussian"
  true_effects = true_causal_effects(experiments_with_b, family=family)

  int_dist_with_b = evaluate_intervention("load", list(experiments_with_b.values()), true_effects, family=family)
  int_dist_without_b = evaluate_intervention("load", list(experiments_without_b.values()), true_effects, family=family)

  f1_dict = {
      "f1_with_b": {"mean":f1_with_b.mean(), "std": f1_with_b.std()},
      "f1_without_b": {"mean":f1_without_b.mean(), "std": f1_without_b.std()},
      "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
      "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
  }

  return f1_dict


def run_experiment_mpdag_v2_light(base_args, base_seed, n_exp):
    # Lighter version that doesn't store all experiments in memory (only the final metrics), to save memory when running many experiments

    f1_with_b_list = []
    f1_without_b_list = []
    int_dist_with_b_list = []
    int_dist_without_b_list = []
    family = "binary" if base_args.get("discrete", False) else "gaussian"

    for s in tqdm(range(n_exp)):
        exp_with_b = get_experiment_mpdag_v2(**base_args, seed=base_seed+s, with_b=True)
        exp_without_b = get_experiment_mpdag_v2(**base_args, seed=base_seed+s, with_b=False)

        # Evaluate immediately, store only scalars
        true_osets_with_b = get_true_osets({exp_with_b["id"]: exp_with_b})
        f1_with_b_list.append(evaluate_oset("load", [exp_with_b], true_osets_with_b)[2])

        true_osets_without_b = get_true_osets({exp_without_b["id"]: exp_without_b})
        f1_without_b_list.append(evaluate_oset("load", [exp_without_b], true_osets_without_b)[2])

        true_effects = true_causal_effects({exp_with_b["id"]: exp_with_b}, family=family)
        int_dist_with_b_list.append(evaluate_intervention("load", [exp_with_b], true_effects, family=family))
        int_dist_without_b_list.append(evaluate_intervention("load", [exp_without_b], true_effects, family=family))

        # Explicitly discard large objects
        del exp_with_b, exp_without_b

    f1_with_b = np.sort(np.concatenate(f1_with_b_list))
    f1_without_b = np.sort(np.concatenate(f1_without_b_list))
    int_dist_with_b = np.concatenate(int_dist_with_b_list)
    int_dist_without_b = np.concatenate(int_dist_without_b_list)

    return {
        "f1_with_b": {"mean": f1_with_b.mean(), "std": f1_with_b.std()},
        "f1_without_b": {"mean": f1_without_b.mean(), "std": f1_without_b.std()},
        "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
        "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
    }


### Real data (sachs)


def get_experiment_mpdag_real(
    data_name: str,
    seed: int,
    real_data: np.ndarray,
    true_dag: nx.DiGraph,
    targets: list[int],
    observed: int,
    with_b: bool,
    fraction_forbidden: float = 0.1,
    fraction_required: float = 0.1,
    alpha: float = 0.01,
    ci_test: str = "fisherz",
    sampling_strategy: str = "local",
) -> dict:
    """
    Run a single LOAD or b-LOAD trial on real data for one target pair.

    Background knowledge is sampled from the true DAG using either the local
    strategy (edges incident on ``targets``) or the global strategy (all
    edges), then converted to an adjacency matrix and passed to
    :func:`run_algorithm_mpdag`.

    Part of the **b-LOAD** experimental pipeline.

    Args:
        data_name (str): Dataset identifier used to build the experiment ID.
        seed (int): Random seed for BK sampling.
        real_data (np.ndarray): Observational data matrix (n_samples × n_nodes).
        true_dag (nx.DiGraph): Ground-truth DAG for BK sampling.
        targets (list[int]): ``[treatment, outcome]`` 0-based indices.
        observed (int): Total number of nodes (columns of ``real_data``).
        with_b (bool): Run b-LOAD if ``True``, base LOAD if ``False``.
        fraction_forbidden (float): Fraction of non-edges to forbid.
        fraction_required (float): Fraction of true edges to require.
        alpha (float): CI test significance level.
        ci_test (str): CI test name.
        sampling_strategy (str): ``"local"`` or ``"global"``.

    Returns:
        dict: Result dict from :func:`run_algorithm_mpdag`.
    """


    if sampling_strategy == "local":
        bg_knowledge = sample_local_background_knowledge(true_dag, targets, fraction_required, fraction_forbidden, seed)
    elif sampling_strategy == "global":
        bg_knowledge = sample_background_knowledge_v2(true_dag, fraction_required, fraction_forbidden, seed)
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
    
    initial_G = initialize_background_knowledge(num_nodes=observed, bk_dict=bg_knowledge)
    id = f"{data_name}_{seed}"

    return run_algorithm_mpdag(
        id=id,
        targets=targets,
        data=real_data,
        ci_test=ci_test,
        alpha=alpha,
        with_b=with_b,
        bg_knowledge=initial_G,
        true_dag=true_dag,
    )


def run_experiment_mpdag_real(base_args, base_seed, pairs_to_test):
    """
    Run LOAD, b-LOAD, PC, and b-PC on every target pair and return aggregate metrics.

    All four algorithms share the same background-knowledge sample for a given pair
    and seed, so comparisons are fair.  The result_dict contains mean/std for each
    of the 16 metric×algorithm combinations (4 metrics × 4 algorithms).
    """
    experiments_with_b    = {}   # b-LOAD
    experiments_without_b = {}   # LOAD
    experiments_pc        = {}   # PC  (no BK)
    experiments_bpc       = {}   # b-PC (same BK as b-LOAD)

    data_name          = base_args.get("data_name", "real")
    real_data          = base_args["real_data"]
    true_dag           = base_args["true_dag"]
    observed           = base_args["observed"]
    alpha              = base_args.get("alpha", 0.01)
    ci_test            = base_args.get("ci_test", "fisherz")
    sampling_strategy  = base_args.get("sampling_strategy", "local")
    fraction_required  = base_args.get("fraction_required", 0.1)
    fraction_forbidden = base_args.get("fraction_forbidden", 0.0)
    family             = "binary" if base_args.get("discrete", False) else "gaussian"

    for s, pair in enumerate(tqdm(pairs_to_test)):
        seed   = base_seed + s
        exp_id = f"{data_name}_{seed}"

        # ── Sample BK once; reused by b-LOAD and b-PC ──────────────────────
        if sampling_strategy == "local":
            bk_dict = sample_local_background_knowledge(
                true_dag, pair, fraction_required, fraction_forbidden, seed
            )
        elif sampling_strategy == "global":
            bk_dict = sample_background_knowledge_v2(
                true_dag, fraction_required, fraction_forbidden, seed
            )
        else:
            raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
        bk_matrix = initialize_background_knowledge(num_nodes=observed, bk_dict=bk_dict)

        # ── b-LOAD ─────────────────────────────────────────────────────────
        experiments_with_b[exp_id] = run_algorithm_mpdag(
            id=exp_id, targets=pair, data=real_data,
            ci_test=ci_test, alpha=alpha,
            with_b=True, bg_knowledge=bk_matrix, true_dag=true_dag,
        )

        # ── LOAD (no BK) ───────────────────────────────────────────────────
        experiments_without_b[exp_id] = run_algorithm_mpdag(
            id=exp_id, targets=pair, data=real_data,
            ci_test=ci_test, alpha=alpha,
            with_b=False, bg_knowledge=bk_matrix, true_dag=true_dag,
        )

        # ── PC (no BK) ─────────────────────────────────────────────────────
        experiments_pc[exp_id] = run_algorithm_pc(
            id=exp_id, targets=pair, data=real_data,
            ci_test=ci_test, alpha=alpha, true_dag=true_dag,
        )

        # ── b-PC (same BK as b-LOAD) ───────────────────────────────────────
        experiments_bpc[exp_id] = run_algorithm_bpc(
            id=exp_id, targets=pair, data=real_data,
            ci_test=ci_test, alpha=alpha,
            bg_knowledge=bk_matrix, true_dag=true_dag,
        )

    # ── O-set quality ──────────────────────────────────────────────────────
    true_osets_with_b    = get_true_osets_real_data(experiments_with_b)
    true_osets_without_b = get_true_osets_real_data(experiments_without_b)
    true_osets_pc        = get_true_osets_real_data(experiments_pc)
    true_osets_bpc       = get_true_osets_real_data(experiments_bpc)

    res_with    = evaluate_oset("load", list(experiments_with_b.values()),    true_osets_with_b)
    res_without = evaluate_oset("load", list(experiments_without_b.values()), true_osets_without_b)
    res_pc      = evaluate_oset("pc",   list(experiments_pc.values()),         true_osets_pc)
    res_bpc     = evaluate_oset("pc",   list(experiments_bpc.values()),        true_osets_bpc)

    prec_with_b,    rec_with_b,    f1_with_b    = [np.array(x) for x in res_with]
    prec_without_b, rec_without_b, f1_without_b = [np.array(x) for x in res_without]
    prec_pc,        rec_pc,        f1_pc        = [np.array(x) for x in res_pc]
    prec_bpc,       rec_bpc,       f1_bpc       = [np.array(x) for x in res_bpc]

    # ── Intervention distance ───────────────────────────────────────────────
    true_effects = true_causal_effects(experiments_with_b, family=family)

    int_dist_with_b    = evaluate_intervention("load", list(experiments_with_b.values()),    true_effects, family=family)
    int_dist_without_b = evaluate_intervention("load", list(experiments_without_b.values()), true_effects, family=family)
    int_dist_pc        = evaluate_intervention("pc",   list(experiments_pc.values()),         true_effects, family=family)
    int_dist_bpc       = evaluate_intervention("pc",   list(experiments_bpc.values()),        true_effects, family=family)

    result_dict = {
        # b-LOAD
        "precision_with_b":    {"mean": float(prec_with_b.mean()),       "std": float(prec_with_b.std())},
        "recall_with_b":       {"mean": float(rec_with_b.mean()),        "std": float(rec_with_b.std())},
        "f1_with_b":           {"mean": float(f1_with_b.mean()),         "std": float(f1_with_b.std())},
        "int_dist_with_b":     {"mean": float(int_dist_with_b.mean()),   "std": float(int_dist_with_b.std())},
        # LOAD
        "precision_without_b": {"mean": float(prec_without_b.mean()),    "std": float(prec_without_b.std())},
        "recall_without_b":    {"mean": float(rec_without_b.mean()),     "std": float(rec_without_b.std())},
        "f1_without_b":        {"mean": float(f1_without_b.mean()),      "std": float(f1_without_b.std())},
        "int_dist_without_b":  {"mean": float(int_dist_without_b.mean()),"std": float(int_dist_without_b.std())},
        # PC
        "precision_pc":        {"mean": float(prec_pc.mean()),           "std": float(prec_pc.std())},
        "recall_pc":           {"mean": float(rec_pc.mean()),            "std": float(rec_pc.std())},
        "f1_pc":               {"mean": float(f1_pc.mean()),             "std": float(f1_pc.std())},
        "int_dist_pc":         {"mean": float(int_dist_pc.mean()),       "std": float(int_dist_pc.std())},
        # b-PC
        "precision_bpc":       {"mean": float(prec_bpc.mean()),          "std": float(prec_bpc.std())},
        "recall_bpc":          {"mean": float(rec_bpc.mean()),           "std": float(rec_bpc.std())},
        "f1_bpc":              {"mean": float(f1_bpc.mean()),            "std": float(f1_bpc.std())},
        "int_dist_bpc":        {"mean": float(int_dist_bpc.mean()),      "std": float(int_dist_bpc.std())},
    }

    detailed_dict = {
        "with_b":    {"precision": prec_with_b,    "recall": rec_with_b,    "f1": f1_with_b},
        "without_b": {"precision": prec_without_b, "recall": rec_without_b, "f1": f1_without_b},
        "pc":        {"precision": prec_pc,         "recall": rec_pc,        "f1": f1_pc},
        "bpc":       {"precision": prec_bpc,        "recall": rec_bpc,       "f1": f1_bpc},
    }

    return result_dict, detailed_dict


### Noised: testing robustness to noise in the background knowledge (sampling around target nodes, but with noise)


def get_experiment_mpdag_noised(
      seed: int,
      observed: int,
      exp_degree: float,
      max_degree: int,
      targets: int,
      expl_anc: bool,
      identifiable: bool,
      min_adj_size: int,
      samples_num,
      discrete: bool,
      save: bool,
      with_b: bool,
      logging: bool,
      fraction_forbidden: float = 0.1,
      fraction_required: float = 0.1,
      false_required_ratio: float = 0.5,
      alpha: float = 0.01,
      ci_test: str = "fisherz",
    ) -> dict:
    """
    Run a single experiment with a specific seed.

    Args:
        args (Namespace): The command line arguments.
        done (dict): A dictionary of completed experiments.
        s (int): The seed for the experiment.

    Returns:
        dict: The result of the experiment.
    """

    # Generate the synthetic data
    data = generate_data(
        seed=seed,
        observed=observed,        # Slightly more nodes to ensure a richer MB
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,   # More samples for more reliable Fisher-Z tests
        expl_anc=expl_anc,
        discrete=discrete,
        save=save
    )


    # Extract components
    true_dag = data["true_dag"]
    cpt = data["cpt"]
    targets = data["targets"]


    # Sample background knowledge (e.g., 20% of edges)
    bg_knowledge = sample_local_background_knowledge_noised(
        true_dag=true_dag,
        targets=targets,
        fraction_required=fraction_required,
        fraction_forbidden=fraction_forbidden,
        noise_mu=false_required_ratio,
        seed=seed
    )

    initial_G = initialize_background_knowledge(num_nodes=observed, bk_dict=bg_knowledge)

    return run_algorithm_mpdag(
        ci_test=ci_test,
        alpha=alpha,
        with_b=with_b,
        bg_knowledge=initial_G,
        logging=logging,
        **data
    )


def run_experiment_mpdag_noised(base_args, base_seed, n_exp):

  experiments_with_b = {}
  experiments_without_b = {}

  for s in tqdm(range(n_exp)):
    exp_with_b = get_experiment_mpdag_noised(**base_args, seed=base_seed+s, with_b=True)
    experiments_with_b[exp_with_b["id"]] = exp_with_b

    exp_without_b = get_experiment_mpdag_noised(**base_args, seed=base_seed+s, with_b=False)
    experiments_without_b[exp_without_b["id"]] = exp_without_b

  # Global metrics
  true_osets_with_b = get_true_osets(experiments_with_b)
  f1_with_b = np.sort(evaluate_oset("load", list(experiments_with_b.values()), true_osets_with_b)[2])

  true_osets_without_b = get_true_osets(experiments_without_b)
  f1_without_b = np.sort(evaluate_oset("load", list(experiments_without_b.values()), true_osets_without_b)[2])

  # Intervention metrics
  family = "binary" if base_args.get("discrete", False) else "gaussian"
  true_effects = true_causal_effects(experiments_with_b, family=family)

  int_dist_with_b = evaluate_intervention("load", list(experiments_with_b.values()), true_effects, family=family)
  int_dist_without_b = evaluate_intervention("load", list(experiments_without_b.values()), true_effects, family=family)

  f1_dict = {
      "f1_with_b": {"mean":f1_with_b.mean(), "std": f1_with_b.std()},
      "f1_without_b": {"mean":f1_without_b.mean(), "std": f1_without_b.std()},
      "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
      "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
  }

  return f1_dict


def run_experiment_mpdag_noised_light(base_args, base_seed, n_exp):
    # Lighter version that doesn't store all experiments in memory (only the final metrics), to save memory when running many experiments

    f1_with_b_list = []
    f1_without_b_list = []
    int_dist_with_b_list = []
    int_dist_without_b_list = []
    family = "binary" if base_args.get("discrete", False) else "gaussian"

    for s in tqdm(range(n_exp)):
        exp_with_b = get_experiment_mpdag_noised(**base_args, seed=base_seed+s, with_b=True)
        exp_without_b = get_experiment_mpdag_noised(**base_args, seed=base_seed+s, with_b=False)

        # Evaluate immediately, store only scalars
        true_osets_with_b = get_true_osets({exp_with_b["id"]: exp_with_b})
        f1_with_b_list.append(evaluate_oset("load", [exp_with_b], true_osets_with_b)[2])

        true_osets_without_b = get_true_osets({exp_without_b["id"]: exp_without_b})
        f1_without_b_list.append(evaluate_oset("load", [exp_without_b], true_osets_without_b)[2])

        true_effects = true_causal_effects({exp_with_b["id"]: exp_with_b}, family=family)
        int_dist_with_b_list.append(evaluate_intervention("load", [exp_with_b], true_effects, family=family))
        int_dist_without_b_list.append(evaluate_intervention("load", [exp_without_b], true_effects, family=family))

        # Explicitly discard large objects
        del exp_with_b, exp_without_b

    f1_with_b = np.sort(np.concatenate(f1_with_b_list))
    f1_without_b = np.sort(np.concatenate(f1_without_b_list))
    int_dist_with_b = np.concatenate(int_dist_with_b_list)
    int_dist_without_b = np.concatenate(int_dist_without_b_list)

    return {
        "f1_with_b": {"mean": f1_with_b.mean(), "std": f1_with_b.std()},
        "f1_without_b": {"mean": f1_without_b.mean(), "std": f1_without_b.std()},
        "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
        "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
    }

# ---------------------------------------------------------------------------
# PC / b-PC baselines
# ---------------------------------------------------------------------------


def run_algorithm_pc(
    id: str,
    data: np.ndarray,
    targets: list[int],
    ci_test: str,
    alpha: float,
    true_dag: nx.Optional[DiGraph] = None,
    cpt: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    Run the PC algorithm and return a result dict compatible with evaluate_oset
    and evaluate_intervention.

    Args:
        id (str): Experiment identifier.
        data (np.ndarray): Data matrix.
        targets (list[int]): Target node pair [treatment, outcome].
        ci_test (str): CI test name (e.g. "fisherz").
        alpha (float): Significance level.
        true_dag (nx.DiGraph): Ground-truth DAG (stored for evaluation only).
        cpt: Conditional probability table (stored for evaluation only).
        **kwargs: Extra fields from generate_data (e.g. treatment, outcome) are
            forwarded to CountingTest / CIT and otherwise ignored.

    Returns:
        dict with keys: amat, time, id, targets, tests, cpt, true_dag, data.
    """
    ci_test_fn = CountingTest(data, ci_test, **kwargs)

    start = perf_counter()
    result = pc(data=data, ci_test=ci_test_fn, alpha=alpha)
    result["time"] = perf_counter() - start

    result["id"] = id
    result["targets"] = targets
    result["tests"] = ci_test_fn.get_tests_per_order().tolist()
    result["cpt"] = cpt
    result["true_dag"] = true_dag
    result["data"] = data

    return result


def get_experiment_pc(
    seed: int,
    observed: int,
    exp_degree: float,
    max_degree: int,
    targets: int,
    expl_anc: bool,
    identifiable: bool,
    min_adj_size: int,
    samples_num,
    discrete: bool,
    save: bool,
    alpha: float = 0.01,
    ci_test: str = "fisherz",
    **kwargs,
) -> dict:
    """
    Generate one synthetic experiment and run PC on it.

    Extra kwargs (e.g. fraction_required, fraction_forbidden, logging) are
    silently ignored so that the same base_args dict used for LOAD experiments
    can be passed here without modification.
    """
    data = generate_data(
        seed=seed,
        observed=observed,
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,
        expl_anc=expl_anc,
        discrete=discrete,
        save=save,
    )

    return run_algorithm_pc(
        ci_test=ci_test,
        alpha=alpha,
        **data,
    )


def run_experiment_pc(base_args, base_seed, n_exp):
    """
    Run n_exp PC experiments and return aggregate metrics.
    Holds all experiment results in memory; use run_experiment_pc_light for
    large runs.
    """
    experiments = {}

    for s in tqdm(range(n_exp)):
        exp = get_experiment_pc(**base_args, seed=base_seed + s)
        experiments[exp["id"]] = exp

    true_osets = get_true_osets(experiments)
    prec, rec, f1 = evaluate_oset("pc", list(experiments.values()), true_osets)
    f1 = np.sort(np.array(f1))

    family = "binary" if base_args.get("discrete", False) else "gaussian"
    true_effects = true_causal_effects(experiments, family=family)
    int_dist = evaluate_intervention("pc", list(experiments.values()), true_effects, family=family)

    return {
        "f1": {"mean": float(f1.mean()), "std": float(f1.std())},
        "int_dist": {"mean": float(int_dist.mean()), "std": float(int_dist.std())},
    }


def run_experiment_pc_light(base_args, base_seed, n_exp):
    """
    Memory-efficient variant: evaluates each experiment immediately and discards it,
    accumulating only scalar metrics.
    """
    f1_list = []
    int_dist_list = []
    family = "binary" if base_args.get("discrete", False) else "gaussian"

    for s in tqdm(range(n_exp)):
        exp = get_experiment_pc(**base_args, seed=base_seed + s)

        true_osets = get_true_osets({exp["id"]: exp})
        f1_list.append(evaluate_oset("pc", [exp], true_osets)[2])

        true_effects = true_causal_effects({exp["id"]: exp}, family=family)
        int_dist_list.append(
            evaluate_intervention("pc", [exp], true_effects, family=family)
        )

        del exp

    f1 = np.sort(np.concatenate(f1_list))
    int_dist = np.concatenate(int_dist_list)

    return {
        "f1": {"mean": float(f1.mean()), "std": float(f1.std())},
        "int_dist": {"mean": float(int_dist.mean()), "std": float(int_dist.std())},
    }


# b-PC (PC with background knowledge)


def run_algorithm_bpc(
    id: str,
    data: np.ndarray,
    targets: list[int],
    ci_test: str,
    alpha: float,
    bg_knowledge: np.Optional[ndarray] = None,
    true_dag: nx.Optional[DiGraph] = None,
    cpt: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    Run PC with background knowledge and return a result dict compatible
    with evaluate_oset and evaluate_intervention.

    Args:
        bg_knowledge (np.Optional[ndarray]): Required-edge matrix from
            initialize_background_knowledge. Passed directly to pc_with_bk.
        **kwargs: Extra fields from generate_data forwarded to CountingTest.

    Returns:
        dict with keys: amat, time, id, targets, tests, cpt, true_dag, data.
    """
    ci_test_fn = CountingTest(data, ci_test, **kwargs)

    start = perf_counter()
    result = pc_with_bk(
        data=data,
        ci_test=ci_test_fn,
        alpha=alpha,
        background_knowledge=bg_knowledge,
    )
    result["time"] = perf_counter() - start

    result["id"] = id
    result["targets"] = targets
    result["tests"] = ci_test_fn.get_tests_per_order().tolist()
    result["cpt"] = cpt
    result["true_dag"] = true_dag
    result["data"] = data

    return result


def get_experiment_bpc(
    seed: int,
    observed: int,
    exp_degree: float,
    max_degree: int,
    targets: int,
    expl_anc: bool,
    identifiable: bool,
    min_adj_size: int,
    samples_num,
    discrete: bool,
    save: bool,
    fraction_required: float = 0.1,
    fraction_forbidden: float = 0.0,
    alpha: float = 0.01,
    ci_test: str = "fisherz",
    **kwargs,
) -> dict:
    """
    Generate one synthetic experiment and run PC+BK on it (local BK sampling).

    fraction_forbidden is accepted for interface compatibility but ignored;
    only required edges are used. Extra kwargs (e.g. logging) are silently
    dropped so the same base_args dict used for LOAD experiments works here.
    """
    data = generate_data(
        seed=seed,
        observed=observed,
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,
        expl_anc=expl_anc,
        discrete=discrete,
        save=save,
    )

    bg_knowledge = sample_local_background_knowledge(
        data["true_dag"], data["targets"], fraction_required, 0.0, seed
    )
    initial_G = initialize_background_knowledge(
        num_nodes=observed, bk_dict=bg_knowledge
    )

    return run_algorithm_bpc(
        bg_knowledge=initial_G,
        ci_test=ci_test,
        alpha=alpha,
        **data,
    )


def get_experiment_bpc_v2(
    seed: int,
    observed: int,
    exp_degree: float,
    max_degree: int,
    targets: int,
    expl_anc: bool,
    identifiable: bool,
    min_adj_size: int,
    samples_num,
    discrete: bool,
    save: bool,
    fraction_required: float = 0.1,
    fraction_forbidden: float = 0.0,
    alpha: float = 0.01,
    ci_test: str = "fisherz",
    **kwargs,
) -> dict:
    """
    Generate one synthetic experiment and run PC+BK on it (global BK sampling).
    """
    data = generate_data(
        seed=seed,
        observed=observed,
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,
        expl_anc=expl_anc,
        discrete=discrete,
        save=save,
    )

    bg_knowledge = sample_background_knowledge_v2(
        data["true_dag"], fraction_required, 0.0, seed
    )
    initial_G = initialize_background_knowledge(
        num_nodes=observed, bk_dict=bg_knowledge
    )

    return run_algorithm_bpc(
        bg_knowledge=initial_G,
        ci_test=ci_test,
        alpha=alpha,
        **data,
    )


def run_experiment_bpc_light(base_args, base_seed, n_exp):
    """
    Memory-efficient loop for PC+BK with local background knowledge sampling.
    """
    f1_list = []
    int_dist_list = []
    family = "binary" if base_args.get("discrete", False) else "gaussian"

    for s in tqdm(range(n_exp)):
        exp = get_experiment_bpc(**base_args, seed=base_seed + s)

        true_osets = get_true_osets({exp["id"]: exp})
        f1_list.append(evaluate_oset("pc", [exp], true_osets)[2])

        true_effects = true_causal_effects({exp["id"]: exp}, family=family)
        int_dist_list.append(
            evaluate_intervention("pc", [exp], true_effects, family=family)
        )

        del exp

    f1 = np.sort(np.concatenate(f1_list))
    int_dist = np.concatenate(int_dist_list)

    return {
        "f1": {"mean": float(f1.mean()), "std": float(f1.std())},
        "int_dist": {"mean": float(int_dist.mean()), "std": float(int_dist.std())},
    }


def run_experiment_bpc_v2_light(base_args, base_seed, n_exp):
    """
    Memory-efficient loop for PC+BK with global background knowledge sampling.
    """
    f1_list = []
    int_dist_list = []
    family = "binary" if base_args.get("discrete", False) else "gaussian"

    for s in tqdm(range(n_exp)):
        exp = get_experiment_bpc_v2(**base_args, seed=base_seed + s)

        true_osets = get_true_osets({exp["id"]: exp})
        f1_list.append(evaluate_oset("pc", [exp], true_osets)[2])

        true_effects = true_causal_effects({exp["id"]: exp}, family=family)
        int_dist_list.append(
            evaluate_intervention("pc", [exp], true_effects, family=family)
        )

        del exp

    f1 = np.sort(np.concatenate(f1_list))
    int_dist = np.concatenate(int_dist_list)

    return {
        "f1": {"mean": float(f1.mean()), "std": float(f1.std())},
        "int_dist": {"mean": float(int_dist.mean()), "std": float(int_dist.std())},
    }
