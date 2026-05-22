# b-LOAD: Background-Knowledge-Augmented Local Causal Discovery

This repository implements **b-LOAD** (*Background-knowledge LOAD*), an extension of the Local Ordering-based Ancestral Discovery (LOAD) algorithm that incorporates prior causal constraints — required directed edges — into the local structure-learning loop.  The result is a Maximal Partially Directed Acyclic Graph (MPDAG) that is consistent with both the observational data and the provided background knowledge, from which optimal adjustment sets for causal effect estimation are identified locally.

A **b-PC** baseline (PC algorithm with background-knowledge injection and four-rule Meek propagation) is also provided for comparison.

---

## Repository Structure

```
.
├── src/                            # Core Python library
│   ├── mb_by_mb.py                 # MB-by-MB skeleton: LOAD (base) + b-LOAD inner loop
│   ├── load.py                     # LOAD and b-LOAD top-level causal effect estimators
│   ├── pc.py                       # PC and b-PC global structure learners
│   ├── background_knowledge.py     # BK sampling and adjacency-matrix initialisation
│   ├── counting_ci_tests.py        # CI test wrapper with per-order test counting
│   ├── evaluation.py               # Precision / recall / F1 / intervention distance
│   ├── running.py                  # End-to-end experiment runners
│   ├── data_generation.py          # Synthetic data generation via R
│   ├── analysis.py                 # Result loading, aggregation, and plotting utilities
│   ├── utils.py                    # Graph visualisation helpers
│   └── real_data_applications/
│       ├── sachs.py                # Sachs protein-signalling dataset helpers
│       └── dream.py                # DREAM4 / DREAM5 benchmark helpers
│
├── experiment_scripts/             # Entry-point scripts for each experimental condition
│   ├── run_load_local_bk.py        # LOAD vs b-LOAD — locally sampled BK
│   ├── run_load_global_bk.py       # LOAD vs b-LOAD — globally sampled BK
│   ├── run_load_noised.py          # b-LOAD with noised (partially incorrect) BK
│   ├── run_pc.py                   # PC baseline (no BK)
│   ├── run_bpc_local_bk.py         # b-PC — locally sampled BK
│   └── run_bpc_global_bk.py        # b-PC — globally sampled BK
│
├── generate_data.R                 # R functions for random DAG + SEM generation
├── evaluate.R                      # R functions for CPDAG conversion and optAdjSet
└── requirements.txt
```

---

## Algorithms

### LOAD (base)
The base LOAD algorithm performs local causal structure learning around a target node using a Markov-blanket-by-Markov-blanket (MB-by-MB) expansion.  It identifies the optimal adjustment set (o-set) for a treatment–outcome pair without reconstructing the full global graph.  The core MB-by-MB implementation in `src/mb_by_mb.py` and the top-level estimation logic in `src/load.py` are adapted from the original LOAD authors' public implementation.

### b-LOAD (this work)
b-LOAD extends LOAD with three modifications to the MB-by-MB loop:

1. **BK initialisation** — the growing graph is seeded with required background-knowledge edges so that known causal directions are present from the first iteration.
2. **BK-protected graph update** — after each local structure is merged, required edges are re-stamped to prevent CI-test errors from erasing known constraints.
3. **Four-rule MPDAG Meek** — a fourth Meek orientation rule (using the sep-set from skeleton search) propagates orientations induced by BK edges that the three structural rules alone cannot deduce.

### b-PC (baseline)
b-PC runs the standard PC algorithm to produce a CPDAG, injects required background-knowledge edges, and then applies four-rule Meek to propagate the induced orientations.

---

## Graph Encoding

Throughout `src/`, adjacency matrices use the following internal convention:

| `g[i, j]` | `g[j, i]` | Meaning             |
|-----------|-----------|---------------------|
| `-1`      | `-1`      | Undirected edge i — j |
| `-1`      | `+1`      | Directed edge i → j |
| `0`       | `0`       | No edge             |

Background knowledge is encoded identically: `bk[i,j] = -1` and `bk[j,i] = 1` for a required edge i → j.

The final `"amat"` output (returned by `pc` and `pc_with_bk`) follows the `pcalg` convention: `amat[i][j] = 1` indicates a tail at i (edge from i toward j, directed or undirected).

---

## Getting Started

### Requirements

Python dependencies:
```bash
pip install -r requirements.txt
```

R dependencies (required for data generation and CPDAG/optAdjSet computation):
```r
install.packages(c("pcalg", "graph", "RBGL"))
```

The R scripts `generate_data.R` and `evaluate.R` are sourced automatically when `src/data_generation.py` and `src/evaluation.py` are imported.

### Running Experiments

Each script in `experiment_scripts/` accepts a `--save_path` argument and optional graph/sample-size parameters.  Example usage:

```bash
# b-LOAD vs LOAD with locally sampled background knowledge
python experiment_scripts/run_load_local_bk.py --save_path experiment_runs/load_local.csv

# b-PC with globally sampled background knowledge
python experiment_scripts/run_bpc_global_bk.py --save_path experiment_runs/bpc_global.csv

# PC baseline (no background knowledge)
python experiment_scripts/run_pc.py --save_path experiment_runs/pc.csv
```

> **Note:** Output files are written to `experiment_runs/` by default.  Create this directory locally if needed — it is excluded from version control.

### Background Knowledge Sampling Strategies

Two strategies are available (controlled by the `sampling_strategy` argument in experiment runners):

- **`"local"`** — only edges incident on the treatment or outcome nodes are sampled as required BK.  This is the primary strategy evaluated in the paper.
- **`"global"`** — required edges are drawn uniformly from all true edges in the DAG.

The fraction of edges revealed is controlled by `fraction_required` (0–1).

### Real-Data Experiments

Real-data runners for the Sachs protein-signalling and DREAM4/DREAM5 benchmarks are provided in `src/real_data_applications/`.  Place the data files under `experiments/dream4/` and `experiments/dream5/` respectively, then call:

```python
from src.real_data_applications.dream import run_experiment_dream4, run_experiment_dream5
from src.real_data_applications.sachs import get_sachs_ground_truth_dag, fit_oracle_weights

# DREAM4 network 1, local BK, 30 % of true edges revealed
result_dict, details_dict = run_experiment_dream4(subset_number=1, sampling_strategy="local", frac_req=0.3)
```

### Analysing Results

`src/analysis.py` provides utilities for loading result CSVs, computing the complexity metric π = n_samples / (n_nodes × expected_degree / 2), aggregating into regime tables, and producing F1 and intervention-distance plots:

```python
from src.analysis import load_iter_folder, add_complexity_metric, regime_table_3, plot_f1

df_bload = load_iter_folder("experiment_runs/load_local_*", stat_col_pairs=[...])
df_bload = add_complexity_metric(df_bload)
fig = plot_f1(df_pc, df_bpc, df_bpc_global, df_load, df_bload_global)
```

---

## Citation

If you use this code, please cite:

```
[Citation placeholder — to be filled upon publication]
```
