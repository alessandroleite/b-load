<h1 align="center">b-LOAD: Background-Knowledge-Augmented Local Causal Discovery
</h1>

<p align="center">
<strong>b-Load</strong> is a knowledge-informed extension of the Local Ordering-based Ancestral Discovery (LOAD) algorithm for local causal discovery of optimal adjustment sets. It integrates prior edge constraints into local structure learning and dynamically expands the discovery frontier using Meek's rules, producing knowledge-constrained maximal partially directed acyclic graphs. By leveraging background knowledge, <strong>b-Load</strong> reduces structural uncertainty and enables the identification of adjustment sets that may not be identifiable from observational data alone. The method improves causal effect estimation, particularly in data-scarce and structurally complex settings, while maintaining scalability and robustness to moderate noise in prior knowledge.
</p>

<p align="center">
  <a href="#theproblem">Problem</a> &bull;
  <a href="#install">Install</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#results">Results</a> &bull;
  <a href="#reproducing-the-paper">Paper</a>
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3799682.3840893"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1145%2F3799682.3840893-1f6feb"></a>
  <a href="https://pypi.org/project/b-load/"><img alt="PyPI" src="https://img.shields.io/badge/pyversion-b-load-blue"></a>
  <a href="https://pypi.org/project/b-load/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/b-load"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## The Problem

Local causal discovery methods such as LOAD can recover adjustment sets without learning the full global graph, but they operate under uncertainty and may miss identifiable causal directions that are already known from domain knowledge. This project addresses that limitation by augmenting the local search with background knowledge and Meek-rule propagation.

### LOAD (base)
The base LOAD algorithm performs local causal structure learning around a target node using a Markov-blanket-by-Markov-blanket (MB-by-MB) expansion. It identifies the optimal adjustment set (o-set) for a treatment–outcome pair without reconstructing the full global graph. The core MB-by-MB implementation in `src/mb_by_mb.py` and the top-level estimation logic in `src/load.py` are adapted from the original LOAD authors' public implementation. The public package exposes these components via `bload.mb_by_mb` and `bload.load`.

### b-LOAD (this work)
b-LOAD extends LOAD with three key modifications to the MB-by-MB loop:

1. **BK initialisation** — the growing graph is seeded with required background-knowledge edges so that known causal directions are present from the first iteration.
2. **BK-protected graph update** — after each local structure is merged, required edges are re-stamped to prevent CI-test errors from erasing known constraints.
3. **Four-rule MPDAG Meek** — a fourth Meek orientation rule (using the sep-set from skeleton search) propagates orientations induced by BK edges that the three structural rules alone cannot deduce.

### b-PC (baseline)
b-PC runs the standard PC algorithm to produce a CPDAG, injects required background-knowledge edges, and then applies four-rule Meek to propagate the induced orientations.

### Graph encoding

Throughout the implementation, adjacency matrices use the following internal convention:

| `g[i, j]` | `g[j, i]` | Meaning |
|-----------|-----------|---------|
| `-1` | `-1` | Undirected edge i — j |
| `-1` | `+1` | Directed edge i → j |
| `0` | `0` | No edge |

Background knowledge is encoded identically: `bk[i,j] = -1` and `bk[j,i] = 1` for a required edge i → j.

The final `"amat"` output (returned by `pc` and `pc_with_bk`) follows the `pcalg` convention: `amat[i][j] = 1` indicates a tail at i (edge from i toward j, directed or undirected).

---

## Install

### Requirements

Install the package from PyPI:

```bash
pip install b-load
```

Or install from source in editable mode:

```bash
git clone https://github.com/chrisahn99/b-load.git
cd b-load
pip install -e .
```

R dependencies (required for data generation and CPDAG/optAdjSet computation):

```r
install.packages(c("pcalg", "graph", "RBGL"))
```

The R scripts `generate_data.R` and `evaluate.R` are sourced automatically when the package is imported.

### Repository structure

```text
.
├── bload/                          # Public import namespace for pip users
│   ├── __init__.py                 # Package version metadata
│   ├── load.py                    # Re-export of src.load
│   ├── pc.py                      # Re-export of src.pc
│   ├── running.py                 # Re-export of src.running
│   ├── analysis.py                # Re-export of src.analysis
│   ├── background_knowledge.py    # Re-export of src.background_knowledge
│   ├── data_generation.py         # Re-export of src.data_generation
│   ├── evaluation.py              # Re-export of src.evaluation
│   ├── mb_by_mb.py                # Re-export of src.mb_by_mb
│   ├── counting_ci_tests.py       # Re-export of src.counting_ci_tests
│   ├── utils.py                  # Re-export of src.utils
│   └── real_data_applications/
│       ├── __init__.py
│       ├── sachs.py               # Sachs protein-signalling dataset helpers
│       └── dream.py               # DREAM4 / DREAM5 benchmark helpers
│
├── src/                           # Internal implementation package
│   ├── ...
│
├── experiment_scripts/            # Entry-point scripts for each experimental condition
│   ├── run_load_local_bk.py
│   ├── run_load_global_bk.py
│   ├── run_load_noised.py
│   ├── run_pc.py
│   ├── run_bpc_local_bk.py
│   └── run_bpc_global_bk.py
│
├── generate_data.R
├── evaluate.R
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── README.md
└── .github/workflows/publish.yml
```

---

## Quickstart

### Running experiments

Each script in `experiment_scripts/` accepts a `--save_path` argument and optional graph/sample-size parameters. Example usage:

```bash
# b-LOAD vs LOAD with locally sampled background knowledge
python experiment_scripts/run_load_local_bk.py --save_path experiment_runs/load_local.csv

# b-PC with globally sampled background knowledge
python experiment_scripts/run_bpc_global_bk.py --save_path experiment_runs/bpc_global.csv

# PC baseline (no background knowledge)
python experiment_scripts/run_pc.py --save_path experiment_runs/pc.csv
```

> **Note:** Output files are written to `experiment_runs/` by default. Create this directory locally if needed — it is excluded from version control.

### Background knowledge sampling strategies

Two strategies are available (controlled by the `sampling_strategy` argument in experiment runners):

- **`"local"`** — only edges incident on the treatment or outcome nodes are sampled as required BK. This is the primary strategy evaluated in the paper.
- **`"global"`** — required edges are drawn uniformly from all true edges in the DAG.

The fraction of edges revealed is controlled by `fraction_required` (0–1).

### Real-data experiments

Real-data runners for the Sachs protein-signalling and DREAM4/DREAM5 benchmarks are provided in `bload.real_data_applications` and `src/real_data_applications`. Place the data files under `experiments/dream4/` and `experiments/dream5/` respectively, then call:

```python
from bload.real_data_applications.dream import run_experiment_dream4, run_experiment_dream5
from bload.real_data_applications.sachs import get_sachs_ground_truth_dag, fit_oracle_weights

# DREAM4 network 1, local BK, 30 % of true edges revealed
result_dict, details_dict = run_experiment_dream4(subset_number=1, sampling_strategy="local", frac_req=0.3)
```

### Analysing results

`bload.analysis` provides utilities for loading result CSVs, computing the complexity metric $\pi = n_{samples} / (n_{nodes} \times expected\_degree / 2)$, aggregating into regime tables, and producing F1 and intervention-distance plots:

```python
from bload.analysis import load_iter_folder, add_complexity_metric, regime_table_3, plot_f1

df_bload = load_iter_folder("experiment_runs/load_local_*", stat_col_pairs=[...])
df_bload = add_complexity_metric(df_bload)
fig = plot_f1(df_pc, df_bpc, df_bpc_global, df_load, df_bload_global)
```

---

## Results

The repository is organized around the experimental comparison between standard LOAD, background-knowledge-augmented b-LOAD, and PC-style baselines. The key outputs are CSV result files produced by the experiment scripts and summary plots generated via `src/analysis.py`.

These experiments typically report:

- structural accuracy (precision, recall, F1),
- intervention-distance performance,
- the effect of revealed background knowledge on adjustment-set identification, and
- robustness under partially noisy background knowledge.

The analysis utilities in `src/analysis.py` let you aggregate runs across graph sizes, sampling strategies, and difficulty regimes and visualize the outcome trends.

---

## Reproducing the Paper

To reproduce the paper's experiments:

1. Install the Python and R dependencies listed in the installation section.
2. Run the synthetic experiment scripts in `experiment_scripts/` with the desired background-knowledge setting.
3. Save the resulting outputs under `experiment_runs/`.
4. Use the analysis utilities in `src/analysis.py` to aggregate metrics and generate plots.

This workflow matches the study design used to compare:

- LOAD without background knowledge,
- b-LOAD with local and global background knowledge,
- b-PC with the same background-knowledge settings, and
- the PC baseline without knowledge augmentation.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ahnbload206,
    author = {Ahn, Seong Woo and Leite, Alessandro and De Melo Costa, Jos{\'e} Lucas and Popineau, Fabrice and Doan, Bich-Li{\^e}n and Rimmel, Arpad},
    title = {Knowledge-Informed Local Causal Discovery of Optimal Adjustment Sets},
    year = {2026},
    doi = {10.1145/3799682.3840893},
    booktitle = {The 35th ACM International Conference on Information and Knowledge Management},
    pages = {72--82},
}
```

---

## License

MIT, see [LICENSE](LICENSE).
