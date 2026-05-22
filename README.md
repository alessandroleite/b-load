# LOAD in MPDAG

Local Ordering-based Ancestral Discovery (LOAD) in Maximal Partially Directed Acyclic Graphs (MPDAGs).

## Overview
This repository provides the core implementation of the LOAD algorithm and its extension for MPDAGs. It is designed for researchers in causal discovery to run local ancestral discovery experiments with varying degrees of background knowledge.

## Project Structure
- `src/`: Core Python modules.
  - `load.py`: Implementation of `load` and `load_in_mpdag`.
  - `background_knowledge.py`: Utilities for handling structural constraints.
  - `counting_ci_tests.py`: CI test wrappers with performance tracking.
  - `evaluation.py`: Metrics for validating discovered causal relations.
- `experiment_scripts/`:
  - `run_load_local_bk.py`: Runs LOAD vs b-LOAD with locally sampled background knowledge.
  - `run_load_global_bk.py`: Runs LOAD vs b-LOAD with globally sampled background knowledge.
  - `run_load_noised.py`: Runs LOAD with noised background knowledge.
  - `run_pc.py`: Runs the PC algorithm.
  - `run_bpc_local_bk.py`: Runs b-PC with locally sampled background knowledge.
  - `run_bpc_global_bk.py`: Runs b-PC with globally sampled background knowledge.
- `generate_data.R` / `evaluate.R`: R-based utilities for data synthesis and statistical comparison.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments
The scripts in the `experiment_scripts/` directory allow for running experiments with different algorithms and background knowledge sampling strategies:

- `run_load_local_bk.py`: Runs LOAD vs b-LOAD augmented with locally sampled background knowledge.
- `run_load_global_bk.py`: Runs LOAD vs b-LOAD augmented with globally sampled background knowledge.
- `run_load_noised.py`: Runs LOAD with noised background knowledge.
- `run_pc.py`: Runs the PC algorithm.
- `run_bpc_local_bk.py`: Runs b-PC augmented with locally sampled background knowledge.
- `run_bpc_global_bk.py`: Runs b-PC augmented with globally sampled background knowledge.

Example usage:
```bash
python experiment_scripts/run_load_local_bk.py --save_path results.csv
python experiment_scripts/run_pc.py --save_path results_pc.csv
```
*Note: You may need to create a local directory (e.g., `experiment_runs/`) to store outputs, as these are ignored by version control.*

## Implementation Details
The algorithm focuses on target-node-specific ancestral discovery. Key features include:
- Support for **Maximal Partially Directed Acyclic Graphs (MPDAGs)** as prior knowledge.
- Integration of **Conditional Independence (CI) test counting** to measure computational efficiency.
- Local sampling strategies for forbidden and required edges.

## Citation
If you use this code in your research, please cite:
[Insert Citation Here]
