"""
analysis.py — Result loading, aggregation, and visualisation utilities.

This module centralises all post-processing logic that was previously
embedded in the synthetic_results analysis notebook.  It covers four
concerns:

1. **Result loading** — reading experiment CSV files produced by the
   experiment scripts, including the special "100-iter" parallelised
   format where each configuration is split across multiple job files.

2. **Complexity metrics and CIs** — computing the statistical-power
   metric π = n_samples / (expected_edges / 2) and 95 % confidence
   intervals for every dataframe.

3. **Regime tables** — aggregating mean F1 and intervention distance
   across Low / Medium / High (or Low / High) complexity regimes for
   all six methods (PC, b-PC local, b-PC global, LOAD, b-LOAD local,
   b-LOAD global).

4. **Graphical analysis** — publication-ready F1 score plots (2 × 2
   grid over BK fractions) and intervention-distance panels (three
   views: data efficiency, BK sensitivity, scalability).
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_ITER = 1000  # Total iterations assumed for CI computation (1000 across all jobs)

# Column groups describing the statistics to aggregate for each df type
PC_STAT_PAIRS: list[tuple[str, str]] = [
    ("f1_mean", "f1_std"),
    ("int_dist_mean", "int_dist_std"),
]
"""Stat-column pairs (mean, std) for PC-style result dataframes."""

LOAD_STAT_PAIRS: list[tuple[str, str]] = [
    ("f1_with_b_mean",         "f1_with_b_std"),
    ("f1_without_b_mean",      "f1_without_b_std"),
    ("int_dist_with_b_mean",   "int_dist_with_b_std"),
    ("int_dist_without_b_mean","int_dist_without_b_std"),
]
"""Stat-column pairs (mean, std) for LOAD-style result dataframes."""

# Colour palette shared by all plots (consistent across F1 and int-dist figures)
PLOT_COLORS: dict[str, str] = {
    "pc":          "#888888",  # grey   — PC (no BK)
    "bpc":         "#d62728",  # red    — b-PC (local BK)
    "bpc_glob":    "#9467bd",  # purple — b-PC (global BK)
    "load":        "#1f77b4",  # blue   — LOAD (no BK)
    "bload_local": "#2ca02c",  # green  — b-LOAD (local BK)
    "bload_glob":  "#ff7f0e",  # orange — b-LOAD (global BK)
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Result loading
# ─────────────────────────────────────────────────────────────────────────────

def combine_stats(
    job_results: list[tuple[int, float, float]],
) -> tuple[float, float]:
    """Combine per-job statistics into a single global mean and std.

    Uses the exact parallel-variance formula (law of total variance) so that
    the result is mathematically identical to computing the mean and std over
    the full pooled sample.  This is required because the experiments were
    parallelised: each configuration was split into multiple job files of
    ``n_per_iter`` iterations each, and their statistics cannot be averaged
    naively.

    Args:
        job_results: List of ``(n, mean, var)`` tuples, one per job file,
            where ``n`` is the number of iterations in that file, ``mean`` is
            the per-job sample mean, and ``var`` is the per-job sample
            variance (i.e., ``std ** 2``).

    Returns:
        ``(global_mean, global_std)`` — the combined mean and standard
        deviation over all ``N = sum(n_j)`` iterations.

    Example::

        >>> jobs = [(100, 0.8, 0.01), (100, 0.75, 0.02)]
        >>> combine_stats(jobs)
        (0.775, ...)
    """
    N = sum(n for n, _, _ in job_results)
    global_mean = sum(n * m for n, m, _ in job_results) / N
    combined_var = (
        sum((n - 1) * v + n * (m - global_mean) ** 2 for n, m, v in job_results)
        / (N - 1)
    )
    return global_mean, float(np.sqrt(combined_var))


def load_iter_folder(
    folder: str,
    stat_col_pairs: list[tuple[str, str]],
    n_per_iter: int = 100,
) -> pd.DataFrame:
    """Load a parallelised "100-iter" experiment folder into a single DataFrame.

    In the 100-iter format, each graph configuration is stored as multiple
    CSV files named ``<config>_iter1.csv``, ``<config>_iter2.csv``, …  Each
    file contains results for ``n_per_iter`` independent repetitions.  This
    function groups files by their base config name, aggregates the statistics
    across all iter files using :func:`combine_stats`, and returns one
    aggregated row per configuration.

    Args:
        folder: Path to the directory containing the ``_iter{n}.csv`` files.
        stat_col_pairs: List of ``(mean_col, std_col)`` pairs identifying
            which columns to aggregate.  Non-statistic columns (e.g. graph
            parameters) are taken verbatim from the first iter file.
        n_per_iter: Number of iterations stored in each job file.
            Defaults to 100.

    Returns:
        DataFrame with one aggregated row per configuration.  Returns an
        empty DataFrame and prints a warning if the folder contains no files.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        print(f"  WARNING: no CSV files found in {folder}")
        return pd.DataFrame()

    # Group files by base config name (strip _iter{n}.csv suffix)
    groups: dict[str, list[str]] = defaultdict(list)
    for f in files:
        base = re.sub(r"_iter\d+\.csv$", "", os.path.basename(f))
        groups[base].append(f)

    rows = []
    for base, flist in sorted(groups.items()):
        iter_dfs = [pd.read_csv(f) for f in sorted(flist)]
        row = iter_dfs[0].iloc[0].to_dict()  # metadata from first iter file
        for mean_col, std_col in stat_col_pairs:
            job_results = [
                (n_per_iter, df[mean_col].iloc[0], df[std_col].iloc[0] ** 2)
                for df in iter_dfs
            ]
            row[mean_col], row[std_col] = combine_stats(job_results)
        rows.append(row)

    n_iters = max(len(v) for v in groups.values())
    print(f"    {n_iters} iters × {len(rows)} configs  ← {os.path.basename(folder)}")
    return pd.DataFrame(rows)


def load_standard_folder(pattern: str) -> pd.DataFrame:
    """Load a standard (non-parallelised) experiment folder by glob pattern.

    A "standard" folder stores one CSV per configuration, each already
    containing the final aggregated statistics from all iterations.

    Args:
        pattern: Glob pattern (e.g. ``'../experiment_runs/pc/*.csv'``).

    Returns:
        Concatenated DataFrame of all matching CSVs.  Returns an empty
        DataFrame if no files match.
    """
    files = glob.glob(pattern)
    if not files:
        print(f"  WARNING: no files matching {pattern}")
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Complexity metrics and confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def add_complexity_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Add the statistical-power metric π to a result DataFrame (in-place).

    π is defined as::

        π = n_samples / ((observed × exp_degree) / 2)

    which equals the ratio of available samples to the expected number of
    edges in the graph.  It serves as the x-axis for all F1 plots.

    Args:
        df: DataFrame containing at least the columns ``n_samples``,
            ``observed``, and ``exp_degree``.

    Returns:
        The same DataFrame with a new ``complexity_metric`` column added.
    """
    df["complexity_metric"] = df["n_samples"] / ((df["observed"] * df["exp_degree"]) / 2)
    return df


def add_confidence_intervals_pc(df: pd.DataFrame, n_iter: int = N_ITER) -> pd.DataFrame:
    """Add 95 % confidence-interval columns to a PC-style result DataFrame.

    Adds ``ci_f1`` and ``ci_int_dist`` columns computed as
    ``1.96 × std / √n_iter``.

    Args:
        df: DataFrame with ``f1_std`` and ``int_dist_std`` columns.
        n_iter: Total number of iterations used to produce each row.
            Defaults to :data:`N_ITER` (1000).

    Returns:
        The same DataFrame with CI columns added in-place.
    """
    df["ci_f1"]       = 1.96 * df["f1_std"]       / np.sqrt(n_iter)
    df["ci_int_dist"] = 1.96 * df["int_dist_std"]  / np.sqrt(n_iter)
    return df


def add_confidence_intervals_load(df: pd.DataFrame, n_iter: int = N_ITER) -> pd.DataFrame:
    """Add 95 % confidence-interval columns to a LOAD-style result DataFrame.

    Adds CI columns for both the with-BK and without-BK variants of F1 and
    intervention distance.

    Args:
        df: DataFrame with ``f1_with_b_std``, ``f1_without_b_std``,
            ``int_dist_with_b_std``, and ``int_dist_without_b_std`` columns.
        n_iter: Total number of iterations per row (default 1000).

    Returns:
        The same DataFrame with four new CI columns added in-place.
    """
    df["ci_f1_with_b"]          = 1.96 * df["f1_with_b_std"]          / np.sqrt(n_iter)
    df["ci_f1_without_b"]       = 1.96 * df["f1_without_b_std"]       / np.sqrt(n_iter)
    df["ci_int_dist_with_b"]    = 1.96 * df["int_dist_with_b_std"]    / np.sqrt(n_iter)
    df["ci_int_dist_without_b"] = 1.96 * df["int_dist_without_b_std"] / np.sqrt(n_iter)
    return df


def make_load_no_bk_baseline(df_load: pd.DataFrame) -> pd.DataFrame:
    """Extract the BK-independent LOAD baseline from a LOAD result DataFrame.

    The ``f1_without_b_mean`` metric is identical across all
    ``fraction_required`` values for the same ``(observed, exp_degree,
    n_samples)`` configuration.  This function deduplicates on those three
    keys and returns a lean DataFrame used as the LOAD (no-BK) reference
    series in plots.

    Args:
        df_load: Full LOAD result DataFrame (with BK columns).

    Returns:
        Deduplicated DataFrame containing only the BK-independent columns.
    """
    cols = [
        "observed", "exp_degree", "n_samples", "complexity_metric",
        "f1_without_b_mean",       "ci_f1_without_b",
        "int_dist_without_b_mean", "ci_int_dist_without_b",
    ]
    return (
        df_load
        .drop_duplicates(subset=["observed", "exp_degree", "n_samples"])
        [cols]
        .copy()
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Regime tables
# ─────────────────────────────────────────────────────────────────────────────

def _bin_3(pi: float) -> str:
    """Map π to a three-way complexity regime label."""
    if pi < 10:   return "Low"
    if pi < 100:  return "Medium"
    return "High"


def _bin_2(pi: float) -> str:
    """Map π to a two-way complexity regime label (threshold at π = 50)."""
    return "Low" if pi < 50 else "High"


def _extract_regime_row(
    df: pd.DataFrame,
    method_name: str,
    f1_col: str,
    id_col: str,
    regimes: list[str],
    bin_fn,
) -> dict:
    """Build one summary row by averaging over each complexity regime.

    Args:
        df: Result DataFrame for one method, already containing
            ``complexity_metric``.
        method_name: Display name for the method (used in the table).
        f1_col: Column name for the F1 mean.
        id_col: Column name for the intervention-distance mean.
        regimes: Ordered list of regime labels (e.g. ``['Low','Medium','High']``).
        bin_fn: Callable mapping a π value to a regime label.

    Returns:
        Dict with ``'Method'`` plus ``'{regime} F1'`` and
        ``'{regime} Int. Dist.'`` keys for each regime.
    """
    tmp = df.copy()
    tmp["regime"] = tmp["complexity_metric"].apply(bin_fn)
    row: dict = {"Method": method_name}
    for r in regimes:
        sub = tmp[tmp["regime"] == r]
        row[f"{r} F1"]         = sub[f1_col].mean()  if not sub.empty else np.nan
        row[f"{r} Int. Dist."] = sub[id_col].mean()  if not sub.empty else np.nan
    return row


def _append_best_row(summary: pd.DataFrame, regimes: list[str]) -> pd.DataFrame:
    """Append a 'Best Model' row identifying the winner per regime and metric."""
    best: dict = {"Method": "Best Model"}
    for r in regimes:
        fc, ic = f"{r} F1", f"{r} Int. Dist."
        best[fc] = summary.loc[summary[fc].idxmax(), "Method"] if summary[fc].notna().any() else None
        best[ic] = summary.loc[summary[ic].idxmin(), "Method"] if summary[ic].notna().any() else None
    final = pd.concat([summary, pd.DataFrame([best])], ignore_index=True)
    num = final.select_dtypes(include=[np.number]).columns
    final[num] = final[num].round(4)
    return final


def regime_table_3(
    df_pc: pd.DataFrame,
    df_bpc: pd.DataFrame,
    df_bpc_global: pd.DataFrame,
    df_load: pd.DataFrame,
    df_bload_global: pd.DataFrame,
) -> pd.DataFrame:
    """Build a triple-regime (Low / Medium / High) summary table for all six methods.

    Averages mean F1 score and mean intervention distance within each π
    regime across the full dataset for each of the six comparison methods:
    PC, b-PC (local), b-PC (global), LOAD, b-LOAD (local), b-LOAD (global).

    Regime thresholds: Low (π < 10), Medium (10 ≤ π < 100), High (π ≥ 100).

    Args:
        df_pc: PC (no BK) result DataFrame.
        df_bpc: b-PC local-BK result DataFrame.
        df_bpc_global: b-PC global-BK result DataFrame.
        df_load: LOAD / b-LOAD local-BK result DataFrame.
        df_bload_global: b-LOAD global-BK result DataFrame.

    Returns:
        DataFrame with one row per method plus a 'Best Model' row, and
        columns for F1 and intervention distance in each regime.
    """
    regimes = ["Low", "Medium", "High"]
    df_load_base = df_load.drop_duplicates(subset=["observed", "exp_degree", "n_samples"])

    records = [
        _extract_regime_row(df_pc,          "PC",             "f1_mean",           "int_dist_mean",           regimes, _bin_3),
        _extract_regime_row(df_bpc,         "b-PC (local)",   "f1_mean",           "int_dist_mean",           regimes, _bin_3),
        _extract_regime_row(df_bpc_global,  "b-PC (global)",  "f1_mean",           "int_dist_mean",           regimes, _bin_3),
        _extract_regime_row(df_load_base,   "LOAD",           "f1_without_b_mean", "int_dist_without_b_mean", regimes, _bin_3),
        _extract_regime_row(df_load,        "b-LOAD (local)", "f1_with_b_mean",    "int_dist_with_b_mean",    regimes, _bin_3),
        _extract_regime_row(df_bload_global,"b-LOAD (global)","f1_with_b_mean",    "int_dist_with_b_mean",    regimes, _bin_3),
    ]
    return _append_best_row(pd.DataFrame(records), regimes)


def regime_table_2(
    df_pc: pd.DataFrame,
    df_bpc: pd.DataFrame,
    df_bpc_global: pd.DataFrame,
    df_load: pd.DataFrame,
    df_bload_global: pd.DataFrame,
) -> pd.DataFrame:
    """Build a double-regime (Low / High) summary table for all six methods.

    Same structure as :func:`regime_table_3` but with a single threshold at
    π = 50: Low (π < 50) vs High (π ≥ 50).

    Args:
        df_pc: PC (no BK) result DataFrame.
        df_bpc: b-PC local-BK result DataFrame.
        df_bpc_global: b-PC global-BK result DataFrame.
        df_load: LOAD / b-LOAD local-BK result DataFrame.
        df_bload_global: b-LOAD global-BK result DataFrame.

    Returns:
        DataFrame with one row per method plus a 'Best Model' row, and
        columns for F1 and intervention distance in each regime.
    """
    regimes = ["Low", "High"]
    df_load_base = df_load.drop_duplicates(subset=["observed", "exp_degree", "n_samples"])

    records = [
        _extract_regime_row(df_pc,          "PC",             "f1_mean",           "int_dist_mean",           regimes, _bin_2),
        _extract_regime_row(df_bpc,         "b-PC (local)",   "f1_mean",           "int_dist_mean",           regimes, _bin_2),
        _extract_regime_row(df_bpc_global,  "b-PC (global)",  "f1_mean",           "int_dist_mean",           regimes, _bin_2),
        _extract_regime_row(df_load_base,   "LOAD",           "f1_without_b_mean", "int_dist_without_b_mean", regimes, _bin_2),
        _extract_regime_row(df_load,        "b-LOAD (local)", "f1_with_b_mean",    "int_dist_with_b_mean",    regimes, _bin_2),
        _extract_regime_row(df_bload_global,"b-LOAD (global)","f1_with_b_mean",    "int_dist_with_b_mean",    regimes, _bin_2),
    ]
    return _append_best_row(pd.DataFrame(records), regimes)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graphical analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def agg_by_complexity(
    df: pd.DataFrame,
    mean_col: str,
    ci_col: str,
    cx_col: str = "complexity_metric",
    n_bins: int = 40,
) -> pd.DataFrame:
    """Aggregate a result series into log-spaced complexity bins.

    Groups rows into ``n_bins`` log-spaced bins of π, then averages the
    ``mean_col`` and ``ci_col`` within each bin.  This smooths the zigzag
    pattern that arises from many discrete (n_samples, observed) combinations
    mapping to similar but not identical π values.

    Args:
        df: DataFrame containing ``cx_col``, ``mean_col``, and ``ci_col``.
        mean_col: Name of the metric mean column (e.g. ``'f1_mean'``).
        ci_col: Name of the CI column (e.g. ``'ci_f1'``).
        cx_col: Name of the complexity-metric column.
            Defaults to ``'complexity_metric'``.
        n_bins: Number of log-spaced bins.  Defaults to 40.

    Returns:
        Aggregated DataFrame with one row per non-empty bin, sorted by π.
        Columns: ``cx_col``, ``mean_col``, ``ci_col``.
    """
    lo = df[cx_col][df[cx_col] > 0].min()
    hi = df[cx_col].max()
    bins = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    tmp = df.copy()
    tmp["_bin"] = pd.cut(tmp[cx_col], bins=bins, labels=False, include_lowest=True)
    tmp["_bin_mid"] = tmp["_bin"].map(
        lambda b: np.sqrt(bins[int(b)] * bins[int(b) + 1]) if pd.notna(b) else np.nan
    )
    return (
        tmp.dropna(subset=["_bin"])
        .groupby("_bin_mid")[[mean_col, ci_col]]
        .mean()
        .reset_index()
        .rename(columns={"_bin_mid": cx_col})
        .sort_values(cx_col)
    )


def _plot_line(
    ax: plt.Axes,
    df_agg: pd.DataFrame,
    mean_col: str,
    ci_col: str,
    color: str,
    label: str,
    cx_col: str = "complexity_metric",
    linestyle: str = "-",
    marker: str = "o",
    ci_alpha: float = 0.15,
) -> None:
    """Draw a line with shaded CI band on ``ax``."""
    x  = df_agg[cx_col]
    y  = df_agg[mean_col]
    ci = df_agg[ci_col]
    ax.plot(x, y, color=color, label=label,
            linestyle=linestyle, marker=marker, markersize=5, linewidth=1.8)
    ax.fill_between(x, y - ci, y + ci, color=color, alpha=ci_alpha)


def plot_f1(
    df_pc: pd.DataFrame,
    df_bpc: pd.DataFrame,
    df_bpc_global: pd.DataFrame,
    df_load: pd.DataFrame,
    df_bload_global: pd.DataFrame,
    df_load_no_bk: pd.DataFrame,
    target_fracs: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot F1 scores for all six methods as a 2 × 2 grid over BK fractions.

    Each subplot corresponds to one value of ``fraction_required`` (the
    fraction of known required edges).  The x-axis is the complexity metric
    π on a log scale; the y-axis is the mean F1 score with 95 % CI bands.

    Requires ``scienceplots`` to be installed for publication-quality styling.

    Args:
        df_pc: PC result DataFrame (must have ``complexity_metric``,
            ``ci_f1``, ``f1_mean``).
        df_bpc: b-PC local result DataFrame.
        df_bpc_global: b-PC global result DataFrame.
        df_load: b-LOAD local result DataFrame (also provides b-LOAD baseline
            via ``f1_with_b_mean``).
        df_bload_global: b-LOAD global result DataFrame.
        df_load_no_bk: BK-independent LOAD baseline DataFrame (from
            :func:`make_load_no_bk_baseline`).
        target_fracs: BK fractions to plot (one subplot each).
            Must have exactly 4 values to fill the 2 × 2 grid.
        save_path: If given, saves the figure to this path (e.g.
            ``'f1_plot.pdf'``).

    Returns:
        The matplotlib Figure object.
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except ImportError:
        pass  # fall back to default style

    C = PLOT_COLORS
    agg_pc = agg_by_complexity(df_pc,         mean_col="f1_mean",           ci_col="ci_f1")
    agg_lo = agg_by_complexity(df_load_no_bk, mean_col="f1_without_b_mean", ci_col="ci_f1_without_b")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             sharex=True, sharey=True, layout="constrained")
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(target_fracs):
            ax.set_visible(False)
            continue

        frac = target_fracs[i]
        sub_bpc  = df_bpc        [np.isclose(df_bpc        ["fraction_required"], frac)]
        sub_bpcg = df_bpc_global [np.isclose(df_bpc_global ["fraction_required"], frac)]
        sub_lo   = df_load       [np.isclose(df_load       ["fraction_required"], frac)]
        sub_bg   = df_bload_global[np.isclose(df_bload_global["fraction_required"], frac)]

        _plot_line(ax, agg_pc, "f1_mean", "ci_f1",
                   C["pc"],   "PC",            linestyle="-",  marker="o")
        if not sub_bpc.empty:
            _plot_line(ax, agg_by_complexity(sub_bpc,  "f1_mean", "ci_f1"),
                       "f1_mean", "ci_f1", C["bpc"], "b-PC (local)", linestyle="-", marker="s")
        if not sub_bpcg.empty:
            _plot_line(ax, agg_by_complexity(sub_bpcg, "f1_mean", "ci_f1"),
                       "f1_mean", "ci_f1", C["bpc_glob"], "b-PC (global)", linestyle="-", marker="D")
        _plot_line(ax, agg_lo, "f1_without_b_mean", "ci_f1_without_b",
                   C["load"], "LOAD", linestyle="--", marker="o")
        if not sub_lo.empty:
            _plot_line(ax, agg_by_complexity(sub_lo, "f1_with_b_mean", "ci_f1_with_b"),
                       "f1_with_b_mean", "ci_f1_with_b",
                       C["bload_local"], "b-LOAD (local)", linestyle="--", marker="s")
        if not sub_bg.empty:
            _plot_line(ax, agg_by_complexity(sub_bg, "f1_with_b_mean", "ci_f1_with_b"),
                       "f1_with_b_mean", "ci_f1_with_b",
                       C["bload_glob"], "b-LOAD (global)", linestyle="--", marker="^")

        ax.set_title(f"Frac: {frac:.0%}", fontsize=20, pad=6, fontweight="bold")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, which="major", linestyle="-", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    fig.supxlabel("Statistical Power Metric — log scale", fontsize=20, fontweight="bold")
    fig.supylabel("F1 Score", fontsize=20, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=6,
               frameon=False, prop={"size": 18, "weight": "bold"})

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_intervention_distance(
    df_pc: pd.DataFrame,
    df_bpc: pd.DataFrame,
    df_bpc_global: pd.DataFrame,
    df_load: pd.DataFrame,
    df_bload_global: pd.DataFrame,
    exp_degree: float = 2.0,
    frac: float = 0.3,
    d_panels_a: Sequence[int] | None = None,
    n_panels_bc: Sequence[int] = (100, 1000, 5000, 10000),
    d_panel_b: int = 10,
    save_prefix: str | None = None,
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Produce three intervention-distance figures for all six methods.

    All three figures restrict to a single expected-degree value (``exp_degree``,
    default ER2).

    - **Figure A** — Data efficiency: x-axis = ``n_samples``, panels = graph
      sizes ``d``.  BK fraction fixed at ``frac``.
    - **Figure B** — BK sensitivity: x-axis = ``fraction_required``, panels =
      ``n_samples``.  Graph size fixed at ``d_panel_b``.
    - **Figure C** — Scalability: x-axis = ``observed`` (graph size), panels
      = ``n_samples``.  BK fraction fixed at ``frac``.

    Args:
        df_pc: PC result DataFrame.
        df_bpc: b-PC local result DataFrame.
        df_bpc_global: b-PC global result DataFrame.
        df_load: b-LOAD local (and LOAD baseline) result DataFrame.
        df_bload_global: b-LOAD global result DataFrame.
        exp_degree: Expected graph degree to filter on (default 2.0 = ER2).
        frac: BK fraction used in panel A and C (default 0.3).
        d_panels_a: Graph sizes used as panels in Figure A.  Defaults to
            all unique ``observed`` values in ``df_pc`` for ``exp_degree``.
        n_panels_bc: Sample sizes used as panels in Figures B and C.
            Defaults to ``[100, 1000, 5000, 10000]`` (filtered to available).
        d_panel_b: Graph size fixed for Figure B (default 10).
        save_prefix: If given, saves the three figures as
            ``{save_prefix}_A.pdf``, ``{save_prefix}_B.pdf``,
            ``{save_prefix}_C.pdf``.

    Returns:
        Tuple ``(fig_a, fig_b, fig_c)`` of matplotlib Figure objects.
    """
    C = PLOT_COLORS

    # ── Filter to target exp_degree ───────────────────────────────────────────
    er = exp_degree
    pc_er   = df_pc          [np.isclose(df_pc          ["exp_degree"], er)].copy()
    bpc_er  = df_bpc         [np.isclose(df_bpc         ["exp_degree"], er)].copy()
    bpcg_er = df_bpc_global  [np.isclose(df_bpc_global  ["exp_degree"], er)].copy()
    lo_er   = df_load        [np.isclose(df_load        ["exp_degree"], er)].copy()
    bg_er   = df_bload_global[np.isclose(df_bload_global["exp_degree"], er)].copy()

    all_n = sorted(
        set(bpc_er["n_samples"].unique()) |
        set(bpcg_er["n_samples"].unique()) |
        set(bg_er["n_samples"].unique())
    )

    # ── Shared helpers ────────────────────────────────────────────────────────
    def _pline(ax, x, y, ci, color, label, ls="-", mk="o"):
        ax.plot(x, y, color=color, label=label, linestyle=ls,
                marker=mk, markersize=5, linewidth=2)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.12)

    def _cap_ylim(ax, *series, pct=95, margin=1.20):
        vals = [v for s in series for v in (s if hasattr(s, "__iter__") else [s])
                if not np.isnan(v)]
        if vals:
            ax.set_ylim(bottom=0, top=np.nanpercentile(vals, pct) * margin)

    def _fmt(ax, xlabel, xscale="linear", legend_loc="upper right"):
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Intervention Distance", fontsize=10)
        ax.set_xscale(xscale)
        ax.grid(True, which="major", linestyle="-", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.9, loc=legend_loc)

    # ── Figure A — data efficiency ────────────────────────────────────────────
    D_A = sorted(pc_er["observed"].unique()) if d_panels_a is None else list(d_panels_a)
    fig_a, axes_a = plt.subplots(1, len(D_A), figsize=(4 * len(D_A), 5), sharey=False)
    if len(D_A) == 1:
        axes_a = [axes_a]

    for ax, d in zip(axes_a, D_A):
        pc_s   = pc_er  [pc_er  ["observed"] == d].sort_values("n_samples")
        bpc_s  = bpc_er [(bpc_er ["observed"] == d) & np.isclose(bpc_er ["fraction_required"], frac)].sort_values("n_samples")
        bpcg_s = bpcg_er[(bpcg_er["observed"] == d) & np.isclose(bpcg_er["fraction_required"], frac)].sort_values("n_samples")
        lo_s   = lo_er  [lo_er  ["observed"] == d].drop_duplicates("n_samples").sort_values("n_samples")
        blo_s  = lo_er  [(lo_er ["observed"] == d) & np.isclose(lo_er ["fraction_required"], frac)].sort_values("n_samples")
        bg_s   = bg_er  [(bg_er ["observed"] == d) & np.isclose(bg_er ["fraction_required"], frac)].sort_values("n_samples")

        _pline(ax, pc_s["n_samples"], pc_s["int_dist_mean"], pc_s["ci_int_dist"],
               C["pc"], "PC", "-", "o")
        if not bpc_s.empty:
            _pline(ax, bpc_s["n_samples"], bpc_s["int_dist_mean"], bpc_s["ci_int_dist"],
                   C["bpc"], "b-PC (local)", "-", "s")
        if not bpcg_s.empty:
            _pline(ax, bpcg_s["n_samples"], bpcg_s["int_dist_mean"], bpcg_s["ci_int_dist"],
                   C["bpc_glob"], "b-PC (global)", "-", "D")
        _pline(ax, lo_s["n_samples"], lo_s["int_dist_without_b_mean"], lo_s["ci_int_dist_without_b"],
               C["load"], "LOAD", "--", "o")
        if not blo_s.empty:
            _pline(ax, blo_s["n_samples"], blo_s["int_dist_with_b_mean"], blo_s["ci_int_dist_with_b"],
                   C["bload_local"], "b-LOAD (local)", "--", "s")
        if not bg_s.empty:
            _pline(ax, bg_s["n_samples"], bg_s["int_dist_with_b_mean"], bg_s["ci_int_dist_with_b"],
                   C["bload_glob"], "b-LOAD (global)", "--", "^")

        _cap_ylim(ax, list(pc_s["int_dist_mean"]), list(lo_s["int_dist_without_b_mean"]),
                  list(bpc_s["int_dist_mean"]) if not bpc_s.empty else [],
                  list(bpcg_s["int_dist_mean"]) if not bpcg_s.empty else [],
                  list(blo_s["int_dist_with_b_mean"]) if not blo_s.empty else [],
                  list(bg_s["int_dist_with_b_mean"]) if not bg_s.empty else [])
        ax.set_title(f"d = {d} nodes", fontsize=11)
        _fmt(ax, "n_samples (log scale)", xscale="log")

    plt.tight_layout()
    if save_prefix:
        fig_a.savefig(f"{save_prefix}_A.pdf", bbox_inches="tight")

    # ── Figure B — BK sensitivity ─────────────────────────────────────────────
    N_B = [n for n in n_panels_bc if n in all_n]
    fig_b, axes_b = plt.subplots(1, len(N_B), figsize=(5 * len(N_B), 5), sharey=False)
    if len(N_B) == 1:
        axes_b = [axes_b]

    for ax, n in zip(axes_b, N_B):
        pc_row  = pc_er [(pc_er ["observed"] == d_panel_b) & (pc_er ["n_samples"] == n)]
        bpc_s   = bpc_er[(bpc_er["observed"] == d_panel_b) & (bpc_er["n_samples"] == n)].sort_values("fraction_required")
        bpcg_s  = bpcg_er[(bpcg_er["observed"] == d_panel_b) & (bpcg_er["n_samples"] == n)].sort_values("fraction_required")
        lo_rows = lo_er [(lo_er ["observed"] == d_panel_b) & (lo_er ["n_samples"] == n)]
        lo_row  = lo_rows.drop_duplicates("n_samples")
        blo_s   = lo_rows.sort_values("fraction_required")
        bg_s    = bg_er [(bg_er ["observed"] == d_panel_b) & (bg_er ["n_samples"] == n)].sort_values("fraction_required")

        if not pc_row.empty:
            py, pci = pc_row["int_dist_mean"].iloc[0], pc_row["ci_int_dist"].iloc[0]
            ax.axhline(py, color=C["pc"], lw=2, ls="-", label="PC")
            ax.axhspan(py - pci, py + pci, color=C["pc"], alpha=0.10)
        if not lo_row.empty:
            ly, lci = lo_row["int_dist_without_b_mean"].iloc[0], lo_row["ci_int_dist_without_b"].iloc[0]
            ax.axhline(ly, color=C["load"], lw=2, ls="--", label="LOAD")
            ax.axhspan(ly - lci, ly + lci, color=C["load"], alpha=0.10)

        if not bpc_s.empty:
            _pline(ax, bpc_s["fraction_required"], bpc_s["int_dist_mean"], bpc_s["ci_int_dist"],
                   C["bpc"], "b-PC (local)", "-", "s")
        if not bpcg_s.empty:
            _pline(ax, bpcg_s["fraction_required"], bpcg_s["int_dist_mean"], bpcg_s["ci_int_dist"],
                   C["bpc_glob"], "b-PC (global)", "-", "D")
        if not blo_s.empty:
            _pline(ax, blo_s["fraction_required"], blo_s["int_dist_with_b_mean"], blo_s["ci_int_dist_with_b"],
                   C["bload_local"], "b-LOAD (local)", "--", "s")
        if not bg_s.empty:
            _pline(ax, bg_s["fraction_required"], bg_s["int_dist_with_b_mean"], bg_s["ci_int_dist_with_b"],
                   C["bload_glob"], "b-LOAD (global)", "--", "^")

        _cap_ylim(ax,
                  list(pc_row["int_dist_mean"]) if not pc_row.empty else [],
                  list(lo_row["int_dist_without_b_mean"]) if not lo_row.empty else [],
                  list(bpc_s["int_dist_mean"]) if not bpc_s.empty else [],
                  list(bpcg_s["int_dist_mean"]) if not bpcg_s.empty else [],
                  list(blo_s["int_dist_with_b_mean"]) if not blo_s.empty else [],
                  list(bg_s["int_dist_with_b_mean"]) if not bg_s.empty else [],
                  pct=100)
        ax.set_title(f"n = {n} samples", fontsize=11)
        _fmt(ax, "fraction_required")

    plt.tight_layout()
    if save_prefix:
        fig_b.savefig(f"{save_prefix}_B.pdf", bbox_inches="tight")

    # ── Figure C — scalability ────────────────────────────────────────────────
    N_C = [n for n in n_panels_bc if n in all_n]
    fig_c, axes_c = plt.subplots(1, len(N_C), figsize=(5 * len(N_C), 5), sharey=False)
    if len(N_C) == 1:
        axes_c = [axes_c]

    for ax, n in zip(axes_c, N_C):
        pc_s   = pc_er  [pc_er  ["n_samples"] == n].sort_values("observed")
        bpc_s  = bpc_er [(bpc_er ["n_samples"] == n) & np.isclose(bpc_er ["fraction_required"], frac)].sort_values("observed")
        bpcg_s = bpcg_er[(bpcg_er["n_samples"] == n) & np.isclose(bpcg_er["fraction_required"], frac)].sort_values("observed")
        lo_s   = lo_er  [lo_er  ["n_samples"] == n].drop_duplicates("observed").sort_values("observed")
        blo_s  = lo_er  [(lo_er ["n_samples"] == n) & np.isclose(lo_er ["fraction_required"], frac)].sort_values("observed")
        bg_s   = bg_er  [(bg_er ["n_samples"] == n) & np.isclose(bg_er ["fraction_required"], frac)].sort_values("observed")

        _pline(ax, pc_s["observed"], pc_s["int_dist_mean"], pc_s["ci_int_dist"],
               C["pc"], "PC", "-", "o")
        if not bpc_s.empty:
            _pline(ax, bpc_s["observed"], bpc_s["int_dist_mean"], bpc_s["ci_int_dist"],
                   C["bpc"], "b-PC (local)", "-", "s")
        if not bpcg_s.empty:
            _pline(ax, bpcg_s["observed"], bpcg_s["int_dist_mean"], bpcg_s["ci_int_dist"],
                   C["bpc_glob"], "b-PC (global)", "-", "D")
        _pline(ax, lo_s["observed"], lo_s["int_dist_without_b_mean"], lo_s["ci_int_dist_without_b"],
               C["load"], "LOAD", "--", "o")
        if not blo_s.empty:
            _pline(ax, blo_s["observed"], blo_s["int_dist_with_b_mean"], blo_s["ci_int_dist_with_b"],
                   C["bload_local"], "b-LOAD (local)", "--", "s")
        if not bg_s.empty:
            _pline(ax, bg_s["observed"], bg_s["int_dist_with_b_mean"], bg_s["ci_int_dist_with_b"],
                   C["bload_glob"], "b-LOAD (global)", "--", "^")

        _cap_ylim(ax, list(pc_s["int_dist_mean"]), list(lo_s["int_dist_without_b_mean"]),
                  list(bpc_s["int_dist_mean"]) if not bpc_s.empty else [],
                  list(bpcg_s["int_dist_mean"]) if not bpcg_s.empty else [],
                  list(blo_s["int_dist_with_b_mean"]) if not blo_s.empty else [],
                  list(bg_s["int_dist_with_b_mean"]) if not bg_s.empty else [])
        ax.set_title(f"n = {n} samples", fontsize=11)
        _fmt(ax, "observed (number of nodes)", legend_loc="upper left")

    plt.tight_layout()
    if save_prefix:
        fig_c.savefig(f"{save_prefix}_C.pdf", bbox_inches="tight")

    return fig_a, fig_b, fig_c
