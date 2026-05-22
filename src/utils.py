"""
utils.py — Graph visualisation utilities for debugging and exploration.

These helpers render PDAG / MPDAG adjacency matrices and NetworkX DAGs using
matplotlib and NetworkX, and are intended for interactive / notebook use only.
They are not called by any core algorithm and carry no attribution requirement.

Graph encoding reminder (internal format used throughout this codebase)
-----------------------------------------------------------------------
``g[i, j] == -1``  →  tail at i  (i — j or i → j)
``g[i, j] ==  1``  →  head at i  (j → i)
``g[i, j] ==  0``  →  no edge
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_ground_truth(experiments: dict, exp_id) -> None:
    """
    Draw the ground-truth DAG for a single experiment using a circular layout.

    Args:
        experiments (dict): Dictionary of experiments keyed by experiment ID,
            each value containing at least a ``"true_dag"`` key holding a
            ``nx.DiGraph``.
        exp_id: Key identifying the experiment to visualise.
    """
    true_dag = experiments[exp_id]["true_dag"]
    nx.draw_circular(true_dag, with_labels=True)


def draw_ground_truth_from_dag(true_dag: nx.DiGraph) -> None:
    """
    Draw a ground-truth DAG directly from a ``nx.DiGraph`` using a circular
    layout.

    Args:
        true_dag (nx.DiGraph): The DAG to visualise.
    """
    nx.draw_circular(true_dag, with_labels=True)


def draw_graph_from_array(array: np.ndarray) -> None:
    """
    Draw a directed graph from a plain adjacency matrix (non-zero entry = edge)
    using a circular layout.

    Note: this function does not distinguish between directed and undirected
    edges — every non-zero entry is rendered as a directed arrow.  Use
    :func:`draw_complex_graph` for graphs that contain both directed and
    undirected edges (i.e., PDAGs / MPDAGs).

    Args:
        array (np.ndarray): Square adjacency matrix.  A non-zero entry at
            ``[i, j]`` is interpreted as a directed edge i → j.
    """
    plt.figure()
    nx.draw_circular(nx.DiGraph(array), with_labels=True)
    plt.show()


def draw_complex_graph(array: np.ndarray) -> None:
    """
    Draw a PDAG / MPDAG from its internal adjacency matrix, correctly
    rendering both directed and undirected edges.

    Directed edges (``array[i,j] == 1``, i.e., arrowhead at i ← j) are drawn
    as arrows; undirected edges (``array[i,j] == array[j,i] == -1``) are drawn
    as plain lines without arrowheads.

    Args:
        array (np.ndarray): PDAG / MPDAG adjacency matrix using the internal
            encoding:

            * ``array[i, j] ==  1``  →  arrowhead at i  (j → i)
            * ``array[i, j] == -1``  →  tail at i  (i — j or i → j)
            * ``array[i, j] ==  0``  →  no edge
    """
    arr = np.array(array)
    G_directed = nx.DiGraph()
    G_undirected = nx.Graph()

    nodes = range(len(arr))
    G_directed.add_nodes_from(nodes)
    G_undirected.add_nodes_from(nodes)

    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i, j] == 1:
                G_directed.add_edge(j, i)  # arrowhead at i means edge from j to i
            elif arr[i, j] == -1 and arr[j, i] == -1 and i < j:
                G_undirected.add_edge(i, j)  # undirected: add only once

    pos = nx.circular_layout(G_directed)
    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G_directed, pos, node_color="skyblue")
    nx.draw_networkx_labels(G_directed, pos)
    nx.draw_networkx_edges(G_directed, pos, edgelist=G_directed.edges(), arrows=True)
    nx.draw_networkx_edges(
        G_undirected, pos, edgelist=G_undirected.edges(), arrows=False
    )

    plt.show()
