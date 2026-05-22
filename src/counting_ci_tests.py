"""
counting_ci_tests.py — Instrumented CI test wrapper that counts tests by order.

Provides :class:`CountingTest`, a thin wrapper around ``causallearn``'s
:class:`~causallearn.utils.cit.CIT` that tracks every conditional independence
test performed, keyed by the pair (X, Y) and the conditioning-set size (order).
This instrumentation is used to report the number of CI tests consumed per
algorithm run, broken down by conditioning-set order.
"""

from collections import defaultdict

import numpy as np
from causallearn.utils.cit import CIT


class CountingTest:
    """
    Wrapper around a ``causallearn`` CI test that counts every test performed.

    The count is accumulated in :attr:`tests_done`, a dict mapping each
    ``frozenset({X, Y})`` pair to the set of conditioning-set tuples that have
    been tested for that pair (deduplicating repeated calls with the same
    conditioning set).

    Use :meth:`get_tests_per_order` to retrieve a summary array indexed by
    conditioning-set size.

    Args:
        data (np.ndarray): Data matrix of shape (n_samples, n_nodes), passed
            directly to the underlying :class:`~causallearn.utils.cit.CIT`.
        ci_test (str): Name of the CI test to use (e.g. ``"fisherz"``,
            ``"chisq"``).  Must be a test supported by ``causallearn``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~causallearn.utils.cit.CIT`.
    """

    def __init__(
        self,
        data: np.ndarray,
        ci_test: str,
        **kwargs,
    ):
        self.cit = CIT(data, ci_test, **kwargs)
        # Pre-compute the precision matrix for Fisher-Z to avoid repeated
        # matrix inversions during the experiment.
        if ci_test == "fisherz":
            self.cit.precision_matrix = np.linalg.inv(self.cit.correlation_matrix)

        self.method = self.cit.method
        # tests_done[frozenset({X, Y})] = set of conditioning-set tuples tested
        self.tests_done: dict[frozenset, set] = defaultdict(set)

    def __call__(
        self,
        X: int,
        Y: int,
        condition_set: list[int] | None = [],
        *args,
        **kwargs,
    ) -> float:
        """
        Perform a conditional independence test of X ⊥ Y | condition_set and
        record it in :attr:`tests_done`.

        Args:
            X (int): Index of the first variable.
            Y (int): Index of the second variable.
            condition_set (list[int] | None): Indices of the conditioning
                variables.  ``None`` is treated as the empty set.

        Returns:
            float: p-value returned by the underlying CI test.  A p-value ≥ α
                indicates (conditional) independence.
        """
        if condition_set is None:
            condition_set = []
        self.tests_done[frozenset((X, Y))] |= {tuple(condition_set)}
        return self.cit(X, Y, condition_set)

    def get_tests_per_order(self) -> np.ndarray:
        """
        Summarise the CI tests performed, counted per conditioning-set order.

        Returns:
            np.ndarray: Integer array of length ``n_nodes`` where entry ``k``
                is the total number of CI tests performed with a conditioning
                set of size ``k``.
        """
        num_nodes = self.cit.data.shape[1]
        cond_sets = self.tests_done.values()
        if not cond_sets:
            return np.zeros(num_nodes, dtype=int)
        orders, test_num = np.unique(
            [len(cond) for conds in cond_sets for cond in conds],
            return_counts=True,
        )
        tests_per_order = np.zeros(num_nodes, dtype=int)
        tests_per_order[orders] = test_num
        return tests_per_order
