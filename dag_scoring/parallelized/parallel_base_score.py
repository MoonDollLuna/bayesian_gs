# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS
from functools import lru_cache
from itertools import product
import math

import numpy as np
import pandas as pd

from dag_scoring import BaseScore
from data_caches import ParallelScoreCache


class ParallelBaseScore(BaseScore):
    """
    Abstract base class for parallelized DAG and Bayesian Network structure scoring.

    The key difference between this class and the standard serialized scorers is the inclusion of a
    `ParallelScoreCache` (since `lru_cache` cannot be shared between separate processes) to handle
    memoizing of call inputs.

    In order to access the cache methods, the cache should directly be accessed by using
    object_name.score_cache.method()

    All scoring serialized methods implemented (BDeu, BIC...) must extend this class and implement
    the `local_score` method, using the `parallel_score_cache` class for method input caching.

    Other utility methods, such as `local_score_delta` or `global_score` are already provided and
    reliant on `local_score` implementation.

    Parameters
    ----------
    data: pd.DataFrame
        The data over which the BDeU score will be computed. This data must be given as a pandas DataFrame where
        each row represents an instance of the dataset, with each column representing a variable (following
        the order of nodes)
    cache_dictionary: dict, optional
        Dictionary containing pre-computed local scores for a combination of node - parents input values
    """

    # DATA HANDLING #

    # Data contained within the scorer
    data: pd.DataFrame

    # Possible values for each node
    # Shape: {variable_name: (value_1, value_2...)}
    node_values: dict

    # Cache for input memoizing
    score_cache: ParallelScoreCache

    # CONSTRUCTOR AND INITIALIZATION METHODS #
    def __init__(self, data, cache_dictionary=None):
        # Parent call
        super().__init__(data)

        # Initialize the internal cache
        self.score_cache = ParallelScoreCache(cache_dictionary)

    # SCORING METHODS #
    def local_score(self, node, parents):
        """
        Given a variable and a set of parents, return the local score for said combination.

        In addition to the local score, the method must return a bool value:

            * True if the score was actually computed
            * False if the local score cache was used instead

        Returns
        -------
        tuple[float, bool]
        """

        raise NotImplementedError

    def local_score_delta(self, node, original_parents, new_parents):
        """
        Given a node and the original and new set of parents (after an operation to add, remove
        or invert and edge), computes the difference in local score between the new and old set of parents.

        In addition, returns the number of operations that were computed and the total number of operations (2)

        Parameters
        ----------
        node: str
            The children node
        original_parents: tuple[str]
            The original parents of node
        new_parents: tuple[str]
            The new parents of node after the operation

        Returns
        -------
        tuple[float, int, int]
        """

        # Compute the BDeu score for the original parents
        original_score, original_computed = self.local_score(node, original_parents)

        # Compute the BDeu score for the new parents
        new_score, new_computed = self.local_score(node, new_parents)

        return new_score - original_score, int(original_computed) + int(new_computed), 2

    def global_score(self, dag):
        """
        Computes the global local score of the specified DAG. The global score of a DAG is
        the sum of local scores for each variable and its parents.

        In addition, returns the number of operations that were computed and the total number of operations

        Parameters
        ----------
        dag: DAG
            A directed acyclic graph

        Returns
        -------
        tuple[float, int, int]
        """

        score = 0.0
        computed_operations = 0

        nodes_list = list(dag.nodes)

        # For each variable, find its parents and compute the local score
        for variable in nodes_list:
            parents = dag.get_parents(variable)
            local_score, computed = self.local_score(variable, tuple(parents))

            score += local_score
            computed_operations += int(computed)

        return score, computed_operations, len(nodes_list)
