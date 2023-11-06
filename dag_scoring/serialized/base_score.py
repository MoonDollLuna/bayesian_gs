# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS
from functools import lru_cache

import numpy as np
import pandas as pd


class BaseScore:
    """
    Abstract base class for serialized DAG and Bayesian Network structure scoring.

    All scoring serialized methods implemented (BDeu, BIC...) must extend this class and implement
    the `local_score` method, using lru_cache for memoization.

    Other utility methods, such as `local_score_delta` or `global_score` are already provided and
    reliant on `local_score` implementation.

    Parameters
    ----------
    data: pd.DataFrame
        The data over which the BDeU score will be computed. This data must be given as a pandas DataFrame where
        each row represents an instance of the dataset, with each column representing a variable (following
        the order of nodes)
    """

    # DATA HANDLING #

    # Data contained within the scorer
    data: pd.DataFrame

    # Possible values for each node
    # Shape: {variable_name: (value_1, value_2...)}
    node_values: dict

    # CONSTRUCTOR AND INITIALIZATION METHODS #
    def __init__(self, data):

        # Store the input data and initialize the dictionary
        self.data = data

        # Initialize the dictionary using dictionary comprehension
        self.node_values = {variable: list(self.data[variable].unique()) for variable in list(self.data.columns)}

    # SCORING METHODS #

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Given a variable and a set of parents, return the local score for said combination.
        """

        raise NotImplementedError

    def local_score_delta(self, node, original_parents, new_parents):
        """
        Given a node and the original and new set of parents (after an operation to add, remove
        or invert and edge), computes the difference in local score between the new and old set of parents.

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
        float
        """

        # Compute the BDeu score for the original parents
        original_score = self.local_score(node, original_parents)

        # Compute the BDeu score for the new parents
        new_score = self.local_score(node, new_parents)

        return new_score - original_score

    def global_score(self, dag):
        """
        Computes the global local score of the specified DAG. The global score of a DAG is
        the sum of local scores for each variable and its parents.

        Parameters
        ----------
        dag: DAG
            A directed acyclic graph

        Returns
        -------
        float
        """

        score = 0.0

        # For each variable, find its parents and compute the local score
        for variable in list(dag.nodes):
            parents = dag.get_parents(variable)
            score += self.local_score(variable, tuple(parents))

        return score

    # UTILITY METHODS #

    def _get_state_counts(self, variable, variable_states, parents, parent_state_combinations):
        """
        For each combination of parent states, returns the count of each variable state.

        This method uses pandas' `value_counts` method to efficiently count the unique instances.

        Parameters
        ----------
        variable : str
            Variable of which the states are counted
        variable_states : list[str]
            List of all states of the variable
        parents : list[str] or tuple[str]
            List of parents of the variable
        parent_state_combinations : list[tuple[str]]
            List of all possible combination of parent variables' states

        Returns
        -------
        np.ndarray
            Array where each row represents a state of the variable, each column represents
            a combination of parent variable states and each cell represents the count of said
            variable state for the given parents
        """

        # Initialize the numpy array to be returned
        counts_array = np.zeros((len(variable_states), len(parent_state_combinations)))

        # Generate dictionaries to know the index of each element (both variable and parent combination)
        variable_states_dict = {state: index for index, state in enumerate(variable_states)}
        parent_states_dict = {state_combination: index
                              for index, state_combination
                              in enumerate(parent_state_combinations)}

        # Count the occurrences of each combination of variable and parent states using pandas value_counts
        # Dropna is set to False to avoid looking for NaNs
        subset = [variable] + list(parents) if parents else variable
        counts = self.data.value_counts(subset=subset, dropna=False)

        # Move the counts into the count array
        # Row: index of the original variable state (first element of each index)
        # Column: index of the parents state combination (rest of the elements of each index)
        # Count: number of instances of said combination

        # Variables without parents have to be treated differently
        if not parents:
            for column, count in [(variable_states_dict[comb], counts[comb]) for comb in counts.index]:
                # Update the count
                counts_array[column] = count
        else:
            for row, column, count in [(variable_states_dict[comb[0]],
                                        parent_states_dict[comb[1:]],
                                        counts[comb]) for comb in counts.index]:
                # Update the count
                counts_array[row, column] = count

        return counts_array
