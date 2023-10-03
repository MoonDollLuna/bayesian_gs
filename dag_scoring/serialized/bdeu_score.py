# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import math
from itertools import product
from functools import lru_cache

import pandas as pd
import numpy as np
from scipy.special import gammaln
from pgmpy.base import DAG

from dag_scoring import BaseScore


class BDeuScore(BaseScore):
    """
    Class designed for DAG and Bayesian Network structure scoring based on the Bayesian Dirichlet
    log-equivalent uniform (BDeu) score.

    A BDeU score represents how well the existing architecture represents the existing data set. This
    score is, in addition, decomposable - meaning that differences in the local scores for each variable
    can be re-computed instead of having to compute the full score.

    This class is re-defined to speed up the implementation compared to pgmpy by using:

    - numpy vectorization
    - lru_cache memoizing
    - pandas speed-ups

    It is also assumed that there will be no missing values.

    Parameters
    ----------
    data: pd.DataFrame
        The data over which the BDeU score will be computed. This data must be given as a pandas DataFrame where
        each row represents an instance of the dataset, with each column representing a variable (following
        the order of nodes)
    equivalent_sample_size: int, default=10
        Equivalent sample size for the BDeu score computation.
    """
    # ATTRIBUTES #

    # DATA HANDLING #
    # Data contained within the scorer
    data: pd.DataFrame
    # Possible values for each node
    # Shape: {variable_name: (value_1, value_2...)}
    node_values: dict

    # SCORE COMPUTING #
    # Equivalent Sample Size
    esz: int

    # CONSTRUCTOR AND INITIALIZATION METHODS #
    def __init__(self, data, equivalent_sample_size=10):

        # Store the input data and initialize the dictionary
        self.data = data

        # Initialize the dictionary using dictionary comprehension
        self.node_values = {variable: list(self.data[variable].unique()) for variable in list(self.data.columns)}

        # Store the equivalent sample size
        self.esz = equivalent_sample_size

    # SCORE FUNCTIONS #

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Computes the local BDeu score for a variable given a list of parents.

        This code is based on pgmpy's BDeu Scorer implementation, but modified to:

        - Speed up the calculation by using a numpy array instead of a Pandas dataframe.
        - Include the prior probability of the variable having said parents.

        This function is cached, meaning that repeated calls with the same arguments will only be run once.

        Parameters
        ----------
        node: str
            Child variable of which the BDeu score is being computed
        parents: tuple[str]
            List of parent variables that influence the child variable. If the variables
            have no parents, an empty list must be passed instead.

        Returns
        -------
        float
            Local BDeu score
        """

        # PRE - PROCESSING #

        # Get the variable states and number of possible values
        variable_states = self.node_values[node]
        variable_length = len(variable_states)

        # Generate a list with all possible parent values (per variable)
        # (if no parents are specified, the parent length is assumed to be 1)
        parent_states = [self.node_values[parent] for parent in parents]
        parent_length = math.prod([len(parent) for parent in parent_states])

        # Generate all possible combinations of parent states
        parent_state_combinations = list(product(*parent_states))

        # Get the count of each variable state for each combination of parents
        state_counts = self._get_state_counts(node, variable_states, parents, parent_state_combinations)

        # BDEU CALCULATION #

        # Compute the prior probability
        prior_value = self._get_local_prior_probability(len(parents))

        # Compute constants #
        # Alpha - (equivalent sample size / number of parent states)
        alpha = self.esz / parent_length
        # Beta - (equivalent sample size / number of child and parent states)
        beta = self.esz / (parent_length * variable_length)

        # SECOND TERM (variable term) = (ln(gamma(state counts + beta) / gamma(beta)) #
        # Instead of applying log to each division, sum(nominators) - sum(denominators) will be applied
        # for vectorization

        # Numerator (ln (gamma(state counts + beta))
        # Apply ln(gamma) to each (state count + beta)
        log_gamma_state_counts = gammaln(state_counts + beta)
        # Add up all values
        variable_numerator = np.sum(log_gamma_state_counts)

        # Denominator (ln (gamma(beta))
        # Compute the log gamma value and multiply it for all existing state combinations
        variable_denominator = math.lgamma(beta) * (variable_length * parent_length)

        # Subtract both terms to obtain the final log result
        variable_value = variable_numerator - variable_denominator

        # FIRST TERM (parents term) = (ln(gamma(alpha) / gamma(parents state counts + alpha))
        # Instead of applying log to each division, sum(nominators) - sum(denominators) will be applied
        # for vectorization

        # Obtain the total count for each parent state combinations by collapsing the state counts
        parent_state_counts = np.sum(state_counts, axis=0)

        # Numerator (ln (gamma(alpha))
        # Compute the log gamma value and multiply it for all existing parent state combinations
        parent_numerator = math.lgamma(alpha) * parent_length

        # Denominator (ln (gamma(parents state counts + alpha))
        # Apply ln(gamma) to each (parent state count + alpha)
        log_gamma_parent_state_counts = gammaln(parent_state_counts + alpha)
        # Add up all values
        parent_denominator = np.sum(log_gamma_parent_state_counts)

        # Subtract both terms to obtain the final log result
        parent_value = parent_numerator - parent_denominator

        # FINAL BDEU SCORE #

        # Given the prior probability, the parents term and the variable term, return the sum
        final_value = prior_value + variable_value + parent_value
        return final_value

    def global_score(self, dag):
        """
        Computes the global BDeu score of the specified DAG.

        The global score of a DAG is the sum of local scores for each variable and its parents.

        Parameters
        ----------
        dag: DAG
            A directed acyclic graph

        Returns
        -------
        float
            Global BDeu score
        """

        score = 0.0

        # For each variable, find its parents and compute the local score
        for variable in list(dag.nodes):
            parents = dag.get_parents(variable)
            score += self.local_score(variable, tuple(parents))

        return score

    def local_score_delta(self, node, original_parents, new_parents, *args, **kwargs):
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

    # UTILITY FUNCTIONS #

    def _get_local_prior_probability(self, parents_amount):
        """
        Computes the prior probability of a variable having the specified list of parents.

        Essentially, computes how likely this family would be, penalizing variables with an excessive
        amount of parents.

        This implementation is based on Tetrad's BDeu scorer.

        Parameters
        ----------
        parents_amount: int
            Number of parents contained by the variable

        Returns
        -------
        float
            Local prior probability
        """

        # A constant "structure prior" probability of 1.0 is used as part of the BDeu assumptions.
        structure_prior = 1.0
        # A "sample size" of the number of rows is used
        sample_size = self.data.shape[0] - 1

        # The prior probability has two terms:
        # - How much information is contained within the dag
        # - How likely is that this node has this amount of parents within this DAG
        dag_likelihood = parents_amount * math.log(structure_prior / sample_size)
        variable_likelihood = (sample_size - parents_amount) * math.log(1 - (structure_prior / sample_size))

        return dag_likelihood + variable_likelihood

    def _get_state_counts(self, variable, variable_states, parents, parent_state_combinations):
        """
        For each combination of parent states, returns the count of each variable state.

        This method utilizes masks to filter the numpy array.

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
        # Dropna is set to false to avoid looking for NaNs
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
