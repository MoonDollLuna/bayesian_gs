# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

"""
This class contains all Information Criterion related local scoring methods for Bayesian Network Structure scoring.

More specifically, this class contains:

    * Log Likelihood (LL) score
    * Bayesian Information Criterion (BIC) score / Minimum Description Length (MDL) score
    * Bayesian Information Criterion (BIC) score / Minimum Description Length (MDL) score
    * Akaike Information Criterion (AIC) score
"""

# IMPORTS #
import math
from itertools import product
from functools import lru_cache

from pandas import DataFrame
import numpy as np

from dag_scoring import BaseScore


class LLScore(BaseScore):
    """
    Class designed for DAG and Bayesian Network structure scoring based on Log Likelihood (LL) score.

    The LL score represents how well the existing architecture represents the existing data set, based on
    log-likelihood probabilities. However, this method does not include any penalization for more complex
    BN structures.

    This score is, in addition, decomposable - meaning that differences in the local scores for each variable
    can be re-computed instead of having to compute the full score.

    This class is re-defined to speed up the implementation compared to pgmpy by using:

    - numpy vectorization with pandas speed-ups
    - lru_cache memoizing

    It is also assumed that there will be no missing values.

    Parameters
    ----------
    data: DataFrame
        The data over which the BDeU score will be computed. This data must be given as a pandas DataFrame where
        each row represents an instance of the dataset, with each column representing a variable (following
        the order of nodes)
    """
    # ATTRIBUTES #

    # CONSTRUCTOR AND INITIALIZATION METHODS #
    def __init__(self, data, equivalent_sample_size=10):

        # Parent constructor call
        super().__init__(data)

        # Store the equivalent sample size
        self.esz = equivalent_sample_size

    # SCORE FUNCTIONS #

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Computes the local BIC score for a variable given a list of parents.

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

        # BIC CALCULATION #

        # log (Nijk / Nij)
        # log (Nijk) - Compute the log value of all the counts, for each pair of node state - parents states
        # Values at 0 remain at 0 instead of being changed to minus infinity
        log_numerator = np.log(state_counts, where=state_counts > 0)

        # log (Nij) - Compute the log value of all the counts, for each combination of parent states
        # Values at 0 remain at 0 instead of being changed to minus infinity

        # Obtain the total count for each parent state combinations by collapsing the state counts
        parent_state_counts = np.sum(state_counts, axis=0)
        log_denominator = np.log(parent_state_counts, where=parent_state_counts > 0)

        # Nijk * log (Nijk / Nij) - Log-likelihood score
        # Substract the denominator from all rows of numerators, multiply each element by its state count
        # and then collapse the value
        log_value = (log_numerator - log_denominator) * state_counts
        log_likelihood_score = np.sum(log_value)

        return log_likelihood_score


class BICScore(BaseScore):
    """
    Class designed for DAG and Bayesian Network structure scoring based on the Bayesian Information Criterion (BIC) /
    Minimal Descriptive Length (MDL) score.

    The BIC score represents how well the existing architecture represents the existing data set, based on
    log-likelihood probabilities (the LL score) with penalty for network complexities.

    This score is, in addition, decomposable - meaning that differences in the local scores for each variable
    can be re-computed instead of having to compute the full score.

    This class is re-defined to speed up the implementation compared to pgmpy by using:

    - numpy vectorization with pandas speed-ups
    - lru_cache memoizing

    It is also assumed that there will be no missing values.

    Parameters
    ----------
    data: DataFrame
        The data over which the BDeU score will be computed. This data must be given as a pandas DataFrame where
        each row represents an instance of the dataset, with each column representing a variable (following
        the order of nodes)
    """
    # ATTRIBUTES #

    # CONSTRUCTOR AND INITIALIZATION METHODS #
    def __init__(self, data, equivalent_sample_size=10):

        # Parent constructor call
        super().__init__(data)

        # Store the equivalent sample size
        self.esz = equivalent_sample_size

    # SCORE FUNCTIONS #

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Computes the local BIC score for a variable given a list of parents.

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

        # BIC CALCULATION #

        # log (Nijk / Nij)
        # log (Nijk) - Compute the log value of all the counts, for each pair of node state - parents states
        # Values at 0 remain at 0 instead of being changed to minus infinity
        log_numerator = np.log(state_counts, where=state_counts > 0)

        # log (Nij) - Compute the log value of all the counts, for each combination of parent states
        # Values at 0 remain at 0 instead of being changed to minus infinity

        # Obtain the total count for each parent state combinations by collapsing the state counts
        parent_state_counts = np.sum(state_counts, axis=0)
        log_denominator = np.log(parent_state_counts, where=parent_state_counts > 0)

        # Nijk * log (Nijk / Nij) - Log-likelihood score
        # Substract the denominator from all rows of numerators, multiply each element by its state count
        # and then collapse the value
        log_value = (log_numerator - log_denominator) * state_counts
        log_likelihood_score = np.sum(log_value)

        # Bayesian Information Criterion (BIC)
        # The Log-likelihood score is multiplied by its entropy and how much parameters it needs to encode
        b_score = (variable_length - 1) * parent_length
        bic_score = log_likelihood_score - (0.5 * math.log(len(self.data)) * b_score)

        return bic_score


class AICScore(BaseScore):
    """
    Class designed for DAG and Bayesian Network structure scoring based on the Akaike Information Criterion (AIC)

    The AIC score represents how well the existing architecture represents the existing data set, based on
    log-likelihood probabilities with penalty for network complexities. This is based on the BIC score,
    using a different weight for the complexity (1 instead of -1/2log(N))

    This score is, in addition, decomposable - meaning that differences in the local scores for each variable
    can be re-computed instead of having to compute the full score.

    This class is re-defined to speed up the implementation compared to pgmpy by using:

    - numpy vectorization with pandas speed-ups
    - lru_cache memoizing

    It is also assumed that there will be no missing values.

    Parameters
    ----------
    data: DataFrame
        The data over which the BDeU score will be computed. This data must be given as a pandas DataFrame where
        each row represents an instance of the dataset, with each column representing a variable (following
        the order of nodes)
    """
    # ATTRIBUTES #

    # CONSTRUCTOR AND INITIALIZATION METHODS #
    def __init__(self, data, equivalent_sample_size=10):

        # Parent constructor call
        super().__init__(data)

        # Store the equivalent sample size
        self.esz = equivalent_sample_size

    # SCORE FUNCTIONS #

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Computes the local BIC score for a variable given a list of parents.

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

        # BIC CALCULATION #

        # log (Nijk / Nij)
        # log (Nijk) - Compute the log value of all the counts, for each pair of node state - parents states
        # Values at 0 remain at 0 instead of being changed to minus infinity
        log_numerator = np.log(state_counts, where=state_counts > 0)

        # log (Nij) - Compute the log value of all the counts, for each combination of parent states
        # Values at 0 remain at 0 instead of being changed to minus infinity

        # Obtain the total count for each parent state combinations by collapsing the state counts
        parent_state_counts = np.sum(state_counts, axis=0)
        log_denominator = np.log(parent_state_counts, where=parent_state_counts > 0)

        # Nijk * log (Nijk / Nij) - Log-likelihood score
        # Substract the denominator from all rows of numerators, multiply each element by its state count
        # and then collapse the value
        log_value = (log_numerator - log_denominator) * state_counts
        log_likelihood_score = np.sum(log_value)

        # Bayesian Information Criterion (BIC)
        # The Log-likelihood score is multiplied by its entropy and how much parameters it needs to encode
        b_score = (variable_length - 1) * parent_length
        aic_score = log_likelihood_score - b_score

        return aic_score
