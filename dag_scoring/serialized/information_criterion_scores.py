# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

"""
This file contains all Information Criterion related local scoring methods for Bayesian Network Structure scoring.

More specifically, this file contains:

    * Log Likelihood (LL) score
    * Bayesian Information Criterion (BIC) score / Minimum Description Length (MDL) score
    * Akaike Information Criterion (AIC) score
"""

# IMPORTS #
import math
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

    # SCORE FUNCTIONS #

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Computes the local LL score for a variable given a list of parents.

        This function is cached, meaning that repeated calls with the same arguments will only be run once.

        Parameters
        ----------
        node: str
            Child variable of which the LL score is being computed
        parents: tuple[str]
            List of parent variables that influence the child variable. If the variables
            have no parents, an empty list must be passed instead.

        Returns
        -------
        float
            Local LL score
        """

        # PRE - PROCESSING #
        (variable_states, variable_length, parent_states,
         parent_length, parent_state_combinations, state_counts) = self._preprocess_node_parents(node, parents)

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

    # SCORE FUNCTIONS #

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Computes the local BIC score for a variable given a list of parents.

        This code is based on pgmpy's BIC Scorer implementation, but modified to:

        - Speed up the calculation by using a numpy array instead of a Pandas dataframe.
        - Include the prior probability of the variable having said parents.

        This function is cached, meaning that repeated calls with the same arguments will only be run once.

        Parameters
        ----------
        node: str
            Child variable of which the BIC score is being computed
        parents: tuple[str]
            List of parent variables that influence the child variable. If the variables
            have no parents, an empty list must be passed instead.

        Returns
        -------
        float
            Local BIC score
        """

        # PRE - PROCESSING #
        (variable_states, variable_length, parent_states,
         parent_length, parent_state_combinations, state_counts) = self._preprocess_node_parents(node, parents)

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

    # SCORE FUNCTIONS #

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Computes the local AIC score for a variable given a list of parents.

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
            Local AIC score
        """

        # PRE - PROCESSING #
        (variable_states, variable_length, parent_states,
         parent_length, parent_state_combinations, state_counts) = self._preprocess_node_parents(node, parents)

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
