# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

"""
This file contains all Bayesian Dirichlet (BD) related local scoring methods for Bayesian Network Structure scoring.

More specifically, this file contains:

    * Likelihood-equivalent uniform joint distribution Bayesian Dirichlet (BDeu) score
"""

# IMPORTS #
import math

from pandas import DataFrame
import numpy as np
from scipy.special import gammaln

from dag_scoring import ParallelBaseScore


class ParallelBDeuScore(ParallelBaseScore):
    """
    Class designed for DAG and Bayesian Network structure scoring based on the Bayesian Dirichlet
    log-equivalent uniform (BDeu) score.

    A BDeU score represents how well the existing architecture represents the existing data set. This
    score is, in addition, decomposable - meaning that differences in the local scores for each variable
    can be re-computed instead of having to compute the full score.

    This class is adapted to work in multi-process environment by using a dictionary cache with deltas,
    to share changes between processes.

    It is also assumed that there will be no missing values.

    Parameters
    ----------
    data: DataFrame
        The data over which the BDeU score will be computed. This data must be given as a pandas DataFrame where
        each row represents an instance of the dataset, with each column representing a variable (following
        the order of nodes)
    cache_dictionary: dict, optional
        Dictionary containing pre-computed local scores for a combination of node - parents input values
    equivalent_sample_size: int, default=10
        Equivalent sample size for the BDeu score computation.
    """
    # ATTRIBUTES #

    # SCORE COMPUTING #
    # Equivalent Sample Size
    esz: int

    # CONSTRUCTOR AND INITIALIZATION METHODS #
    def __init__(self, data, cache_dictionary=None, equivalent_sample_size=10):

        # Parent constructor call
        super().__init__(data, cache_dictionary)

        # Store the equivalent sample size
        self.esz = equivalent_sample_size

    # SCORE FUNCTIONS #

    def local_score(self, node, parents):
        """
        Computes the local BDeu score for a variable given a list of parents. In addition, returns whether
        the score was actually computed or looked up within the cache.

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
        tuple[float, bool]
            Local BDeu score and whether the score was actually computed (True) or looked up in the cache (False)
        """

        # Check the cache first
        if local_score := self.score_cache.get_score(node, parents):
            # If the score was already computed, return it without further computations
            return local_score, False
        else:
            # Compute the full score

            # PRE - PROCESSING #
            (variable_states, variable_length, parent_states,
             parent_length, parent_state_combinations, state_counts) = self._preprocess_node_parents(node, parents)

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

            return final_value, True

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
