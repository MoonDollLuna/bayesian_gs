# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from itertools import permutations

import networkx as nx
from pgmpy.models import BayesianNetwork
from pandas import DataFrame

from dag_architectures import ExtendedDAG
from data_processing import BDeUCache
from utils import LogManager

# TODO COMMENT, FINISH
class BaseAlgorithm:
    """
    `HillClimbing` implements a simple Greedy Search approach to Bayesian Network structure building.

    The algorithm works in a loop by trying all possible actions over the existing nodes (either adding
    a new edge to the DAG or removing or inverting an already existing one).

    These actions are evaluated by using a total BDeU score for the Bayesian Network based on the
    data provided, choosing the action that provides the biggest increase in BDeU score.

    This loop is continued until no action improves the BDeU score, at which point a fully constructed
    Bayesian Network based on the existing nodes and data is provided.

    This algorithm serves as a baseline, to which all other algorithms implemented will be compared to.

    Parameters
    ----------
    bayesian_network: BayesianNetwork
        Original bayesian network, used to measure the structure quality
    nodes: list
        List of nodes contained within the data, used to generate the DAG
    data: DataFrame
        Dataframe representing the data to be used when building the DAG
    """

    # ATTRIBUTES #

    # Bayesian network data #
    # Data, variables... from the original Bayesian network, used to build the new DAG

    # Original BN (used to compare structure results)
    bayesian_network: BayesianNetwork
    # Nodes contained within the data
    nodes: list
    # Data from which to generate a DAG
    data: DataFrame

    # Utilities #
    # Utilities to be used by the

    # Log manager
    log_manager: LogManager



    def __init__(self, bayesian_network, nodes, data):
        """
        Prepares all necessary data and structures for Greedy Search.

        `estimate_dag` generates a DAG optimizing the BDeU score for the specified data.

        Parameters
        ----------
        bayesian_network: BayesianNetwork
            Original bayesian network, used to measure the structure quality
        nodes: list
            List of nodes contained within the data, used to generate the DAG
        data: DataFrame
            Dataframe representing the data to be used when building the DAG
        """

        # Store the information
        self.bayesian_network = bayesian_network
        self.nodes = nodes
        self.data = data

        # Initialize the log manager
        # TODO - LOG MANAGER PATH
        self.log_manager = LogManager()