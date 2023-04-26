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


def find_legal_operations(dag):
    """
    Given a DAG and a set of variables, find all legal operations, returning three sets containing:
        - All possible edges to add.
        - All possible edges to remove.
        - All possible edges to invert.

    This takes care of avoiding possible cycles.

    Parameters
    ----------
    dag: ExtendedDAG
        DAG over which the operations are tried

    Returns
    -------
    tuple[set, set, set]
    """

    # Get the list of nodes from the DAG
    nodes = list(dag.nodes())

    # EDGE ADDITIONS #

    # Generate the initial set of possible additions (all possible permutations of nodes)
    add_edges = set(permutations(nodes, 2))

    # Remove invalid edge additions
    # Remove existing edges
    add_edges = add_edges - set(dag.edges())
    # Remove inverted edges that already exist
    add_edges = add_edges - set([(Y, X) for (X, Y) in dag.edges()])
    # Remove edges that can lead to a cycle
    add_edges = add_edges - set([(X, Y) for (X, Y) in add_edges if nx.has_path(dag, Y, X)])

    # EDGE REMOVALS #

    # Generate the initial set of possible removals (only the existing edges)
    remove_edges = set(dag.edges())

    # EDGE INVERSIONS

    # Generate the initial set of possible removals (only the existing edges)
    invert_edges = set(dag.edges())

    # Remove the edges that, when inverted, would lead to a cycle
    invert_edges = invert_edges - set([(X, Y) for (X, Y) in invert_edges if not any(map(lambda path: len(path) > 2, nx.all_simple_paths(dag, X, Y)))])

    return add_edges, remove_edges, invert_edges


class HillClimbing:
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

    def estimate_dag(self, starting_dag=None, epsilon=0.0001, max_iterations=1e6, silent=False):
        """
        TODO FINISH
        Performs Hill Climbing to find a local best DAG based on BDeU.

        Note that the found DAG may not be optimal, but good enough.

        Parameters
        ----------
        starting_dag: ExtendedDAG, optional
            Starting DAG. If not specified, an empty DAG is used.
        epsilon: float
            BDeU threshold. If the BDeU does not improve above the specified threshold,
            the algorithm stops
        max_iterations: int
            Maximum number of iterations to perform.
        silent: bool
            Whether warnings and loading screens should be printed on the screen or ignored.
            A log will be written regardless

        Returns
        -------
        ExtendedDAG
        """

        # LOCAL VARIABLE DECLARATION #

        # PARAMETER INITIALIZATION #

        # Store the DAG and, if necessary, create an empty one with the existing nodes
        if starting_dag:
            dag = starting_dag
        else:
            dag = ExtendedDAG(self.nodes)







