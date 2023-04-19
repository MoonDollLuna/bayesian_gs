# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import networkx as nx
from pgmpy.models import BayesianNetwork
from pandas import DataFrame

from dag_architectures import ExtendedDAG
from data_processing import BDeUCache
from utils import LogManager


class GreedySearch:
    """
    `GreedySearch` implements a simple Greedy Search approach to Bayesian Network structure building.

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

    # Parameters #
    # Parameters to be used by the Greedy Search algorithm

    # Epsilon - the search stops once the difference in BDeU score does not improve above the epsilon threshold
    epsilon: float

    # Utilities #
    # This includes utilities like the log manager or the BDeU cache

    # BDeU cache
    bdeu_cache: BDeUCache

    # Log manager
    log_manager: LogManager

    # Data analysis management #
    # These attributes are stored to be shared with the Log Manager to print
    # and analyze the results of the algorithm

    # Total operations "tried"
    # This may include operations for which the BDeU score was previously known
    total_operations: int

    # Operations that needed to compute a new BDeU score
    computed_operations: int

    # Total time of operation for the algorithm
    time_taken: float

    def __init__(self, bayesian_network, nodes, data):
        """
        Prepares all necessary data and structures for Greedy Search.

        `estimate_DAG` generates a DAG optimizing the BDeU score for the specified data.

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

    def estimate_DAG(self, starting_DAG=None, epsilon=0.0001, max_iterations=1e6):
        """
        TODO FINISH
        TODO AVOID CYCLES
        Performs Greedy Search to generate
        Parameters
        ----------
        starting_DAG
        epsilon
        max_iterations

        Returns
        -------

        """

        # TODO INITIALIZE VARIABLES AND BDEU CACHE
        pass

    # TODO LEGAL OPERATIONS