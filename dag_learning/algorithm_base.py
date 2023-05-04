# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BDeuScore
from pandas import DataFrame

from data_processing import BDeUCache
from utils import LogManager


class BaseAlgorithm:
    """
    "BaseAlgorithm" provides a basic framework of methods that all DAG learning algorithms must follow.
    All DAG learning algorithms should extend this class.

    In addition, a basic constructor is provided with most of the data required to be stored. Further
    information more specific to each algorithm (such as metrics or hyperparameters) should be specified
    either as a `build_dag` argument or as an extension of the constructor.

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

    # Data structures #
    # Data structures and utilities to be used during the algorithm execution

    # Log manager
    log_manager: LogManager
    # BDeU cache
    bdeu_cache: BDeUCache
    # BDeU scorer
    bdeu_scorer: BDeuScore

    # CONSTRUCTOR #

    def __init__(self, bayesian_network, nodes, data):
        """
        Prepares all necessary data and structures for a DAG building algorithm.

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
        # TODO MAKE BAYESIAN NETWORK OPTIONAL
        self.bayesian_network = bayesian_network
        self.nodes = nodes
        # TODO ALLOW DATA TO BE READ FROM A CSV
        self.data = data

        # Initialize the utility classes
        # TODO - LOG MANAGER PATH
        self.log_manager = LogManager()
        self.bdeu_cache = BDeUCache()
        self.bdeu_scorer = BDeuScore(self.data)
