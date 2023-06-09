# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from numpy import ndarray
from pgmpy.models import BayesianNetwork
from pandas import DataFrame, read_csv

from dag_scoring import ScoreCache, BDeuScore
from utils import ResultsLogger


class BaseAlgorithm:
    """
    "BaseAlgorithm" provides a basic framework of methods that all DAG learning algorithms must follow.
    All DAG learning algorithms should extend this class.

    In addition, a basic constructor is provided with most of the data required to be stored. Further
    information more specific to each algorithm (such as metrics or hyperparameters) should be specified
    either as a `build_dag` argument or as an extension of the constructor.

    Parameters
    ----------
    data: str, DataFrame or ndarray
        Dataset or location of the dataset from which the DAG will be built. The following can be specified:

        - A path to a .csv file
        - A DataFrame containing the data and variable names
        - A numpy Array containing the data

        If a numpy array is specified, the variable names MUST be passed as argument.
    nodes: list[str], optional
        List of ordered variable names contained within the data.
        This argument is ignored unless a numpy Array is given as data - in which case, it is mandatory.
    bayesian_network: BayesianNetwork or str, optional
        Bayesian Network (or path to the BIF file describing it) used for final measurements (like
        the log likelihood of the dataset)
    """

    # ATTRIBUTES #

    # Bayesian network data #
    # Data, variables... from the original Bayesian network, used to build the new DAG

    # Data from which to generate a DAG
    data: str or ndarray or DataFrame
    # Nodes contained within the data
    nodes: list
    # Original BN (used to compare structure results)
    bayesian_network: BayesianNetwork

    # Data structures #
    # Data structures and utilities to be used during the algorithm execution

    # Log manager
    log_manager: ResultsLogger
    # Local score cache
    score_cache: ScoreCache
    # BDeU scorer
    bdeu_scorer: BDeuScore

    # CONSTRUCTOR #

    def __init__(self, data, nodes=None, bayesian_network=None):

        # Process the input data and, if necessary, convert it into a numpy array
        if isinstance(data, ndarray):
            # Numpy
            self.data = data

            # Nodes MUST be given as an argument
            if nodes is None:
                raise ValueError("A list of variable names must be given if a numpy array is passed as argument.")
            else:
                self.nodes = nodes
        elif isinstance(data, (str, DataFrame)):
            # Path to a CSV file or Pandas DataFrame

            # If a path to a CSV file is given, read the data from it
            if isinstance(data, str):
                data = read_csv(data)

            # Convert the data into a numpy array and extract the node names
            self.data = data.to_numpy(dtype='<U8')
            self.nodes = data.columns.values.tolist()

        else:
            raise TypeError("Data must be provided as a CSV file, a Pandas dataframe or a Numpy array.")

        # If a bayesian network is specified, store it for structure checks
        self.bayesian_network = bayesian_network

        # Initialize the utility classes

        # Local score cache and BDeu scorer
        self.score_cache = ScoreCache()
        self.bdeu_scorer = BDeuScore(self.data, self.nodes)
        # TODO - LOG MANAGER PATH
        self.log_manager = ResultsLogger()

