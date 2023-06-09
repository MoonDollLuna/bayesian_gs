# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.


# IMPORTS #
import os.path

from numpy import ndarray
from pgmpy.models import BayesianNetwork
from pandas import DataFrame, read_csv

from dag_scoring import ScoreCache, BDeuScore
from utils import ResultsLogger

from typing import Optional


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
    equivalent_sample_size: int, default=10
        Equivalent sample size used to compute BDeu scores.
    bdeu_score_method: {"forloop", "unique", "mask"}, default="unique"
        Method used to count state frequencies. Possible values:

            * "unique": np.unique over the sliced dataset
            * "forloop": Standard for loop
            * "mask": Masking to segment the dataset into smaller datasets with each parent state combinations

        "unique" should be used, other methods are kept for compatibility’s sake.
    results_path: str, optional
        Path to store the results logger file. If not specified, no logging will be done.
    input_file_name: str, optional
        Filename of the input data. Only used if data is not specified as a CSV and if results_path is not None.
    flush_frequency: int, default=300
        Time (in seconds) between results logger flushes / how often the file is written to.
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

    # Local score cache
    score_cache: ScoreCache
    # BDeU scorer
    bdeu_scorer: BDeuScore
    # Log manager
    log_manager: Optional[ResultsLogger]

    # CONSTRUCTOR #

    def __init__(self, data, nodes=None, bayesian_network=None,
                 equivalent_sample_size=10, bdeu_score_method="unique",
                 results_path=None, input_file_name=None, flush_frequency=300):

        # Process the input data and, if necessary, convert it into a numpy array
        if isinstance(data, (str, DataFrame)):
            # Path to a CSV file or Pandas DataFrame

            # If a path to a CSV file is given, read the data from it and extract the input file name
            if isinstance(data, str):
                input_file_name = os.path.splitext(os.path.basename(data))[0]
                data = read_csv(data)

            # Convert the data into a numpy array and extract the node names
            self.data = data.to_numpy(dtype='<U8')
            self.nodes = data.columns.values.tolist()
        elif isinstance(data, ndarray):
            # Numpy
            self.data = data

            # Nodes MUST be given as an argument
            if nodes is None:
                raise ValueError("A list of variable names must be given if a numpy array is passed as argument.")
            else:
                self.nodes = nodes
        else:
            raise TypeError("Data must be provided as a CSV file, a Pandas dataframe or a Numpy array.")

        # Check if an input name has been given, if required
        if results_path and not input_file_name:
            raise AttributeError("An input file name must be specified if results logging is used.")

        # If a bayesian network is specified, store it for structure checks
        self.bayesian_network = bayesian_network

        # Initialize the utility classes

        # Local score cache and BDeu scorer
        self.score_cache = ScoreCache()
        self.bdeu_scorer = BDeuScore(self.data, self.nodes,
                                     equivalent_sample_size=equivalent_sample_size,
                                     count_method=bdeu_score_method)

        # If required, create the Results Logger
        if results_path:
            self.log_manager = ResultsLogger(results_path, input_file_name, flush_frequency)
        else:
            self.log_manager = None

