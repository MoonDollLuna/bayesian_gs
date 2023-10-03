# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.


# IMPORTS #
import os.path

from numpy import ndarray
from pgmpy.models import BayesianNetwork
from pandas import DataFrame, read_csv

from dag_scoring import BaseScore, BDeuScore
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
    score_method: {"bdeu"}
        Scoring method used to build the Bayesian Network.
        Any required arguments are passed through **score_arguments
    results_path: str, optional
        Path (without file name) to store the results logger file. If not specified, no logging will be done.
    resulting_bif_path: str, optional
        Path (without file name) to store the resulting DAG (as a bayesian network with estimated CPDs).
        If not specified, the resulting DAG will not be stored.
    output_file_name: str, optional
        Filename of the output data (both results log and resulting DAG).
        Only used if data is not specified as a CSV and if results_path or resulting_dag_path is not None.
        By default, uses the name of the CSV file (without the .CSV extension)
    flush_frequency: int, default=300
        Time (in seconds) between results logger flushes / how often the file is written to.
    **score_arguments
        Arguments to provide to the scoring method. Currently, only BDeu is available as a scoring method.
    """

    # ATTRIBUTES #

    # Bayesian network data #
    # Data, variables... from the original Bayesian network, used to build the new DAG

    # Dataframe, used for PGMPY operations
    data: DataFrame or None
    # Nodes contained within the data
    nodes: list
    # Original BN (used to compare structure results)
    bayesian_network: BayesianNetwork

    # Data structures #
    # Data structures and utilities to be used during the algorithm execution

    # Score method used
    score_type: str
    # Local scorer
    local_scorer: BaseScore
    # Log manager
    results_logger: Optional[ResultsLogger]

    # DAG storage #
    # Path to store the DAG
    dag_path: Optional[str]
    # File name of the resulting DAG BIF file
    dag_name: str

    # CONSTRUCTOR #

    def __init__(self, data, nodes=None, bayesian_network=None, score_method="bdeu",
                 results_path=None, output_file_name=None, flush_frequency=300,
                 resulting_bif_path=None, **score_arguments):

        # Process the input data
        if isinstance(data, (str, DataFrame)):

            # Path to a CSV file or Pandas DataFrame
            # If a path to a CSV file is given, read the data from it and extract the input file name
            if isinstance(data, str):
                output_file_name = os.path.splitext(os.path.basename(data))[0]

                # Data is always read as string, to avoid data type sniffing
                # In addition, no values are interpreted as NaN to avoid converting Nones into NaNs
                data = read_csv(data, dtype=str, keep_default_na=False)

            # Store the data and extract the nodes
            self.data = data
            self.nodes = list(self.data.columns)

        elif isinstance(data, ndarray):
            # Numpy
            # Nodes MUST be given as an argument
            if nodes is None:
                raise ValueError("A list of variable names must be given if a numpy array is passed as argument.")

            # Create a Pandas dataframe from the given information
            self.data = DataFrame(data=data, columns=nodes)
            self.nodes = nodes

        else:
            raise TypeError("Data must be provided as a CSV file, a Pandas dataframe or a Numpy array.")

        # Check if an output name has been given, if required
        if results_path and not output_file_name:
            raise AttributeError("An output file name must be specified if results logging is used.")

        # If a bayesian network is specified, store it for structure checks
        self.bayesian_network = bayesian_network

        # Initialize the scorer class
        if score_method == "bdeu":
            self.score_type = "bdeu"
            equivalent_sample_size = score_arguments["bdeu_equivalent_sample_size"] if \
                ("bdeu_equivalent_sample_size" in score_arguments) else 10

            self.local_scorer = BDeuScore(self.data, equivalent_sample_size=equivalent_sample_size)
        else:
            # Currently, only BDeu is implemented
            raise NotImplementedError("Only BDeu scoring is currently available")

        # If required, create the Results Logger
        if results_path:
            self.results_logger = ResultsLogger(results_path, output_file_name, flush_frequency)
        else:
            self.results_logger = None

        # If specified, store both DAG path and name
        self.dag_path = resulting_bif_path
        self.dag_name = output_file_name
