# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.


# IMPORTS #
import os.path

from numpy import ndarray
from pgmpy.readwrite.BIF import BIFReader
from pgmpy.models import BayesianNetwork
from pandas import DataFrame, read_csv

from dag_scoring import BaseScore, BDeuScore, LLScore, BICScore, AICScore
from utils import ResultsLogger

from typing import Optional


class BaseAlgorithm:
    """
    "BaseAlgorithm" provides a basic framework of methods that all DAG learning algorithms must follow.
    All DAG learning algorithms should extend this class.

    In addition, a basic constructor is provided with most of the data required to be stored. Further
    information more specific to each algorithm (such as metrics or hyperparameters) should be specified
    either as a `search` argument or as an extension of the constructor.

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
        Bayesian Network (or path to the BIF file describing it) of the used dataset,
        used for final measurements and comparisons (like the log likelihood of the dataset)
    score_method: {"bdeu"}
        Scoring method used to build the Bayesian Network.
        Any required arguments are passed through **score_arguments
    results_log_path: str, optional
        Path (without file name) to store the results logger file. If not specified, no logging will be done.
    results_bif_path: str, optional
        Path (without file name) to store the resulting DAG (as a bayesian network with estimated CPDs).
        If not specified, the resulting DAG will not be stored.
    results_file_name: str, optional
        Filename of the output data (both results log and resulting DAG).
        Only used if data is not specified as a CSV and if results_path or resulting_dag_path is not None.
        By default, uses the name of the CSV file (without the .CSV extension)
    results_flush_freq: int, default=300
        Time (in seconds) between results logger flushes / how often the file is written to.
    **score_arguments
        Arguments to provide to the scoring method. Currently, only BDeu is available as a scoring method.
    """

    # DATA #
    # Class associated to each score type
    _recognized_scorers = {
        "bdeu": BDeuScore,
        "ll": LLScore,
        "bic": BICScore,
        "aic": AICScore
    }

    # ATTRIBUTES #

    # Bayesian network data #
    # Data, variables... from the original Bayesian network, used to build the new DAG

    # Dataframe, used for PGMPY operations
    data: DataFrame
    # Nodes contained within the data
    nodes: list
    # Original BN (used to compare structure results)
    bayesian_network: BayesianNetwork

    # Data structures #
    # Data structures and utilities to be used during the algorithm execution

    # Score method used
    score_type: str
    # TODO - MAKE THIS EITHER BASESCORE OR BASEPARALLELSCORE
    # Local scorer
    local_scorer: BaseScore
    # Log manager
    results_logger: ResultsLogger

    # DAG storage #
    # Path to store the DAG
    dag_path: Optional[str]
    # File name of the resulting DAG BIF file
    dag_name: str

    # CONSTRUCTOR #

    def __init__(self, data, nodes=None, bayesian_network=None, score_method="bdeu",
                 results_log_path=None, results_bif_path=None, results_file_name=None, results_flush_freq=300,
                 **score_arguments):

        # Process the input data
        if isinstance(data, (str, DataFrame)):

            # Path to a CSV file or Pandas DataFrame
            # If a path to a CSV file is given, read the data from it and extract the input file name
            if isinstance(data, str):
                results_file_name = os.path.splitext(os.path.basename(data))[0]

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
                raise ValueError("A list of variable names must be given if a numpy array is passed as an argument.")

            # Create a Pandas dataframe from the given information
            self.data = DataFrame(data=data, columns=nodes)
            self.nodes = nodes

        else:
            raise TypeError("Data must be provided as a CSV file, a Pandas dataframe or a Numpy array.")

        # Check if an output name has been given, if required
        if results_log_path and not results_file_name:
            raise AttributeError("An output file name must be specified if results logging is used without specifying"
                                 "a path for the data.")

        # If a bayesian network is specified, parse it and store it for structure checks
        if isinstance(bayesian_network, str):
            self.bayesian_network = BIFReader(bayesian_network).get_model()
        else:
            self.bayesian_network = bayesian_network

        # Initialize the scorer class
        if score_method in self._recognized_scorers:
            self.score_type = score_method

            # BDeu requires an additional argument
            if score_method == "bdeu":
                equivalent_sample_size = score_arguments["bdeu_equivalent_sample_size"] if \
                    ("bdeu_equivalent_sample_size" in score_arguments) else 10
                self.local_scorer = self._recognized_scorers[score_method](self.data,
                                                                           equivalent_sample_size=equivalent_sample_size)
            else:
                self.local_scorer = self._recognized_scorers[score_method](self.data)
        else:
            raise TypeError("A valid scoring method must be specified")

        # Create the Results Logger. If no path is specified, only console logging will be considered
        self.results_logger = ResultsLogger(results_log_path, results_file_name, results_flush_freq)

        # If specified, store both DAG path and name
        self.dag_path = results_bif_path
        self.dag_name = results_file_name
