# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import numpy as np
from pandas import DataFrame


class BDeuScore:
    """
    Class designed for DAG and Bayesian Network structure scoring based on the Bayesian Dirichlet
    log-equivalent uniform (BDeu) score.

    A BDeU score represents how well the existing architecture represents the existing data set. This
    score is, in addition, decomposable - meaning that differences in the local scores for each variable
    can be re-computed instead of having to compute the full score.

    This class is re-defined to speed up the implementation compared to pgmpy by using numpy vectorization.
    This class also assumes that no missing data will be provided.

    Parameters
    ----------
    data: DataFrame or np.ndarray
        The data over which the BDeU score will be computed.
        If given as a Pandas DataFrame, the dataset will be converted internally into a numpy array
        TODO Add from csv
    equivalent_sample_size: int, default=10
        Equivalent sample size for the BDeu score computation.
    nodes: list[str], optional
        List of ordered names of all variables. If not given, names will be extracted from the data.
        Must be provided if a numpy array is provided as data.
    """
    # ATTRIBUTES #

    # DATA HANDLING #
    # Data contained within the scorer
    data: np.ndarray
    # Index of all nodes within the numpy array
    # Shape: {variable_name: index}
    node_index: dict
    # Possible values for each node
    # Shape: {variable_name: [value_1, value_2...]}
    node_values: dict

    # SCORE COMPUTING #
    # Equivalent Sample Size
    esz: int

    # CONSTRUCTOR AND INITIALIZATION METHODS #
    def __init__(self, data, equivalent_sample_size=10, nodes=None):

        # Process the input data and, if necessary, convert it into a numpy array
        if isinstance(data, np.ndarray):
            self.data = data
            self._initialize_dictionaries(nodes)
        elif isinstance(data, DataFrame):
            self.data = data.to_numpy()
            self._initialize_dictionaries(data.columns.values.tolist())
        else:
            raise TypeError("Data must be provided as a Numpy array or a Pandas dataframe")

        # Store the equivalent sample size
        self.esz = equivalent_sample_size

    def _initialize_dictionaries(self, variable_names):
        """
        Given a list of the variable names, initializes the internal dictionaries
        for faster lookup.

        Parameters
        ----------
        variable_names: list[str]
            Ordered list of the variables contained within the data
        """

        # Initialize both dictionaries
        self.node_index = {}
        self.node_values = {}

        # Get the index and name of each variable
        for index, name in enumerate(variable_names):

            # Store the index
            self.node_index[name] = index

            # Get the unique values in the column
            self.node_values[name] = np.unique(self.data[:, index]).tolist()

    # SCORE FUNCTIONS #

    def local_score(self, variable, parents):
        """
        Computes the local BDeu score for a variable given a list of parents.

        Parameters
        ----------
        variable: str
            Child variable of which the BDeu score is being computed
        parents: list[str]
            List of parent variables that influence the child variable. If the variables
            have no parents, an empty list must be passed instead.

        Returns
        -------
        float
            BDeu score
        """

        # PRE - PROCESSING

        # Get the variable states and number of possible values
        variable_states = self.node_values[variable]
        variable_length = len(variable_states)

        # Generate a list with all possible parent values (per variable)
        parent_states = [self.node_values[parent] for parent in parents]
        parent_length = sum([len(parent) for parent in parent_states])

        # If no parents are specified, parent length must be 1
        if parent_length == 0:
            parent_length = 1

        # Number
