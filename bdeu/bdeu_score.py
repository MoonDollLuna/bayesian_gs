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

    # CONSTRUCTOR #
    def __init__(self, data, equivalent_sample_size=10, nodes=None):

        # Store the numpy array or, if necessary, convert it into a numpy array
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, DataFrame):
            self.data = data.to_numpy()
        else:
            raise TypeError("Data must be provided as a Numpy array or a Pandas dataframe")
