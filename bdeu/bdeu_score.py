# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from itertools import product

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

        # Generate all possible combinations of parent states
        parent_state_combinations = list(product(*parent_states))

        state_counts = self.get_state_counts(variable, variable_states, parents, parent_state_combinations)

    def get_state_counts(self, variable, variable_states, parents, parent_state_combinations):
        """
        For each combination of parent states, returns the count of each variable state
        for each combination of parent states.

        Parameters
        ----------
        variable : str
            Variable of which the states are counted
        variable_states : list[str]
            List of all states of the variable
        parents : list[str]
            List of parents of the variable
        parent_state_combinations : list[list[str]]
            List of all possible combination of parent variables' states

        Returns
        -------
        np.ndarray
            Array where each row represents a state of the variable, each column represents
            a combination of parent variable states and each cell represents the count of said
            variable state for the given parents
        """

        # Initialize the numpy array to be returned
        counts_array = np.zeros((len(variable_states), len(parent_state_combinations)))

        # Generate and apply all necessary masks
        masks = self._get_array_masks(parents, parent_state_combinations)
        for mask_index, mask in enumerate(masks):

            # Apply the mask to the data array to only keep the relevant columns
            masked_data = self.data[mask]

            # Count the instances of the child variable for each state
            variable_index = self.node_index[variable]
            masked_variable = masked_data[:, variable_index]
            states, state_counts = np.unique(masked_variable, return_counts=True)
            counts_dict = dict(zip(states, state_counts))

            # Store the counts for each variable state
            for state_index, state in enumerate(variable_states):

                # If the state has not appeared in the count, it is set to zero
                counts_array[mask_index, state_index] = counts_dict[state] if state in counts_dict else 0

        return counts_array

    def _get_array_masks(self, parents, parent_state_combinations):
        """
        Generates all array masks to apply over the data array

        Parameters
        ----------
        parents : list[str]
            List of parents of the variable
        parent_state_combinations : list[list[str]]
            List of all possible combination of parent variables' states

        Returns
        -------
        list[np.ndarray]
            List of all masks to apply to the data
        """

        # Create a list to store all possible masks
        masks = []

        # For each combination of parents, generate a mask
        for combination in parent_state_combinations:

            # Generate an initial, all true mask equal to the length of the data array
            mask = np.full(self.data.shape[1], True)

            # Apply the appropriate condition for all parent states
            for parent, parent_state in zip(parents, combination):

                parent_index = self.node_index[parent]
                mask = mask & (self.data[:, parent_index] == parent_state)

            # Store the mask
            masks.append(mask)

        return masks
