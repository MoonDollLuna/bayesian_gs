# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from extended_dag import ExtendedDAG
import numpy as np
import torch


class AdjacencyDAG(ExtendedDAG):
    """
    `AdjacencyDAG` extends the pgmpy implementation of a Directed Acyclic Graph (DAG) to include an
    adjacency matrix implementation, to be used as Tensors in Neural Networks.

    Parameters
    ----------
    variables: list of str, optional
        List of nodes to initialize the DAG with
    """

    # ATTRIBUTES #
    
    # Adjacency matrix representing the DAG
    # If no variables are declared during DAG construction, a specific method must be called
    # to initialize the adjacency matrix
    adjacency_matrix: np.ndarray

    # Dictionary representing the index for each variable in the adjacency matrix
    # Created to speed-up the lookup process
    variable_index_dict: dict

    # Flag to indicate if the adjacency matrix has been already initialized
    adjacency_matrix_initialized: bool

    # CONSTRUCTOR #

    def __init__(self, variables=None):
        """
        Creates a DAG with an additional adjacency matrix to represent the edges between variables

        If no variables are specified, an empty DAG is constructed instead
        """

        # Initialize the super constructor
        super().__init__()

        # If there are variables, add them to the DAG and create the adjacency matrix
        # Otherwise, the adjacency matrix remains uninitialized
        if variables:
            self.add_nodes_from(nodes=variables)
            self.initialize_adjacency_matrix()

    # HELPER METHODS #

    def initialize_adjacency_matrix(self):
        """
        Updates the internal DAG adjacency matrix representation to include all added variables, and
        initializes a dictionary containing the index of each variable within the adjacency matrix

        This method MUST be called before edge modification, and can only be called once
        """

        # This method can only be called once
        if not self.adjacency_matrix_initialized:

            # Create the adjacency matrix
            self.adjacency_matrix = np.zeros((len(self), len(self)))

            # Create the lookup dictionary
            self.variable_index_dict = {}
            for index in range(len(self)):
                self.variable_index_dict[list(self)[index]] = index

            # Mark the adjacency matrix flag
            self.adjacency_matrix_initialized = True

        else:
            print("The adjacency matrix has already been initialized")

    # EDGE MANIPULATION #

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v, and represent said edge within the adjacency matrix.

        The nodes u and v will be automatically added if they are not already in the graph.

        This method can only be called if update_adjacency_matrix was called previously.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge
        """

        # Ensure that the adjacency matrix is already initialized
        if self.adjacency_matrix_initialized:

            # Add the edge using the original method
            super().add_edge(u, v, weight)

            # Add the edge to the adjacency matrix
            self.adjacency_matrix[self.variable_index_dict[u], self.variable_index_dict[v]] = 1

        else:
            raise ValueError

    def remove_edge(self, u, v):
        """
        Removes an edge between u and v, and removes the edge within the adjacency matrix.

        The nodes u and v will remain in the graph, even if they no longer have any edges.

        This method can only be called if update_adjacency_matrix was called previously.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.
        """

        # Ensure that the adjacency matrix is already initialized
        if self.adjacency_matrix_initialized:

            # Remove the edge using the original method
            super().add_edge(u, v)

            # Remove the edge from the adjacency matrix
            self.adjacency_matrix[self.variable_index_dict[u], self.variable_index_dict[v]] = 0

        else:
            raise ValueError

    def invert_edge(self, u, v, weight=None):
        """
        Inverts the edge between u and v, replacing it with an edge between v and u.
        Also reflects the change within the adjacency matrix.

        This method can only be called if update_adjacency_matrix was called previously.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge
        """

        # Ensure that the adjacency matrix is already initialized
        if self.adjacency_matrix_initialized:

            # Invert the edge using the original method
            super().invert_edge(u, v, weight)

            # Invert the edge in the adjacency matrix
            self.adjacency_matrix[self.variable_index_dict[u], self.variable_index_dict[v]] = 0
            self.adjacency_matrix[self.variable_index_dict[v], self.variable_index_dict[u]] = 1

        else:
            raise ValueError

    # NEURAL NETWORK UTILITIES

    def get_tensor(self):
        """
        Returns the adjacency matrix as a tensor prepared for use as a neural network input

        Returns
        -------
        np.ndarray
        """

        return torch.from_numpy(self.adjacency_matrix)

