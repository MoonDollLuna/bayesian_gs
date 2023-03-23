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

    The adjacency matrix is built every time a node (or a list of nodes) is added to the DAG.
    The existing edges are kept in the newly built DAG.

    If a node (or series of nodes) is removed from the DAG, the adjacency matrix is completely rebuilt
    from scratch (to avoid possible incoherencies)

    Parameters
    ----------
    variables: list of str or int, optional
        List of nodes to initialize the DAG with
    """

    # ATTRIBUTES #
    
    # Adjacency matrix representing the DAG
    # This matrix is updated every time a node (or list of nodes) is added or removed from the DAG
    _adjacency_matrix: np.ndarray

    # CONSTRUCTOR #

    def __init__(self, variables=None):
        """
        Creates a DAG with an additional adjacency matrix to represent the edges between variables

        If no variables are specified, an empty DAG is constructed instead
        """

        # Initialize the super constructor (with the specified variables)
        super().__init__(variables)

        # If the DAG was built with variables, build the adjacency matrix
        if variables:
            self._update_adjacency_matrix()

    # ADJACENCY MATRIX METHODS #

    def _update_adjacency_matrix(self, force_reset=False):
        """
        Updates (or, if necessary, builds from scratch) the adjacency matrix representing the
        edges existing between the nodes in the DAG

        Parameters
        ----------
        force_reset : bool, default = False
            If specified, the adjacency matrix is always built from scratch
        """

        # Check if the matrix needs to be re-built
        # (either first time building or deleting nodes)
        if self._adjacency_matrix is None or force_reset:

            # Create a new adjacency matrix from scratch
            self._adjacency_matrix = np.zeros((len(self), len(self)))

            # Add the already existing edges into the adjacency matrix
            self._update_adjacency_matrix_edges()

        # Otherwise, adds additional rows and columns to the
        # adjacency matrix
        else:

            # Find the difference between the current size and the expected size
            current_size = len(self._adjacency_matrix)
            expected_size = len(self)
            size_to_increase = expected_size - current_size

            # Add columns up to the expected size
            for _ in range(size_to_increase):
                self._adjacency_matrix = np.c_[self._adjacency_matrix, np.zeros(current_size)]

            # Add rows up to the expected size
            for _ in range(size_to_increase):
                self._adjacency_matrix = np.r_[self._adjacency_matrix, np.zeros(expected_size)]

    def _update_adjacency_matrix_edges(self):
        """
        Given an already existing and empty adjacency matrix, add all already existing edges
        back into the adjacency matrix

        This method is mostly useful when deleting nodes
        """

        # For each edge in the edges list, add it back to the adjacency matrix
        for u, v in list(self.edges()):

            # Convert the edge names to indices
            u = self.node_to_index[u]
            v = self.node_to_index[v]

            # Add the edge into the adjacency matrix
            self._adjacency_matrix[u, v] = 1

    def get_adjacency_matrix(self):
        """
        Returns the adjacency matrix.

        Returns
        -------
        np.ndarray
        """

        return self._adjacency_matrix

    def get_adjacency_matrix_tensor(self):
        """
        Returns the adjacency matrix as a tensor prepared for use as a neural network input

        Returns
        -------
        torch.Tensor
        """

        return torch.from_numpy(self._adjacency_matrix)

    # EDGE MANIPULATION #

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v, and represent said edge within the adjacency matrix.

        The nodes u and v will be automatically added if they are not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Add the edge using the original method
        super().add_edge(u, v, weight)

        # Ensure that the node names are converted to ints
        u, v = self.convert_nodes_to_indices(u, v)

        # Add the edge to the adjacency matrix
        self._adjacency_matrix[u, v] = 1

    def remove_edge(self, u, v):
        """
        Removes an edge between u and v, and removes the edge within the adjacency matrix.

        The nodes u and v will remain in the graph, even if they no longer have any edges.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Remove the edge using the original method
        super().add_edge(u, v)

        # If the nodes are given as names, transform them to ints
        u, v = self.convert_nodes_to_indices(u, v)

        # Add the edge to the adjacency matrix
        self._adjacency_matrix[u, v] = 0

    def invert_edge(self, u, v, weight=None):
        """
        Inverts the edge between u and v, replacing it with an edge between v and u.
        Also reflects the change within the adjacency matrix.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Invert the edge using the original method
        super().invert_edge(u, v, weight)

        # If the nodes are given as names, transform them to ints
        u, v = self.convert_nodes_to_indices(u, v)

        # Invert the edge in the adjacency matrix
        self._adjacency_matrix[u, v] = 0
        self._adjacency_matrix[v, u] = 1
