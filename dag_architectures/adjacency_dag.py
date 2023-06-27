# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from dag_architectures import ExtendedDAG
from pgmpy.models.BayesianNetwork import BayesianNetwork

import numpy as np
import torch


class AdjacencyDAG(ExtendedDAG):
    """
    `AdjacencyDAG` extends the pgmpy implementation of a Directed Acyclic Graph (DAG) to include an
    adjacency matrix implementation, to be used as Tensors in Neural Networks.

    The adjacency matrix is built every time a node (or a list of nodes) is added to the DAG.
    The existing edges are kept in the newly built DAG.

    If a node (or series of nodes) is removed from the DAG, the adjacency matrix is completely rebuilt
    from scratch (to avoid possible inconsistencies)

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

    def _update_adjacency_matrix(self, removed=None):
        """
        Updates the adjacency matrix representing the edges existing between the nodes in the DAG

        Parameters
        ----------
        removed: list of int, optional
            If specified, the following nodes will be removed from the adjacency matrix instead
        """

        # Check if the matrix needs to be built for the first time
        if self._adjacency_matrix is None:

            # Create a new adjacency matrix from scratch
            self._adjacency_matrix = np.zeros((len(self), len(self)))

        # Check if nodes have been removed from the matrix
        elif removed:

            # Remove each node separately
            for node_id in removed:

                # Remove the node in both axis
                self._adjacency_matrix = np.delete(self._adjacency_matrix, node_id, axis=0)
                self._adjacency_matrix = np.delete(self._adjacency_matrix, node_id, axis=1)

        # Otherwise, adds additional rows and columns to the adjacency matrix
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

    # NODE MANIPULATION

    def add_node(self, node, weight=None, latent=False):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str or int
            The node to add to the graph.

        weight: int, float
            The weight of the node.

        latent: boolean (default: False)
            Specifies whether the variable is latent or not.

        Raises
        ------
        TypeError
            If the node is neither an int or a string
        """

        # Add the node to the Extended DAG
        super().add_node(node, weight, latent)

        # Update the adjacency matrix accordingly
        self._update_adjacency_matrix()

    def add_nodes_from(self, nodes, weights=None, latent=False):
        """
        Add multiple nodes to the Graph.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, or any hashable python
            object).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the variable at index i.

        latent: list, tuple (default=False)
            A container of boolean. The value at index i tells whether the
            node at index i is latent or not.

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Add the list of nodes to the Extended DAG
        super().add_nodes_from(nodes, weights, latent)

        # Update the adjacency matrix accordingly
        self._update_adjacency_matrix()

    def remove_node(self, n):
        """
        Removes the node n and all adjacent edges. In addition, update all internal dictionaries
        to avoid inconsistencies.

        Attempting to remove a non-existent node will raise an exception to comply with the
        original methods.

        Parameters
        ----------
        n : str or int
           A node in the graph

        Raises
        ------
        NetworkXError
           If n is not in the graph.
        TypeError
            If the node is neither an int or a string
        """

        # Remove the node from the Extended DAG
        node_id = super().remove_node(n)

        # Update the adjacency matrix accordingly
        self._update_adjacency_matrix(removed=[node_id])

    def remove_nodes_from(self, nodes):
        """
        Remove multiple nodes.

        If any node in nodes does not belong to the DAG, the removal will silently fail without error
        to comply with the original methods

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).  If a node
            in the container is not in the graph it is silently ignored.

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Remove the nodes from the Extended DAG
        nodes_id = super().remove_nodes_from(nodes)

        # Update the adjacency matrix accordingly
        self._update_adjacency_matrix(removed=nodes_id)

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

        # Update the adjacency matrix (in case a new node has been added)
        self._update_adjacency_matrix()

        # Ensure that the node names are converted to ints
        u, v = self.convert_nodes_to_indices(u, v)

        # Add the edge to the adjacency matrix
        self._adjacency_matrix[u, v] = 1

    def add_edges_from(self, ebunch, weights=None):
        """
        Add all the edges in ebunch to the DAG and the adjacency matrix.

        If nodes referred in the ebunch are not already present, they
        will be automatically added.

        Parameters
        ----------
        ebunch : list of tuple(str or int, str or int)
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the edge at index i.

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Add the edge using the original method
        super().add_edges_from(ebunch, weights)

        # Update the adjacency matrix (in case a new node has been added)
        self._update_adjacency_matrix()

        # Add each edge manually to the adjacency matrix
        for u, v in ebunch:
            u, v = self.convert_nodes_to_indices(u, v)
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
        super().remove_edge(u, v)

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

    # BAYESIAN NETWORK METHODS #

    @staticmethod
    def from_bayesian_network(bayesian_network):
        """
        Converts a Bayesian Network into an Extended DAG (discarding the CPDs)

        Parameters
        ----------
        bayesian_network: BayesianNetwork

        Returns
        -------
        AdjacencyDAG
        """

        dag = AdjacencyDAG(list(bayesian_network.nodes))
        dag.add_edges_from(list(bayesian_network.edges))

        return dag

