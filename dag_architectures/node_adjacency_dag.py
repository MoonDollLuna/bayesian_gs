# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from typing import List
from dag_architectures import ExtendedDAG
from pgmpy.models.BayesianNetwork import BayesianNetwork

import numpy as np
import networkx as nx
import torch


class NodeAdjacencyDAG(ExtendedDAG):
    """
    `NodeAdjacencyDAG` extends the pgmpy implementation of a Directed Acyclic Graph (DAG) to include an
    node adjacency list representation, to be used as Tensors for neural networks.

    In this case, we understand a "Node Adjacency List" as a list of arrays, where each array i represents
    the connections of all other variables to the variable i, where the value in each position j means:
        - -1: The variable itself
        - 0 : Said variable j is not a parent of i
        - 1 : Said variable j is a parent of i

    This representation simplifies the process of computing the estimated BDeU score using a neural network.
    The representation is built during the node and edge manipulation process.

    Parameters
    ----------
    variables: list of str, optional
        List of nodes to initialize the DAG with

    Examples
    --------
    0: [-1, 0, 1, 0, 0]

    Represents that variable 0 (in the list of ordered variables) has variable 2 as a parent, and no other
    variables as parents
    """

    # ATTRIBUTES #

    # List of arrays representing the parents of each variable / node
    # These lists are updated in size every time a node (or a list of nodes) are added or removed from the DAG
    _node_adjacency_list: List[np.ndarray]

    # CONSTRUCTOR #

    def __init__(self, variables=None):
        """
        Creates a DAG with additional node adjacency lists to represent the edges between variables

        If no variables are specified, an empty DAG is constructed instead
        """

        # Initialize the super constructor (with the specified variables)
        super().__init__(variables)

        # If the DAG was built with variables, built the node adjacency list
        if variables:
            self._update_node_adjacency_list()

    # NODE ADJACENCY LIST METHODS #

    def _update_node_adjacency_list(self, removed=None):
        """
        Updates the node adjacency list representing the edges existing between the nodes in the DAG

        Parameters
        ----------
        removed: list of int, optional
            If specified, the following nodes will be removed from the node adjacency list instead
        """

        # Check if the nodes list needs to be built for the first time
        if self._node_adjacency_list is None:

            # Create the initial list
            self._node_adjacency_list = []

            # Add each node to the list
            for node_id in range(len(self)):

                # Create the list and specify the node itself
                new_node = np.zeros(len(self))
                new_node[node_id] = -1

                self._node_adjacency_list.append(new_node)

        # Check if nodes have been removed from the matrix
        elif removed:

            # Remove the nodes from the node adjacency list
            # (nodes are removed first to avoid extra work)
            for node_id in removed:
                del self._node_adjacency_list[node_id]

            # For each remaining node, remove the appropriate indices
            for adjacency_list in self._node_adjacency_list:
                for node_id in removed:
                    del adjacency_list[node_id]

        # Otherwise, add additional nodes as required
        else:

            # Find the size difference between the original list and the new list
            current_size = len(self._node_adjacency_list)
            expected_size = len(self)

            size_to_increase = expected_size - current_size

            # FAILSAFE: Operate only if the size has increased
            if size_to_increase > 0:

                # Add the node to the already existing adjacency lists
                for index in range(len(self._node_adjacency_list)):
                    self._node_adjacency_list[index] = np.append(self._node_adjacency_list[index], [0] * size_to_increase)

                # Add the new nodes
                for node_index in range(size_to_increase):

                    # Create the new adjacency list and specify the node itself
                    new_node = np.zeros(expected_size)
                    new_node[-(size_to_increase - node_index)] = -1

    def get_node_adjacency_list(self, node):
        """
        Returns the node adjacency list for a specific node

        Parameters
        ----------
        node: str or int
            The node to get the node adjacency list from

        Returns
        -------
        np.ndarray

        Raises
        ------
        NetworkXError
           If n is not in the graph.
        TypeError
            If the node is neither an int or a string
        """

        # Ensure that the input is either an int or a string
        if not isinstance(node, (int, str)):
            raise TypeError("Nodes must be either strings or integers")

        # If node is a string name, get the appropriate node index
        # Otherwise, raise a NetworkXError
        try:
            if isinstance(node, str):
                node = self.node_to_index[node]
        except KeyError:
            raise nx.NetworkXError("There is no node with name {} in the graph".format(node))

        # Return the node adjacency list
        return self._node_adjacency_list[node]

    def get_node_adjacency_list_tensor(self, node):
        """
        Returns the node adjacency list for a specific node as a tensor prepared for use as a neural network input

        Parameters
        ----------
        node: str or int
            The node to get the node adjacency list from

        Returns
        -------
        torch.Tensor

        Raises
        ------
        NetworkXError
           If n is not in the graph.
        TypeError
            If the node is neither an int or a string
        """

        # Ensure that the input is either an int or a string
        if not isinstance(node, (int, str)):
            raise TypeError("Nodes must be either strings or integers")

        # If node is a string name, get the appropriate node index
        # Otherwise, raise a NetworkXError
        try:
            if isinstance(node, str):
                node = self.node_to_index[node]
        except KeyError:
            raise nx.NetworkXError("There is no node with name {} in the graph".format(node))

        # Return the node adjacency list
        return torch.from_numpy(self._node_adjacency_list[node])

    # NODE MANIPULATION #

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

        # Update the node adjacencies accordingly
        self._update_node_adjacency_list()

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

        # Update the node adjacencies accordingly
        self._update_node_adjacency_list()

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

        # Update the node adjacencies accordingly
        self._update_node_adjacency_list(removed=[node_id])

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

        # Update the node adjacencies accordingly
        self._update_node_adjacency_list(removed=nodes_id)

    # EDGE MANIPULATION #

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v, and represent said edge within the node adjacency list.

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

        # Update the node adjacencies (in case a new node has been added)
        self._update_node_adjacency_list()

        # Ensure that the node names are converted to ints
        u, v = self.convert_nodes_to_indices(u, v)

        # Add the edge to the node adjacency list
        self._node_adjacency_list[u][v] = 1

    def add_edges_from(self, ebunch, weights=None):
        """
        Add all the edges in ebunch to the DAG and the node adjacency list

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
        self._update_node_adjacency_list()

        # Add each edge manually to the adjacency matrix
        for u, v in ebunch:
            u, v = self.convert_nodes_to_indices(u, v)
            self._node_adjacency_list[u][v] = 1

    def remove_edge(self, u, v):
        """
        Removes an edge between u and v, and removes the edge within the node adjacency list

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
        self._node_adjacency_list[u][v] = 0

    def invert_edge(self, u, v, weight=None):
        """
        Inverts the edge between u and v, replacing it with an edge between v and u.
        Also reflects the change within the node adjacency list

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
        self._node_adjacency_list[u][v] = 0
        self._node_adjacency_list[v][u] = 1

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

        dag = NodeAdjacencyDAG(list(bayesian_network.nodes))
        dag.add_edges_from(list(bayesian_network.edges))

        return dag