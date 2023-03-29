# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from typing import List
from extended_dag import ExtendedDAG

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
    # TODO - Add node, add nodes from, remove node, remove nodes from
    # EDGE MANIPULATION #
    # TODO - Add edges from
    # TODO - Fix methods to remove initialized

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v, and represent said edge within the node adjacency list.

        The nodes u and v will be automatically added if they are not already in the graph.

        This method can only be called if initialize_node_adjacency_list was called previously.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge
        """

        # Ensure that the adjacency matrix is already initialized
        if self.initialized:

            # Add the edge using the original method
            super().add_edge(u, v, weight)

            # Add the edge to the adjacency matrix
            self._node_adjacency_list[self.variable_index_dict[v]][self.variable_index_dict[u]] = 1

        else:
            raise ValueError

    def remove_edge(self, u, v):
        """
        Removes an edge between u and v, and removes the edge within the adjacency matrix.

        The nodes u and v will remain in the graph, even if they no longer have any edges.

        This method can only be called if initialize_node_adjacency_list was called previously.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.
        """

        # Ensure that the adjacency matrix is already initialized
        if self.initialized:

            # Remove the edge using the original method
            super().add_edge(u, v)

            # Remove the edge from the adjacency matrix
            self._node_adjacency_list[self.variable_index_dict[v]][self.variable_index_dict[u]] = 0

        else:
            raise ValueError

    def invert_edge(self, u, v, weight=None):
        """
        Inverts the edge between u and v, replacing it with an edge between v and u.
        Also reflects the change within the adjacency matrix.

        This method can only be called if initialize_node_adjacency_list was called previously.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge
        """

        # Ensure that the adjacency matrix is already initialized
        if self.initialized:

            # Invert the edge using the original method
            super().invert_edge(u, v, weight)

            # Invert the edge in the adjacency matrix
            self._node_adjacency_list[self.variable_index_dict[v]][self.variable_index_dict[u]] = 0
            self._node_adjacency_list[self.variable_index_dict[u]][self.variable_index_dict[v]] = 1

        else:
            raise ValueError