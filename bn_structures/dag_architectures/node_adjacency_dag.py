# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from typing import List

from extended_dag import ExtendedDAG
import numpy as np
import torch


class NodeAdjacencyDAG(ExtendedDAG):
    """
    `NodeAdjacencyDAG` extends the pgmpy implementation of a Directed Acyclic Graph (DAG) to include an
    node adjacency list representation, to be used as Tensors for neural networks.

    In this case, we understand a "Node Adjacency List" as an array i representing the connections of
    all other variables to the variable i, where the value in each position j means:
        - -1: The variable itself
        - 0 : Said variable j is not a parent of i
        - 1 : Said variable j is a parent of i

    This representation simplifies the process of computing the BDeU score

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
    # If no variables are declared during DAG construction, a specific method must be called
    # to initialize the list
    node_adjacency_list: List[np.ndarray]

    # Dictionary representing the index for each variable in the adjacency matrix
    # Created to speed-up the lookup process
    variable_index_dict: dict

    # Flag to indicate if the list of arrays has been already initialized
    initialized: bool

    # CONSTRUCTOR #

    def __init__(self, variables=None):
        """
        Creates a DAG with additional node adjacency lists to represent the edges between variables

        If no variables are specified, an empty DAG is constructed instead
        """

        # Initialize the super constructor
        super().__init__()

        # If there are variables, add them to the DAG and create the adjacency matrix
        # Otherwise, the adjacency matrix remains uninitialized
        if variables:
            self.add_nodes_from(nodes=variables)
            self.initialize_node_adjacency_list()

    # HELPER METHODS #

    def initialize_node_adjacency_list(self):
        """
        Updates the internal DAG node adjacency list representation to include all added variables, and
        initializes a dictionary containing the index of each variable within the list

        This method MUST be called before edge modification, and can only be called once
        """

        # This method can only be called once
        if not self.initialized:

            # Create the initial node adjacency list
            self.node_adjacency_list = []

            # Create the lookup dictionary and the adjacency list for each variable
            self.variable_index_dict = {}
            for index in range(len(self)):

                # Dictionary index
                self.variable_index_dict[list(self)[index]] = index

                # Adjacencty list
                adjacency_list = np.zeros(len(self))
                adjacency_list[index] = -1
                self.node_adjacency_list.append(adjacency_list)

            # Mark the adjacency matrix flag
            self.initialized = True

        else:
            print("The adjacency matrix has already been initialized")

    # EDGE MANIPULATION #

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
            self.node_adjacency_list[self.variable_index_dict[v]][self.variable_index_dict[u]] = 1

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
            self.node_adjacency_list[self.variable_index_dict[v]][self.variable_index_dict[u]] = 0

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
            self.node_adjacency_list[self.variable_index_dict[v]][self.variable_index_dict[u]] = 0
            self.node_adjacency_list[self.variable_index_dict[u]][self.variable_index_dict[v]] = 1

        else:
            raise ValueError

    # NEURAL NETWORK UTILITIES

    def get_tensor(self, variable):
        """
        Returns the adjacency list of a specific variable as a tensor prepared for use as a neural network input

        Parameters
        ----------
        variable : str or int
            Variable to get the adjacency list of

        Returns
        -------
        torch.Tensor
        """

        # Get the appropriate list depending on whether a string or an int is provided
        if isinstance(variable, str):
            adjacency = self.node_adjacency_list[self.variable_index_dict[variable]]
        else:
            adjacency = self.node_adjacency_list[variable]

        return torch.from_numpy(adjacency)

