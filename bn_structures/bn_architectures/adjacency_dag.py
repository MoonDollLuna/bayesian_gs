# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

from bn_structures import BaseDAG
import numpy as np

from typing import List


class AdjacencyDAG(BaseDAG):
    """
    `AdjacencyDAG` represents a simple implementation of a Bayesian Network structure (more especifically, the
    DAG or Directed Acyclic Graph) utilizing an adjacency matrix.

    This class contains
        - An ordered list of variables (as a list of variable names)
        - An adjacency matrix representing edges between variables, where (i, j) = 1 represents
          a directed edge from variables I to J
    """

    ##############
    # ATTRIBUTES #
    ##############

    # List of variables
    variables: List[str]
    # Adjacency matrix
    adjacency_matrix: np.ndarray

    ###############
    # CONSTRUCTOR #
    ###############

    def __init__(self, variables):
        """
        Creates an empty DAG for the specified Bayesian Network variables.

        Parameters
        ----------
        variables : List[str]
            A list of ordered variables contained within the Bayesian Network
        """

        # Store the variables
        self.variables = variables

        # Create a numpy array to store the edges
        self.adjacency_matrix = np.identity(len(self.variables))

    def get_edges(self):
        """
        Gets the edges contained within the Directed Acyclic Graph (DAG)

        The edges are represented as an adjacency matrix.

        Returns
        -------
        np.ndarray
        """

        return self.adjacency_matrix

    #####################
    # EDGE MODIFICATION #
    #####################

    def add_edge(self, node_1_id, node_2_id):
        """
        Creates an edge in the Bayesian Network architecture directed from Node 1 (node_1_id) to Node 2 (node_2_id).
        The node IDs are defined by their order in the internal node list

        If the edge was added successfully, returns True. Returns False otherwise

        Parameters
        ----------
        node_1_id : int
            ID of the first node / variable
        node_2_id : int
            ID of the second node / variable

        Return
        ------
        bool
        """

        # Ensure that the edges are contained within the actual Bayesian Network
        if 0 <= node_1_id < len(self.variables) and 0 <= node_2_id < len(self.variables):

            # Ensure that the edge can be created (it does not exist already AND the inverse edge
            # does not exist already)
            if self.adjacency_matrix[node_1_id, node_2_id] != 1 and self.adjacency_matrix[node_2_id, node_1_id] != 1:

                # Create the edge
                self.adjacency_matrix[node_1_id, node_2_id] = 1
                return True

        return False

    def remove_edge(self, node_1_id, node_2_id):
        """
        Removes an edge in the Bayesian Network architecture directed from Node 1 (node_1_id) to Node 2 (node_2_id),
        if it exists. The node IDs are defined by their order in the internal node list

        This method cannot remove the link between a node and itself

        If the edge was added successfully, returns True. Returns False otherwise

        Parameters
        ----------
        node_1_id : int
            ID of the first node / variable
        node_2_id : int
            ID of the second node / variable

        Return
        ------
        bool
        """

        # Ensure that the edges are contained within the actual Bayesian Network
        if 0 <= node_1_id < len(self.variables) and 0 <= node_2_id < len(self.variables):

            # Ensure that the edge does actually exist
            if self.adjacency_matrix[node_1_id, node_2_id] == 1:

                # Remove the edge
                self.adjacency_matrix[node_1_id, node_2_id] = 0
                return True

        return False

    def invert_edge(self, node_1_id, node_2_id):
        """
        Inverts an edge in the Bayesian Network architecture between Node 1 (node_1_id) and Node 2 (node_2_id),
        if it exists. The node IDs are defined by their order in the internal node list

        Parameters
        ----------
        node_1_id : int
            ID of the first node / variable
        node_2_id : int
            ID of the second node / variable

        Return
        ------
        bool
        """

        # Ensure that the edges are contained within the actual Bayesian Network
        if 0 <= node_1_id < len(self.variables) and 0 <= node_2_id < len(self.variables):

            # Ensure that the edge does actually exist
            if self.adjacency_matrix[node_1_id, node_2_id] == 1:

                # Invert the edge
                self.adjacency_matrix[node_1_id, node_2_id] = 0
                self.adjacency_matrix[node_2_id, node_1_id] = 1
                return True

        return False
