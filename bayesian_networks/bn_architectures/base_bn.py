# BAYESIAN NETWORK GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

class BaseBN:
    """
    `BaseBN` represents an abstract definition of a Bayesian Network structure (more especifically, the
    DAG or Directed Acyclic Graph representing the conections between nodes in the Bayesian Network)

    Each Bayesian Network architecture implementation must contain:
        - A list of nodes (variables) contained within the architecture
        - A representation of the edges existing between the nodes

    This class contains methods that are required to be implemented by Bayesian Network structure representations
    defined within the following framework.
    """

    ##############
    # ATTRIBUTES #
    ##############

    # A list containing all the nodes (variables) within the Bayesian Network
    # Should be initialized by the constructor

    # TODO - Possibly another representation would be more efficient?
    node_list: list

    ####################
    # ABSTRACT METHODS #
    ####################

    # INFORMATION GATHERING #

    def get_nodes(self):
        """
        Returns the list of nodes contained within the Bayesian Network architecture

        Returns
        -------
        list
        """

        raise NotImplementedError

    def get_edges(self):
        """
        Returns the edges contained within the Bayesian Network architecture
        """

        raise NotImplementedError

    # EDGE MODIFICATION #

    def add_edge(self, node_1_id, node_2_id):
        """
        Creates an edge in the Bayesian Network architecture directed from Node 1 (node_1_id) to Node 2 (node_2_id).

        The node IDs are defined by their order in the internal node list

        Parameters
        ----------
        node_1_id : int
            ID of the first node / variable
        node_2_id : int
            ID of the second node / variable
        """

        raise NotImplementedError

    def remove_edge(self, node_1_id, node_2_id):
        """
        Removes an edge in the Bayesian Network architecture directed from Node 1 (node_1_id) to Node 2 (node_2_id),
        if it exists

        The node IDs are defined by their order in the internal node list

        Parameters
        ----------
        node_1_id : int
            ID of the first node / variable
        node_2_id : int
            ID of the second node / variable
        """

        raise NotImplementedError

    def invert_edge(self, node_1_id, node_2_id):
        """
        Inverts an edge in the Bayesian Network architecture between Node 1 (node_1_id) and Node 2 (node_2_id),
        if it exists

        The node IDs are defined by their order in the internal node list

        Parameters
        ----------
        node_1_id : int
            ID of the first node / variable
        node_2_id : int
            ID of the second node / variable
        """

        raise NotImplementedError

    # TODO ADD HASH METHODS FOR DICTIONARIES AND SETS
