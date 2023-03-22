# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import networkx as nx
from pgmpy.base import DAG

# TODO MANAGEMENT WHEN DICTIONARY NAMES FAIL
class ExtendedDAG(DAG):
    """
    Extension of the DAG (Directed Acyclical Graph) class provided by pgmpy, adding additional functionality
    and hooks to be used by the proposed solutions within this project.

    This extension includes:
        - The ability to directly add a series of variables to the DAG from construction.
        - The ability to use variable indexes instead of variable names for edge operations.
        - Removal and Inversion operations for edges.

    However, this extension expects node names to be strings, in order to allow
    for node name - node index conversion without ambiguity.

    Proposed DAG implementations must extend this class and implement all methods.

    Parameters
    ----------
    variables: list of str, optional
        List of nodes to initialize the DAG with
    """

    # ATTRIBUTES #

    # Dictionary containing the variable name -> variable index lookup
    node_to_index: dict

    # Dictionary containing the variable index -> variable name lookup
    index_to_node: dict

    # CONSTRUCTOR #

    def __init__(self, variables=None):
        # Construct the standard DAG
        super().__init__()

        # Initialize both dictionaries
        # These dictionaries are used for faster lookups
        self.node_to_index = {}
        self.index_to_node = {}

        # If specified, add the variables to the DAG
        self.add_nodes_from(variables)

    # DICTIONARY MANAGEMENT

    def _add_node_to_dictionaries(self, node):
        """
        Adds the node to both dictionaries for faster lookup

        Parameters
        ----------
        node: str
            The node to add to the graph.
        """

        # Find the appropriate index for this node
        index = len(self.index_to_node)

        # Add the node to both dictionaries
        self.node_to_index[node] = index
        self.index_to_node[index] = node

    def _rebuild_dictionaries(self):
        """
        Rebuilds both node_to_index and index_to_node dictionaries from scratch

        This method is intended to be used after removing a node from the DAG
        """

        # Reset both dictionaries
        self.node_to_index = {}
        self.index_to_node = {}

        # Assign indices and nodes to every node
        for index, node in enumerate(list(self)):
            self.node_to_index[node] = index
            self.index_to_node[index] = node

    def convert_nodes_to_indices(self, u, v):
        """
        Automatically transforms both u and v node strings into node indices

        Parameters
        ----------
        u, v : int or str
            Index or name of the nodes contained within the edge

        Returns
        -------
        tuple(int, int)
        """

        if isinstance(u, str):
            u = self.node_to_index[u]
        if isinstance(v, str):
            v = self.node_to_index[v]

        return u, v

    def convert_indices_to_nodes(self, u, v):
        """
        Automatically transforms both u and v node indices into node strings

        Parameters
        ----------
        u, v : int or str
            Index or name of the nodes contained within the edge

        Returns
        -------
        tuple(str, str)
        """

        if isinstance(u, int):
            u = self.index_to_node[u]
        if isinstance(v, int):
            v = self.index_to_node[v]

        return u, v

    # NODE MANIPULATION #

    def add_node(self, node, weight=None, latent=False):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str
            The node to add to the graph.

        weight: int, float
            The weight of the node.

        latent: boolean (default: False)
            Specifies whether the variable is latent or not.
        """

        # If the node is not a string (for example, a numerical name),
        # convert it to a string name
        node = str(node)

        # Adds the node using the appropriate method
        super().add_node(node, weight, latent)

        # Adds the node to the dictionary
        self._add_node_to_dictionaries(node)

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
        """

        # If any of the added nodes are not a string (for example, numerical names)
        # convert them to string names
        nodes = [str(node) for node in nodes]

        # Adds the node using the appropriate method
        super().add_nodes_from(nodes, weights, latent)

        # Add all nodes to the dictionary
        for node in nodes:
            self._add_node_to_dictionaries(node)

    def remove_node(self, n):
        """
        Removes the node n and all adjacent edges. In addition, update all internal dictionaries
        to avoid inconsistencies.

        Attempting to remove a non-existent node will raise an exception.

        Parameters
        ----------
        n : str or int
           A node in the graph

        Raises
        ------
        NetworkXError
           If n is not in the graph.
        """

        # If n is an index, check if it exists and get the appropriate node name
        # Otherwise, raise a NetworkXError
        try:
            if isinstance(n, int):
                n = self.index_to_node[n]
        except KeyError:
            raise nx.NetworkXError

        # Remove the node from the original directed graph
        super(nx.DiGraph, self).remove_node(n)

        # Rebuild the dictionaries to avoid inconsistencies
        self._rebuild_dictionaries()

    def remove_nodes_from(self, nodes):
        """
        Remove multiple nodes.

        If any node in nodes does not belong to the DAG, the removal will silently fail without error

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).  If a node
            in the container is not in the graph it is silently ignored.
        """

        # Remove the nodes from the original directed graph
        super(nx.DiGraph, self).remove_nodes_from(nodes)

        # Rebuild the dictionaries to avoid inconsistencies
        self._rebuild_dictionaries()

    # EDGE MANIPULATION #

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph (if specified as strings).

        Note that nodes may be

        Parameters
        ----------
        u, v : int, str
            Index or name of the nodes contained within the edge

        weight: int, float (default=None)
            The weight of the edge
        """

        # TODO ENSURE THAT IF A NEW NODE IS GIVEN AS AN EDGE, IT IS ALSO ADDED TO THE DICTIONARIES

        # If the edges are given as indices, transform them to their appropriate names
        u, v = self.convert_indices_to_nodes(u, v)

        # Add the edges to the graph
        super().add_edge(u, v, weight)

    def remove_edge(self, u, v):
        """
        Removes an edge between u and v.

        The nodes u and v will remain in the graph, even if they no longer have any edges

        Parameters
        ----------
        u, v : int, str
            Index or name of the nodes contained within the edge

        Raises
        ------
        NetworkXError
           If u or v are not in the graph.
        """

        # If the edges are given as indices, transform them to their appropriate names
        u, v = self.convert_indices_to_nodes(u, v)

        # Remove the edges from the graph
        super(nx.DiGraph, self).remove_edge(u, v)

    def invert_edge(self, u, v, weight=None):
        """
        Inverts the edge between u and v, replacing it with an edge between v and u.

        Parameters
        ----------
        u, v : int, str
            Index or name of the nodes contained within the edge

        weight: int, float (default=None)
            The weight of the edge

        Raises
        ------
        NetworkXError
           If u or v are not in the graph.
        """

        # If the edges are given as indices, transform them to their appropriate names
        u, v = self.convert_indices_to_nodes(u, v)

        # Removes the edge
        super(nx.DiGraph, self).remove_edge(u, v)
        # Adds the opposite edge back
        super().add_edge(v, u, weight=weight)
