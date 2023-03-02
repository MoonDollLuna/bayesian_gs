# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import networkx as nx
from pgmpy.base import DAG


class ExtendedDAG(DAG):
    """
    Extension of the DAG (Directed Acyclical Graph) class provided by pgmpy, adding additional functionality
    and hooks to be used by the proposed solutions within this project.

    This extension includes:
        - The ability to directly add a series of variables to the DAG from construction.
        - The ability to use variable indexes instead of variable names for edge operations.
        - Removal and Inversion operations for edges.
        - Methods to return a list of existing edges (as indexes or variable names).

    Proposed DAG implementations must extend this class and implement all methods.

    Parameters
    ----------
    variables: list of str, optional
        List of nodes to initialize the DAG with
    """

    # ATTRIBUTES #

    # TODO Add edges to this list and update AdjacencyDAG / NodeAdjacencyDAG to not include duplicated methods
    # List of existing edges within the DAG, as (start, end) tuples
    _existing_edges: list

    # Dictionary containing the variable name -> variable index lookup
    _node_to_index: dict

    # Dictionary containing the variable index -> variable name lookup
    _index_to_node: dict

    # CONSTRUCTOR #

    def __init__(self, variables=None):
        # Construct the standard DAG
        super().__init__()

        # Initialize both dictionaries
        # These dictionaries are used for faster lookups
        self._node_to_index = {}
        self._index_to_node = {}

        # If specified, add the variables to the DAG
        self.add_nodes_from(variables)

    # NODE MANIPULATION #

    def add_node(self, node, weight=None, latent=False):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        weight: int, float
            The weight of the node.

        latent: boolean (default: False)
            Specifies whether the variable is latent or not.
        """

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

        # Adds the node using the appropriate method
        super().add_nodes_from(nodes, weights, latent)

        # Add all nodes to the dictionary
        for node in nodes:
            self._add_node_to_dictionaries(node)

    def _add_node_to_dictionaries(self, node):
        """
        Adds the node to both dictionaries for faster lookup

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.
        """

        # Find the appropriate index for this node
        index = len(self._index_to_node) + 1

        # Add the node to both dictionaries
        self._node_to_index[node] = index
        self._index_to_node[index] = node

    # EDGE MANIPULATION #

    # TODO
    def add_edge(self, u, v, weight=None):
        pass

    def remove_edge(self, u, v):
        """
        Removes an edge between u and v.

        The nodes u and v will remain in the graph, even if they no longer have any edges

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.
        """

        super(nx.DiGraph, self).remove_edge(u, v)

    def invert_edge(self, u, v, weight=None):
        """
        Inverts the edge between u and v, replacing it with an edge between v and u.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge
        """

        # Removes the edge
        super(nx.DiGraph, self).remove_edge(u, v)
        # Adds the opposite edge back
        super().add_edge(v, u, weight=weight)
