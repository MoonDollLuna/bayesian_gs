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

    # TODO
    # Dictionary containing the

    # CONSTRUCTOR #

    def __init__(self, variables=None):
        # Construct the standard DAG
        super().__init__()

        # If specified, add the variables to the DAG



    # NODE MANIPULATION #

    def add_node(self, node, weight=None, latent=False):
        pass

    def add_nodes_from(self, nodes, weights=None, latent=False):
        pass


    # EDGE MANIPULATION #

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
