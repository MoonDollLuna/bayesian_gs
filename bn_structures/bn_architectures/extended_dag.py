# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import networkx as nx
import numpy as np
from pgmpy.base import DAG


class ExtendedDAG(DAG):
    """
    Extension of the DAG (Directed Acyclical Graph) class provided by pgmpy, adding additional functionality
    and hooks to be used by the proposed solutions within this project.

    Proposed DAG implementations must extend this class and implement all methods.
    """

    def remove_edge(self, u, v):
        """
        Removes an edge between u and v.

        The nodes u and v will remain in the graph, even if they no longer have any

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
