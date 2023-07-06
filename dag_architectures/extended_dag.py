# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import networkx as nx
from pgmpy.base import DAG
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pandas import DataFrame


class ExtendedDAG(DAG):
    """
    Extension of the DAG (Directed Acyclical Graph) class provided by pgmpy, adding additional functionality
    and hooks to be used by the proposed solutions within this project.

    This extension includes:
        - The ability to directly add a series of variables to the DAG from construction.
        - The ability to use variable indexes instead of variable names for edge operations.
        - Improved node removal to return node indices.
        - Removal and Inversion operations for edges.

    However, this extension expects node names to be strings, in order to allow
    for node name - node index conversion without ambiguity.

    Strict node naming and usage is enforced by only allowing strings and integers and inputs -
    other inputs when not appropriate will lead to TypeError exceptions.

    Proposed DAG implementations must extend this class and implement all methods.

    Parameters
    ----------
    variables: list of str or int, optional
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

        If a node does not exist, the method will return None instead of failing

        Parameters
        ----------
        u, v : int or str
            Index or name of the nodes contained within the edge

        Returns
        -------
        tuple(int or None, int or None)

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        NetworkXError
           If u or v are not in the graph.
        """

        # Ensure that the inputs are either ints or strings
        if not isinstance(u, (int, str)) or not isinstance(v, (int, str)):
            raise TypeError("Nodes must be either strings or integers")

        if isinstance(u, str):
            try:
                u = self.node_to_index[u]
            except KeyError:
                raise nx.NetworkXError("Node {} is not in the graph".format(u))
        if isinstance(v, str):
            try:
                v = self.node_to_index[v]
            except KeyError:
                raise nx.NetworkXError("Node {} is not in the graph".format(v))

        return u, v

    def convert_indices_to_nodes(self, u, v):
        """
        Automatically transforms both u and v node indices into node strings

        If a node does not exist, the method will return None instead of failing

        Parameters
        ----------
        u, v : int or str
            Index or name of the nodes contained within the edge

        Returns
        -------
        tuple(str or None, str or None)

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        NetworkXError
           If u or v are not in the graph.
        """

        # Ensure that the inputs are either ints or strings
        if not isinstance(u, (int, str)) or not isinstance(v, (int, str)):
            raise TypeError("Nodes must be either strings or integers")

        if isinstance(u, int):
            try:
                u = self.index_to_node[u]
            except KeyError:
                raise nx.NetworkXError("Node {} is not in the graph".format(u))
        if isinstance(v, int):
            try:
                v = self.index_to_node[v]
            except KeyError:
                raise nx.NetworkXError("Node {} is not in the graph".format(v))

        return u, v

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

        # Ensure that the inputs are either ints or strings
        if not isinstance(node, (int, str)):
            raise TypeError("Nodes must be either strings or integers")

        # If the node is not a string (for example, a numerical name),
        # force it into a string name
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

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Ignore None type objects
        if nodes is None:
            return

        # Check that all elements in the list are either integers or strings
        if not all(isinstance(node, (int, str)) for node in nodes):
            raise TypeError("Nodes must be either strings or integers")

        # If any of the added nodes are not a string (for example, numerical names)
        # force them into string names
        nodes = [str(node) for node in nodes]

        # Adds the node using the appropriate method
        super().add_nodes_from(nodes, weights, latent)

        # Add all nodes to the dictionary
        for node in nodes:
            self._add_node_to_dictionaries(node)

    def remove_node(self, n):
        """
        Removes the node n and all adjacent edges. In addition, update all internal dictionaries
        to avoid inconsistencies and return the original index of the removed node

        Attempting to remove a non-existent node will raise an exception to comply with the
        original methods.

        Parameters
        ----------
        n : str or int
           A node in the graph

        Returns
        -------
        int
            Original index of the removed node

        Raises
        ------
        NetworkXError
           If n is not in the graph.
        TypeError
            If the node is neither an int or a string
        """

        # Ensure that the inputs are either ints or strings
        if not isinstance(n, (int, str)):
            raise TypeError("Nodes must be either strings or integers")

        # If n is an index, check if it exists and get the appropriate node name
        # Otherwise, raise a NetworkXError
        try:
            if isinstance(n, int):
                n = self.index_to_node[n]
        except KeyError:
            raise nx.NetworkXError("There is no node with ID {} in the graph".format(n))

        # Get the index of the node
        node_id = self.node_to_index[n]

        # Remove the node from the original directed graph
        super(nx.DiGraph, self).remove_node(n)

        # Rebuild the dictionaries to avoid inconsistencies
        self._rebuild_dictionaries()

        # Return the original index of the node
        return node_id

    def remove_nodes_from(self, nodes):
        """
        Remove multiple nodes, and return their original indices.

        If any node in nodes does not belong to the DAG, the removal will silently fail without error
        to comply with the original methods

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).  If a node
            in the container is not in the graph it is silently ignored.

        Returns
        -------
        list[int]
            Original indices of the removed nodes

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Check that all elements in the list are either integers or strings
        if not all(isinstance(node, (int, str)) for node in nodes):
            raise TypeError("Nodes must be either strings or integers")

        # Convert all nodes into string names
        # Any conversion that fails will silently fail instead of raising an exception
        new_nodes = []
        for node in nodes:
            # Strings are not converted
            if isinstance(node, str):
                new_nodes.append(node)
            # ONLY INTS try to be converted, all other types will be ignored
            elif isinstance(node, int):
                try:
                    node = self.index_to_node[node]
                    new_nodes.append(node)
                except KeyError:
                    # Silently fail
                    pass

        # Get the index of all nodes to be removed
        nodes_id = [self.node_to_index[n] for n in new_nodes]

        # Remove the nodes from the original directed graph
        super(nx.DiGraph, self).remove_nodes_from(nodes)

        # Rebuild the dictionaries to avoid inconsistencies
        self._rebuild_dictionaries()

        # Return the original indices of the removed nodes
        return nodes_id

    # EDGE MANIPULATION #

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u, v : int, str
            Index or name of the nodes contained within the edge

        weight: int, float (default=None)
            The weight of the edge

        Raises
        ------
        TypeError
            If a node is neither an int or a string
        """

        # Ensure that the inputs are either ints or strings
        if not isinstance(u, (int, str)) or not isinstance(v, (int, str)):
            raise TypeError("Nodes must be either strings or integers")

        # If u or v are indices, check if they already exist
        # Otherwise, they will be considered unadded nodes -
        # and added to the DAG as strings

        if isinstance(u, int):
            try:
                u = self.index_to_node[u]
            except KeyError:
                u = str(u)

        if isinstance(v, int):
            try:
                v = self.index_to_node[v]
            except KeyError:
                v = str(v)

        # Add the edges to the graph
        super().add_edge(u, v, weight)

    def add_edges_from(self, ebunch, weights=None):
        """
        Add all the edges in ebunch.

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

        # Ensure that the inputs are either ints or strings
        if not all([isinstance(start, (int, str)) and isinstance(end, (int, str)) for start, end in ebunch]):
            raise TypeError("Nodes must be either strings or integers")

        # For all pair of edges, check if u and v exist
        # If not, assume that whichever node does not exist is a new node to be added
        # and transform it into a string

        new_ebunch = []

        for u, v in ebunch:
            if isinstance(u, int):
                try:
                    u = self.index_to_node[u]
                except KeyError:
                    u = str(u)

            if isinstance(v, int):
                try:
                    v = self.index_to_node[v]
                except KeyError:
                    v = str(v)

            new_ebunch.append((u, v))

        # Add the list of edges
        super().add_edges_from(new_ebunch, weights)

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
        TypeError
            If a node is neither an int or a string.
        """

        # If the edges are given as indices, transform them to their appropriate names
        u, v = self.convert_indices_to_nodes(u, v)

        # Remove the edges from the graph
        # Exceptions are checked by the superclass method
        super().remove_edge(u, v)

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
       TypeError
            If a node is neither an int or a string.
        """

        # If the edges are given as indices, transform them to their appropriate names
        u, v = self.convert_indices_to_nodes(u, v)

        # Removes the edge
        # Exceptions are checked by the superclass method
        super().remove_edge(u, v)
        # Adds the opposite edge back
        super().add_edge(v, u, weight=weight)

    # BAYESIAN NETWORK METHODS #

    def to_bayesian_network(self, dataset=None, state_names=None):
        """
        Converts the current Extended DAG into a Bayesian Network. If a dataset is specified, CPDs are estimated too
        (using the original Bayesian Networks states if specified).

        Parameters
        ----------
        dataset: DataFrame, optional
            Dataset to estimate CPDs from
        state_names: dict
            State names to use for the CPD computation

        Returns
        -------
        BayesianNetwork
        """

        # Create the Bayesian Network and add all the nodes and edges
        bn = BayesianNetwork()
        bn.add_nodes_from(list(self.nodes))
        bn.add_edges_from(list(self.edges))

        print(state_names["HISTORY"])
        print(dataset["HISTORY"].unique())

        # If a dataset is specified, estimate the CPDs
        if dataset is not None:
            bn.fit(dataset, state_names=state_names)

        return bn

    @staticmethod
    def from_bayesian_network(bayesian_network):
        """
        Converts a Bayesian Network into an Extended DAG (discarding the CPDs)

        Parameters
        ----------
        bayesian_network: BayesianNetwork

        Returns
        -------
        ExtendedDAG
        """

        dag = ExtendedDAG(list(bayesian_network.nodes))
        dag.add_edges_from(list(bayesian_network.edges))

        return dag

