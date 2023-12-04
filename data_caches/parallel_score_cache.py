# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS
from typing import Iterable


def _convert_to_key(node, parents):
    """
    Converts the node and list of parents into an immutable tuple that can be used as a key for the dictionary.

    Parameters
    ----------
    node: str
        Node of which the local score is being considered
    parents: Iterable[str]
        Iterable of parents of the node. If the node has no parents, the iterable must be empty
    """

    return node, tuple(parents)


class ParallelScoreCache:
    """
    Local score cache implementation designed for efficient parallelized in-memory usage versus in-file
    memoizing (such as joblib's implementation).

    The class works by storing the expected local score for a pair of node and node parents.

    The cache keeps track of the "delta" changes of the cache, allowing for quicker sharing of exclusively
    new dictionary values to other processes - quicker than sharing the full cache.

    This class is used instead of Python's built in LRU caches since Python caches are designed to work
    per-interpreter, not allowing the sharing of recorded calls.

    Parameters
    ----------
    initial_values: dict, optional
        If specified, the cache is created with some initialized key-value pairs
    """

    # ATTRIBUTES #

    # Inner dictionary, storing the local scores for each call inputs
    _score_dict: dict
    # Delta dictionary, storing the newest values until flushed
    _delta_dict: dict

    # CONSTRUCTOR #
    def __init__(self, initial_values=None):

        # Initialize the dictionary and, if needed, extend it with the specified values
        self._score_dict = {}
        if initial_values:
            self._score_dict |= initial_values

        self._delta_dict = {}

    # LOOKUP METHODS
    def get_score(self, node, parents):
        """
        Returns the local score for a node and its parents, if it is contained. Otherwise, returns None.

        Parameters
        ----------
        node: str
            Node of which the local score is being considered
        parents: Iterable[str]
            Iterable of parents of the node. If the node has no parents, the iterable must be empty

        Returns
        -------
        float | None
        """

        return self._score_dict.get(_convert_to_key(node, parents))

    # ADDITION METHODS
    def add_score(self, node, parents, score):
        """
        Adds (or updates) a single local score to both the local and delta cache

        Parameters
        ----------
        node: str
            Node of which the local score is being considered
        parents: Iterable[str]
            Iterable of parents of the node. If the node has no parents, the iterable must be empty
        score: float
            Local score for the pair of node and parents
        """

        self._score_dict[_convert_to_key(node, parents)] = score
        self._delta_dict[_convert_to_key(node, parents)] = score

    def add_dictionary(self, dictionary):
        """
        Adds a dictionary of pre-existing values

        Parameters
        ----------
        dictionary: dict
            Dictionary containing pairs of (node, parents) and scores
        """

        self._score_dict |= dictionary

    def aggregate_dictionaries(self, *dictionaries):
        """
        Aggregates an indeterminate number of pre-existing dictionaries of values

        Parameters
        ----------
        dictionaries: Iterable[dict]
            Any number of dictionaries containing pairs of (node, parents) and scores
        """

        for dictionary in dictionaries:
            self._score_dict |= dictionary

    # DELTA MANAGEMENT METHODS

    def get_delta(self):
        """
        Returns the cache delta
        """

        return self._delta_dict

    def clear_delta(self):
        """
        Empties the current delta values
        """

        self._delta_dict = {}
