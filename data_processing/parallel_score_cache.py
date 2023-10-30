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

    return node, frozenset(parents)


class ParallelScoreCache:
    """
    Local score cache implementation designed for efficient parallelized in-memory usage versus in-file memoizing
    (such as joblib's implementation).

    The class implements a delta system that "stages" all updates before updating the main dictionary -
    allowing the specific delta of updates to be shared amongst child processes, instead of sharing the whole
    growing dictionary.

    In order to "apply" the delta changes, a specific method must be called to merge the results.

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

    # TODO - MODIFY TO UPDATE BOTH DICTIONARIES ALWAYS, AND TO EMPTY DELTA WHEN SPECIFIED
    # Delta dictionary, storing the temporary values
    _delta_dict: dict

    # CONSTRUCTOR #
    def __init__(self, initial_values=None):

        # Initialize the dictionary and, if needed, extend it
        self._score_dict = {}
        if initial_values:
            self._score_dict |= initial_values

        self._delta_dict = {}

    # LOOKUP METHODS
    # TODO - MOVE INTO MAGIC METHOD
    def get_score(self, node, parents):
        """
        Returns the local score for a node and its parents, if it is contained. Otherwise, returns None.

        This only checks scores in the dictionary, ignoring the unstaged (delta) scores.

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

        # Exceptions are used for speedup
        try:
            return self._score_dict[_convert_to_key(node, parents)]
        except KeyError:
            return None

    # DELTA ADDITION METHODS
    # TODO MOVE INTO MAGIC METHOD
    def add_score(self, node, parents, score):
        """
        Adds (or updates) a single score to the delta dictionary.

        Parameters
        ----------
        node: str
            Node of which the local score is being considered
        parents: Iterable[str]
            Iterable of parents of the node. If the node has no parents, the iterable must be empty
        score: float
            Local score for the pair of node and parents
        """

        self._delta_dict[_convert_to_key(node, parents)] = score

    def add_dictionary(self, dictionary):
        """
        Adds a dictionary of values to the delta dictionary.

        Parameters
        ----------
        dictionary: dict
            Dictionary containing pairs of (node, parents) and scores
        """

        self._delta_dict |= dictionary

    def aggregate_dictionaries(self, *dictionaries):
        """
        Aggregates an indeterminate number of dictionaries of values into the delta dictionary.

        Parameters
        ----------
        dictionaries: Iterable[dict]
            Any number of dictionaries containing pairs of (node, parents) and scores
        """

        for dictionary in dictionaries:
            self._delta_dict |= dictionary