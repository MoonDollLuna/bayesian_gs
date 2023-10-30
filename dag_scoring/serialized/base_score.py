# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS

from functools import lru_cache


class BaseScore:
    """
    Abstract base class for serialized DAG and Bayesian Network structure scoring.

    All scoring serialized methods implemented (BDeu, BIC...) must extend this class and implement
    the required methods.

    This class assumes that Python's LRU cache is used for caching calls to local_score.
    """

    @lru_cache(maxsize=None)
    def local_score(self, node, parents):
        """
        Given a variable and a set of parents, return the local score for said combination.
        """

        raise NotImplementedError

    def local_score_delta(self, node, original_parents, new_parents):
        """
        Given a variable and two sets of parents (an original set and a new set),
        returns the difference in local score between (node, new parents) and (node, original parents)
        """

        raise NotImplementedError

    def global_score(self, dag):
        """
        Given a whole DAG, find the total score (the local score for each variable and set of parents)
        """

        raise NotImplementedError



