# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

def nodes_to_key(node, parents):
    """
    Converts the node and its parents to a tuple representation used as a key for the inner dictionary

    Parameters
    ----------
    node: nodes
        Nodes can be any hashable Python object.
    parents: list[nodes]
        Nodes can be any hashable Python object.

    Returns
    -------
    vector
    """

    return node, tuple(parents)


class BDeUCache:
    """
    Cache storing the BDeU score of all families of node and parents.

    A cache is used to store local BDeU scores, in order to speed up the process
    by avoiding repeat computations
    """

    # ATTRIBUTES #

    # Inner dictionary of family -> local BDeU score
    # Families (keys) are represented as a tuple (node, (parents))
    _bdeu_scores: dict

    # CONSTRUCTOR #
    def __init__(self):

        # Initialize the dictionary
        self._bdeu_scores = {}

    # BDEU RETRIEVAL - INSERTION METHODS
    def has_bdeu(self, node, parents):
        """
        Checks if a local BDeU score has already been computed / is contained within the cache
        for the specified family (node and parents)

        Parameters
        ----------
        node : str or int
            Child of the family
        parents : list of str or list of int
            Parents of the node

        Returns
        -------
        bool
        """

        key = nodes_to_key(node, parents)
        return key in self._bdeu_scores

    def get_bdeu_score(self, node, parents):
        """
        Gets the local BDeU score of the already existing family within the cache

        Parameters
        ----------
        node : str or int
            Child of the family
        parents : list of str or list of int
            Parents of the node

        Returns
        -------
        float
        """

        key = nodes_to_key(node, parents)
        return self._bdeu_scores[key]

    def insert_bdeu_score(self, node, parents, score):
        """
        Inserts the local BDeU score for a given family within the cache

        Parameters
        ----------
        node : str or int
            Child of the family
        parents : list of str or list of int
            Parents of the node
        score : float
            Local BDeU score of the family
        """

        key = nodes_to_key(node, parents)
        self._bdeu_scores[key] = score

    # HELPER METHODS

    def wipe_cache(self):
        """
        Empties the BDeU cache contents.
        """

        self._bdeu_scores = {}
