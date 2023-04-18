# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
from dag_architectures import ExtendedDAG
from data_processing import BDeUCache
from utils import LogManager


class GreedySearch:
    """
    `GreedySearch` implements a simple Greedy Search approach to Bayesian Network structure building.

    The algorithm works in a loop by trying all possible actions over the existing nodes (either adding
    a new edge to the DAG or removing or inverting an already existing one).

    These actions are evaluated by using a total BDeU score for the Bayesian Network based on the
    data provided, choosing the action that provides the biggest increase in BDeU score.

    This loop is continued until no action improves the BDeU score, at which point a fully constructed
    Bayesian Network based on the existing nodes and data is provided.

    This algorithm serves as a baseline, to which all other algorithms implemented will be compared to.

    Parameters
    ----------
    epsilon: float
        The Greedy Search process stops once the difference in BDeU score does not improve above this threshold
    """

    # ATTRIBUTES #

    # Parameters #
    # Parameters to be used by the Greedy Search algorithm

    # Epsilon - the search stops once the difference in BDeU score does not improve above the epsilon threshold
    epsilon: float

    # Utilities #
    # This includes utilities like the log manager or the BDeU cache

    # BDeU cache
    bdeu_cache: BDeUCache

    # Log manager
    log_manager: LogManager

    # Data analysis management #
    # These attributes are stored to be shared with the Log Manager to print
    # and analyze the results of the algorithm

    # Total operations "tried"
    # This may include operations for which the BDeU score was previously known
    total_operations: int

    # Operations that needed to compute a new BDeU score
    computed_operations: int

    # Total time of operation for the algorithm
    time_taken: float

    def __init__(self, epsilon):
        """
        Initializes a GreedySearch instance and all necessary auxiliary managers.
        """

        # Stores the epsilon threshold
        self.epsilon = epsilon

        # Creates the BDeU cache
        self.bdeu_cache = BDeUCache()

        # Creates the log manager
        # TODO - Specify log manager path
        self.log_manager = LogManager()

    def build_DAG(self):
        pass
