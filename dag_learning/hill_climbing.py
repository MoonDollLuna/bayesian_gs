# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import math

from dag_learning import BaseAlgorithm
from dag_architectures import ExtendedDAG

from itertools import permutations
from time import time
from tqdm import tqdm

import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BDeuScore
from pandas import DataFrame


def find_legal_hillclimbing_operations(dag):
    """
    Given a DAG and a set of variables, find all legal operations, creating three sets containing:
        - All possible edges to add.
        - All possible edges to remove.
        - All possible edges to invert.

    All of these operations are included into a single set, where each element has shape:

        (operation [add, remove, invert], (origin node, destination node))

    This takes care of avoiding possible cycles and trying illegal operations during the hill climbing process.

    Parameters
    ----------
    dag: ExtendedDAG
        DAG over which the operations are tried

    Returns
    -------
    set
    """

    # Get the list of nodes from the DAG
    nodes = list(dag.nodes())

    # EDGE ADDITIONS #

    # Generate the initial set of possible additions (all possible permutations of nodes)
    add_edges = set([("add", permutation) for permutation in permutations(nodes, 2)])

    # Remove invalid edge additions
    # Remove existing edges
    add_edges = add_edges - set([("add", edge) for edge in dag.edges()])
    # Remove inverted edges that already exist
    add_edges = add_edges - set([("add", (Y, X)) for (X, Y) in dag.edges()])
    # Remove edges that can lead to a cycle
    add_edges = add_edges - set([("add", (X, Y)) for (X, Y) in add_edges if nx.has_path(dag, Y, X)])

    # EDGE REMOVALS #

    # Generate the initial set of possible removals (only the existing edges)
    remove_edges = set([("remove", edge) for edge in dag.edges()])

    # EDGE INVERSIONS

    # Generate the initial set of possible removals (only the existing edges)
    invert_edges = set([("invert", edge) for edge in dag.edges()])

    # Remove the edges that, when inverted, would lead to a cycle
    invert_edges = invert_edges - set([("invert", (X, Y)) for (X, Y) in invert_edges if not any(map(lambda path: len(path) > 2, nx.all_simple_paths(dag, X, Y)))])

    # Join all sets into a single set
    return add_edges | remove_edges | invert_edges


class HillClimbing(BaseAlgorithm):
    """
    `HillClimbing` implements a simple Greedy Search approach to Bayesian Network structure building.

    The algorithm works in a loop by trying all possible actions over the existing nodes (either adding
    a new edge to the DAG or removing or inverting an already existing one).

    These actions are evaluated by using a total BDeU score for the Bayesian Network based on the
    data provided, choosing the action that provides the biggest increase in BDeU score.

    This loop is continued until no action improves the BDeU score, at which point a fully constructed
    Bayesian Network based on the existing nodes and data is provided.

    This algorithm serves as a baseline, to which all other algorithms implemented will be compared to.

    Parameters
    ----------
    bayesian_network: BayesianNetwork
        Original bayesian network, used to measure the structure quality
    nodes: list
        List of nodes contained within the data, used to generate the DAG
    data: DataFrame
        Dataframe representing the data to be used when building the DAG
    """

    def __init__(self, bayesian_network, nodes, data):
        """
        Prepares all necessary data and structures for Greedy Search.

        `estimate_dag` generates a DAG optimizing the BDeU score for the specified data.

        Parameters
        ----------
        bayesian_network: BayesianNetwork
            Original bayesian network, used to measure the structure quality
        nodes: list
            List of nodes contained within the data, used to generate the DAG
        data: DataFrame
            Dataframe representing the data to be used when building the DAG
        """

        # Call the super constructor
        super().__init__(bayesian_network, nodes, data)

    def estimate_dag(self, starting_dag=None, epsilon=0.0001, max_iterations=1e6,
                     wipe_cache=False, logged=True, silent=True):
        """
        TODO FINISH
        Performs Hill Climbing to find a local best DAG based on BDeU.

        Note that the found DAG may not be optimal, but good enough.

        Parameters
        ----------
        starting_dag: ExtendedDAG, optional
            Starting DAG. If not specified, an empty DAG is used.
        epsilon: float
            BDeU threshold. If the BDeU does not improve above the specified threshold,
            the algorithm stops
        max_iterations: int
            Maximum number of iterations to perform.
        wipe_cache: bool
            Whether the BDeU cache should be wiped or not
        logged: bool
            Whether the log file is written to or not
        silent: bool
            Whether warnings and loading screens should be printed on the screen or ignored.
            A log will be written regardless

        Returns
        -------
        ExtendedDAG
        """

        # LOCAL VARIABLE DECLARATION #

        # Log handling variables #
        # Iterations performed
        iterations: int = 0

        # Total operations checked and operations that needed new BDeU calculations
        total_operations: int = 0
        computed_operations: int = 0

        # Initial time and current time taken by the algorithm
        initial_time: float = time()
        time_taken: float = 0.0

        # BDeU metrics #
        # Best BDeu score
        best_bdeu: float = 0.0

        # Delta BDeU (change per iteration)
        delta_bdeu: float = math.inf

        # Metrics used to evaluate the resulting DAG #

        # Structural moral hamming distance (SMHD)
        smhd: int = 0
        # Average markov mantle
        average_markov: float = 0
        # Difference between the original and the resulting Markov mantle
        average_markov_difference: float = 0

        # PARAMETER INITIALIZATION #

        # Store the DAG and, if necessary, create an empty one with the existing nodes
        if starting_dag:
            dag = starting_dag
        else:
            dag = ExtendedDAG(self.nodes)

        # If necessary, wipe out the BDeU cache
        if wipe_cache:
            self.bdeu_cache.wipe_cache()

        # Prepare the BDeU Score estimator
        bdeu_scorer = BDeuScore(self.data)

        # MAIN LOOP #
        # TODO ADD LOG

        # Run the loop until:
        #   - The BDeU score improvement is not above the tolerance threshold
        #   - The maximum number of iterations is reached
        while iterations < max_iterations and delta_bdeu > epsilon:

            # Update the iterations
            iterations += 1

            # Reset the delta and specify the currently taken action
            delta_bdeu = 0
            current_best_bdeu = best_bdeu
            action_taken = None

            # Compute all possible actions for the current DAG
            actions = find_legal_hillclimbing_operations(dag)

            # Loop through all actions (using TQDM)
            for action, (X, Y) in tqdm(actions, disable=not silent):

                # Depending on the action:
                # Addition
                if action == "add":

                    # Compute the hypothetical parents list
                    parents_list = dag.get_parents(Y) + [X]

                    # Check if the BDeU already exists
                    if self.bdeu_cache.has_bdeu(Y, parents_list):
                        local_bdeu = self.bdeu_cache.get_bdeu_score(Y, parents_list)
                    else:
                        # If not, compute the new BDeU score and store it
                        local_bdeu = bdeu_scorer.local_score(Y, parents_list)
                        self.bdeu_cache.insert_bdeu_score(Y, parents_list, local_bdeu)

                    # Compute the new current BDeU
                    current_bdeu = best_bdeu + local_bdeu

                    # If the action improves the BDeU, store it
                    if current_bdeu > current_best_bdeu:
                        current_best_bdeu = current_bdeu
                        action_taken = (action, (X, Y))

