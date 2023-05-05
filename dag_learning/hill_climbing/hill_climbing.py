# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import math

from dag_learning import BaseAlgorithm, \
    find_legal_hillclimbing_operations, compute_average_markov_mantle, compute_smhd
from dag_architectures import ExtendedDAG

from pgmpy.models import BayesianNetwork
from pgmpy.metrics import log_likelihood_score
from pgmpy.sampling import BayesianModelSampling

from time import time
from tqdm import tqdm
from pandas import DataFrame


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

    # CONSTRUCTOR #

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

    # MAIN METHODS #

    def estimate_dag(self, starting_dag=None, epsilon=0.0001, max_iterations=1e6, test_data_size=10000,
                     wipe_cache=False, verbose=0, logged=True):
        """
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
        test_data_size: int
            Amount of data to generate when testing the log likelihood
        wipe_cache: bool
            Whether the BDeU cache should be wiped or not
        verbose: int, default = 0
            Verbosity of the program, where:
                - 0: No information is printed
                - 1: Progress bar for each iteration is printed
                - 2: Action taken for each step and final results are printed
                - 3: Intermediate results for each step are printed
                - 4: Image of the final graph is printed
                - 5: DAG is directly printed
        logged: bool
            Whether the log file is written to or not

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
        bdeu_delta: float = math.inf

        # Metrics used to evaluate the resulting DAG #

        # Original-model free metrics

        # Log likelihood - How likely the resulting DAG is given a series of data
        log_likelihood: float = 0
        # Average markov mantle
        average_markov: float = 0

        # Metrics that require an existing Bayesian network

        # Structural moral hamming distance (SMHD)
        smhd: int = 0
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

        # MAIN LOOP #
        # TODO ADD LOG

        # Run the loop until:
        #   - The BDeU score improvement is not above the tolerance threshold
        #   - The maximum number of iterations is reached
        while iterations < max_iterations and bdeu_delta > epsilon:

            # Reset the delta and specify the currently taken action
            bdeu_delta = 0
            current_best_bdeu = best_bdeu
            action_taken = None

            # Compute all possible actions for the current DAG
            actions = find_legal_hillclimbing_operations(dag)

            # Loop through all actions (using TQDM) and if necessary print the iteration number
            if verbose >= 2:
                print("= ITERATION {} =".format(iterations + 1))
            for action, (X, Y) in tqdm(actions, disable=(verbose == 0)):

                # Depending on the action, compute the hypothetical parents list and child
                # Addition
                if action == "add":

                    # Compute the hypothetical parents lists
                    original_parents_list = dag.get_parents(Y)
                    new_parents_list = original_parents_list + [X]

                    # Compute the BDeU delta and the possible new BDeU score
                    current_bdeu_delta, local_total_operations, local_operations_computed = self._compute_bdeu_delta(Y, original_parents_list, new_parents_list)
                    current_bdeu = best_bdeu + current_bdeu_delta

                    # Update the metrics
                    total_operations += local_total_operations
                    computed_operations += local_operations_computed

                    # If the action improves the BDeU, store it and all required information
                    if current_bdeu > current_best_bdeu:
                        # Best BDeU and delta
                        current_best_bdeu = current_bdeu
                        bdeu_delta = current_bdeu_delta
                        action_taken = (action, (X, Y))

                # Removal
                elif action == "remove":

                    # Compute the hypothetical parents lists
                    original_parents_list = dag.get_parents(Y)
                    new_parents_list = original_parents_list[:].remove(X)

                    # Compute the BDeU delta and the possible new BDeU score
                    current_bdeu_delta, local_total_operations, local_operations_computed = self._compute_bdeu_delta(Y, original_parents_list, new_parents_list)
                    current_bdeu = best_bdeu + current_bdeu_delta

                    # Update the metrics
                    total_operations += local_total_operations
                    computed_operations += local_operations_computed

                    # If the action improves the BDeU, store it and all required information
                    if current_bdeu > current_best_bdeu:
                        # Best BDeU and delta
                        current_best_bdeu = current_bdeu
                        bdeu_delta = current_bdeu_delta
                        action_taken = (action, (X, Y))

                # Inversion
                elif action == "invert":

                    # Compute the hypothetical parents lists
                    # Note: in this case, two families are being changed

                    original_x_parents_list = dag.get_parents(X)
                    new_x_parents_list = original_x_parents_list + [Y]

                    original_y_parents_list = dag.get_parents(Y)
                    new_y_parents_list = original_y_parents_list[:].remove(X)

                    # Compute the BDeU deltas
                    current_x_bdeu_delta, local_total_x_operations, local_x_operations_computed = self._compute_bdeu_delta(
                        X, original_x_parents_list, new_x_parents_list)
                    current_y_bdeu_delta, local_total_y_operations, local_y_operations_computed = self._compute_bdeu_delta(
                        Y, original_y_parents_list, new_y_parents_list)

                    current_bdeu_delta = current_x_bdeu_delta + current_y_bdeu_delta
                    current_bdeu = best_bdeu + current_bdeu_delta

                    # Update the metrics
                    total_operations += local_total_x_operations + local_total_y_operations
                    computed_operations += local_x_operations_computed + local_y_operations_computed

                    # If the action improves the BDeU, store it and all required information
                    if current_bdeu > current_best_bdeu:
                        # Best BDeU and delta
                        current_best_bdeu = current_bdeu
                        bdeu_delta = current_bdeu_delta
                        action_taken = (action, (X, Y))

            # ALL OPERATIONS TRIED

            # Check if an operation was chosen
            if action_taken:

                # Apply the chosen operation
                operation, (X, Y) = action_taken

                if operation == "add":
                    dag.add_edge(X, Y)
                elif operation == "remove":
                    dag.remove_edge(X, Y)
                elif operation == "invert":
                    dag.invert_edge(X, Y)

                # Store the best bdeu
                best_bdeu = current_best_bdeu

            # Update the metrics

            iterations += 1
            time_taken = time() - initial_time

            # Print the required information according to the verbosity level
            if verbose >= 2:
                print("- Action taken: {}".format(action_taken))
            if verbose >= 3:
                print("* Current BDeU: {}".format(current_best_bdeu))
                print("* BDeU delta: {}".format(bdeu_delta))
                print("* Computed BDeU checks: {}".format(computed_operations))
                print("* Total BDeU checks: {}".format(total_operations))
                print("* Time taken: {}".format(time_taken))
                print("")
            if verbose >= 5:
                print("- Nodes: {}".format(list(dag.nodes)))
                print("- Edges: {}".format(list(dag.edges)))
                print("")

        # END OF THE LOOP - DAG FINALIZED

        # Compute the necessary metrics

        # Log likelihood
        # TODO IF NO ORIGINAL MODEL, USE TRAINING DATA SET
        # TODO EITHER CONVERT TO BN OR FIND WAY TO DO ON DAG

        # Generate the new data
        # test_data = BayesianModelSampling(self.bayesian_network).forward_sample(size=test_data_size)
        # log_likelihood = log_likelihood_score(dag, test_data)

        # Average Markov mantle
        average_markov = compute_average_markov_mantle(dag)

        # Average Markov mantle difference
        average_markov_difference = abs(compute_average_markov_mantle(self.bayesian_network) - average_markov)

        # Structural moral hamming distance (SMHD)
        smhd = compute_smhd(self.bayesian_network, dag)

        # If necessary, print these metrics
        if verbose >= 2:
            print("\n FINAL RESULTS \n\n")
            print("- Time taken: {}".format(time_taken))
            # print("Log likelihood: {}".format(log_likelihood))
            print("- Average Markov mantle size: {}".format(average_markov))
            print("- Difference in average Markov mantle sizes: {}".format(average_markov_difference))
            print("- SMHD: {}".format(smhd))
        if verbose >= 4:
            dag.to_daft().show()

        return dag

    # HELPER METHODS #

    def _compute_bdeu_delta(self, node, original_parents, new_parents):
        """
        Given a node and the original and new set of parents (after an operation to add, remove
        or invert and edge), computes the difference in BDeU score between the new and old set of parents.

        In addition, returns the total number of checks and the number of checks that required a
        new BDeU score computation.

        Parameters
        ----------
        node: str
            The children node
        original_parents: list
            The original parents of node
        new_parents: list
            The new parents of node after the operation

        Returns
        -------
        tuple[float, int, int]
        """

        # Counters
        operations = 0
        computed_operations = 0

        # ORIGINAL PARENTS
        # Check if the BDeU score already exists
        if self.bdeu_cache.has_bdeu(node, original_parents):
            # BDeU score exists: retrieve it
            original_bdeu = self.bdeu_cache.get_bdeu_score(node, original_parents)

            operations += 1
        else:
            # BDeU score does not exist: compute it
            original_bdeu = self.bdeu_scorer.local_score(node, original_parents)
            self.bdeu_cache.insert_bdeu_score(node, original_parents, original_bdeu)

            operations += 1
            computed_operations += 1

        # NEW PARENTS
        # Check if the BDeU score already exists
        if self.bdeu_cache.has_bdeu(node, new_parents):
            # BDeU score exists: retrieve it
            new_bdeu = self.bdeu_cache.get_bdeu_score(node, new_parents)

            operations += 1
        else:
            # BDeU score does not exist: compute it
            new_bdeu = self.bdeu_scorer.local_score(node, new_parents)
            self.bdeu_cache.insert_bdeu_score(node, new_parents, new_bdeu)

            operations += 1
            computed_operations += 1

        # Compute the BDeU delta
        bdeu_delta = new_bdeu - original_bdeu

        return bdeu_delta, operations, computed_operations
