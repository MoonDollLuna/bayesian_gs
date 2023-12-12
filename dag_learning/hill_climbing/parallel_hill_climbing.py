# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# Standard library
import math
from time import time
import datetime
import os
from pathlib import Path

# PyDoc
from typing import Iterable

# Loading bars
from tqdm import tqdm

# NN GS speedup
from dag_learning import BaseAlgorithm, find_legal_hillclimbing_operations
from dag_scoring import average_markov_blanket, structural_moral_hamming_distance, percentage_difference
from dag_scoring import ParallelBaseScore
from dag_architectures import ExtendedDAG

# PGMPY - Log Likelihood, bayesian network construction
from pgmpy.sampling import BayesianModelSampling
from pgmpy.metrics import log_likelihood_score
from pgmpy.readwrite.BIF import BIFWriter


# CHILD METHODS #
# An initializer and several parallelized methods are provided
def child_initializer(scorer, dag):
    """
    Initializes the child processes / threads by storing a copy of the local scorer, containing:
        - The full dataset
        - The local cache

    In addition, a copy of the DAG being built is sent to be used for local score computations.

    This way, the full dataset and DAG does not need to be pickled and shared over the network constantly.
    """

    # Declare the global variables
    global local_scorer, local_dag

    # Store the scorer and the original DAG
    local_scorer = scorer
    local_dag = dag


# Dictionary handling
def child_update_dag_dictionary(dictionary, action=None):
    """
    Given a dictionary of cache deltas (new cache entries since the last iteration) and an action
    in the shape (action, (origin, goal)):

        - Update the dictionary with the new delta updates aggregated by the master process.
        - Update the local DAG with the action chosen by the master process.

    Parameters
    ----------
    dictionary: dict
        Dictionary of new cache entries since the last iteration
    action: tuple[str, tuple[str, str]]
        Action in the shape (<add/remove/invert>, (origin, goal))
    """

    # Access the global variables
    global local_scorer, local_dag

    # Update the dictionary with the provided delta
    local_scorer.score_cache.add_dictionary(dictionary)

    # According to the action (if one is provided), update the local DAG
    if action:
        operation, (X, Y) = action

        if operation == "add":
            local_dag.add_edge(X, Y)
        elif operation == "remove":
            local_dag.remove_edge(X, Y)
        elif operation == "invert":
            local_dag.invert_edge(X, Y)


def child_dictionary_get_deltas():
    """
    Returns the cache deltas of each child process / threads, and wipes the cache delta.

    Returns
    -------
    dictionary
    """

    # Access the global variable and update the cache
    global local_scorer

    # Extract the delta, wipe the cache delta and return it
    delta = local_scorer.score_cache.get_delta()
    local_scorer.score_cache.clear_delta()

    return delta


# Main work method
def child_evaluate_actions(chunked_action_list):
    """
    Given a list of actions in the shape (action_name, (start, end)),
    chooses the best action according to the local scorer and returns it, along its score.

    In addition, returns the number of total operation checks and computations (non-cached operation checks).

    Compared to the original method, only the best action selection is performed in a parallel way.
    Final best action selection and application is still handled by the master process.

    Parameters
    ----------
    chunked_action_list: list
        List of actions, pre-chunked by the manager

    Returns
    -------
    tuple[tuple[str, tuple[str, str]], float, int, int]
    """

    # Access the global variables
    global local_scorer, local_dag

    # Counter for current best score delta and action taken
    score_delta = 0
    action_taken = None

    # Tracking of computed and total checks
    total_checks = 0
    computed_checks = 0

    # Loop through all actions - TQDM is not used in this case
    for action, (X, Y) in chunked_action_list:

        # Score improvement from the operation
        action_score_delta = 0.0

        # Depending on the action, compute the hypothetical parents list and child
        # In addition, keep track of computed operations
        # Addition
        if action == "add":

            # Compute the hypothetical parents lists
            original_parents_list = local_dag.get_parents(Y)
            new_parents_list = original_parents_list + [X]

            # TODO: INCLUDE TOTAL AND COMPUTED CHECKS
            # Compute the score delta and the possible new score
            action_score_delta = local_scorer.local_score_delta(Y,
                                                                tuple(original_parents_list),
                                                                tuple(new_parents_list))

        # Removal
        elif action == "remove":

            # Compute the hypothetical parents lists
            original_parents_list = dag.get_parents(Y)
            new_parents_list = original_parents_list[:]
            new_parents_list.remove(X)

            # Compute the score delta and the possible new score
            action_score_delta = self.local_scorer.local_score_delta(Y,
                                                                     tuple(original_parents_list),
                                                                     tuple(new_parents_list))

        # Inversion
        elif action == "invert":

            # Compute the hypothetical parents lists
            # Note: in this case, two families are being changed
            original_x_parents_list = dag.get_parents(X)
            new_x_parents_list = original_x_parents_list + [Y]

            original_y_parents_list = dag.get_parents(Y)
            new_y_parents_list = original_y_parents_list[:]
            new_y_parents_list.remove(X)

            # Compute the score deltas
            x_score_delta = self.local_scorer.local_score_delta(X,
                                                                tuple(original_x_parents_list),
                                                                tuple(new_x_parents_list))
            y_spore_delta = self.local_scorer.local_score_delta(Y,
                                                                tuple(original_y_parents_list),
                                                                tuple(new_y_parents_list))

            # Join the score deltas and the operation deltas
            action_score_delta = x_score_delta + y_spore_delta

        # Operation checked:
        # Compute the final score for the checked operation
        operation_score = best_score + action_score_delta

        # If the action improves the score, store it and all required information
        if operation_score > current_best_score:
            # Best score and delta
            current_best_score = operation_score
            score_delta = action_score_delta
            action_taken = (action, (X, Y))

    # ALL ACTIONS TRIED

    # Check if an operation was chosen
    if action_taken:

        # Apply the chosen operation
        operation, (X, Y) = action_taken

        if operation == "add":
            dag.add_edge(X, Y)
            add_actions += 1
        elif operation == "remove":
            dag.remove_edge(X, Y)
            remove_actions += 1
        elif operation == "invert":
            dag.invert_edge(X, Y)
            invert_actions += 1

        # Store the best score
        best_score = current_best_score
    pass


class ParallelHillClimbing(BaseAlgorithm):
    """
    `ParallelHillClimbing implements a parallelized Greedy Search approach to Bayesian Network structure building,
    by parallelizing the standard HillClimbing algorithm.

    The algorithm works like HillClimbing, obtaining the same results. However, the following key differences are
    present:
    - The local score computation of possible actions is parallelized and distributed amongst several children
      processes or threads for speed-up
    - The "parent" process / thread joins the results, handles the master cache and distributes the workload
    as needed.

    TODO: SEE IF THIS IS TRUE
    This algorithm can be launched as both multi-threaded (same process, multiple threads) or multi-process
    (different processes sharing messages) by using concurrent.futures.
    """

    # MAIN METHODS #

    def search(self, starting_dag=None, epsilon=0.0001, max_iterations=1e6,
               log_likelihood_size=1000, verbose=0):
        """
        Performs Hill Climbing to find a local best DAG based on the score metric specified in the constructor.

        Note that the found DAG may not be optimal, but good enough.

        If specified in the constructor, a log file is also created.

        Parameters
        ----------
        starting_dag: ExtendedDAG, optional
            Starting DAG. If not specified, an empty DAG is used.
        epsilon: float
            Score threshold. If the score does not improve above the specified threshold,
            the algorithm stops
        max_iterations: int
            Maximum number of iterations to perform.
        verbose: int, default = 0
            Verbosity of the program, where:
                - 0: No information is printed
                - 1: Final results and iteration numbers are printed
                - 2: Progress bar for each iteration is printed
                - 3: Action taken for each step is printed
                - 4: Intermediate results for each step are printed
                - 5: Graph is printed (as a string)
        log_likelihood_size: int, default=1000
            Size of the data sample generated for the log likelihood score

        Returns
        -------
        (ExtendedDAG, dict)
        """

        # LOCAL VARIABLE DECLARATION #

        #################################################
        # Metrics used to evaluate the learning process #
        #################################################

        # Iterations performed
        iterations: int = 0

        # Total checks and checks that needed to be computed
        total_checks: int = 0
        computed_checks: int = 0

        # Number of actions performed of each type
        add_actions: int = 0
        remove_actions: int = 0
        invert_actions: int = 0

        # Initial time and current time taken by the algorithm
        initial_time: float = time()
        time_taken: float = 0.0

        # Score metrics #
        # Best score
        best_score: float = 0.0

        # Delta score (change per iteration)
        score_delta: float = math.inf

        # PARAMETER INITIALIZATION #

        # Store the DAG and, if necessary, create an empty one with the existing nodes
        if starting_dag:
            dag = starting_dag
        else:
            dag = ExtendedDAG(self.nodes)

        # MAIN LOOP #

        # Write the initial header info - depending on the scoring method, different data might be shown
        header_dictionary = {
            "Algorithm used": ("Hill Climbing", None, False),
            "Score method used": (self.score_type, None, True)
        }

        if self.score_type == "bdeu":
            header_dictionary["Equivalent sample size"] = (self.local_scorer.esz, None, True)

        header_dictionary.update({
            "Dataset used": (self.dag_name, None, False),
            "Date": (datetime.datetime.fromtimestamp(initial_time), None, False),
            "Timestamp for the date": (initial_time, None, True)
        })

        self.results_logger.write_comment_block("EXPERIMENT DETAILS", header_dictionary,
                                                verbosity=verbose, minimum_verbosity=1)

        # Pre-write the CSV column names
        self._write_column_names()

        # Compute the initial score
        # It is assumed that none of these scores will have been computed before
        for node in tqdm(list(dag.nodes), desc="Initial scoring", disable=(verbose < 4)):
            # Compute the score for each node
            best_score += self.local_scorer.local_score(node, tuple(dag.get_parents(node)))

            # Update the metrics
            computed_checks += 1
            total_checks += 1

        # Log the initial iteration (iteration 0) - this data should not be printed on screen
        initial_time_taken = time() - initial_time
        self.results_logger.write_data_row([iterations, "None", "None", "None",
                                            best_score, best_score, computed_checks, computed_checks,
                                            total_checks, total_checks, initial_time_taken, initial_time_taken])

        # If necessary, output the initial score - this will not be logged
        if verbose >= 4:
            print("Initial score: {}".format(best_score))

        # Run the loop until:
        #   - The score improvement is not above the tolerance threshold
        #   - The maximum number of iterations is reached
        while iterations < max_iterations and score_delta > epsilon:

            # Reset the delta and specify the currently taken action
            score_delta = 0
            current_best_score = best_score
            action_taken = None

            # Compute all possible actions for the current DAG
            actions = find_legal_hillclimbing_operations(dag)

            # If necessary (TQDM not used) print the header
            if verbose == 1:
                print("= ITERATION {}".format(iterations + 1))

            # Loop through all actions (using TQDM)
            for action, (X, Y) in tqdm(actions,
                                       desc=("= ITERATION {}: ".format(iterations + 1)),
                                       disable=(verbose < 2)):

                # Score improvement from the operation
                action_score_delta = 0.0

                # Depending on the action, compute the hypothetical parents list and child
                # Addition
                if action == "add":

                    # Compute the hypothetical parents lists
                    original_parents_list = dag.get_parents(Y)
                    new_parents_list = original_parents_list + [X]

                    # Compute the score delta and the possible new score
                    action_score_delta = self.local_scorer.local_score_delta(Y,
                                                                             tuple(original_parents_list),
                                                                             tuple(new_parents_list))

                # Removal
                elif action == "remove":

                    # Compute the hypothetical parents lists
                    original_parents_list = dag.get_parents(Y)
                    new_parents_list = original_parents_list[:]
                    new_parents_list.remove(X)

                    # Compute the score delta and the possible new score
                    action_score_delta = self.local_scorer.local_score_delta(Y,
                                                                             tuple(original_parents_list),
                                                                             tuple(new_parents_list))

                # Inversion
                elif action == "invert":

                    # Compute the hypothetical parents lists
                    # Note: in this case, two families are being changed
                    original_x_parents_list = dag.get_parents(X)
                    new_x_parents_list = original_x_parents_list + [Y]

                    original_y_parents_list = dag.get_parents(Y)
                    new_y_parents_list = original_y_parents_list[:]
                    new_y_parents_list.remove(X)

                    # Compute the score deltas
                    x_score_delta = self.local_scorer.local_score_delta(X,
                                                                        tuple(original_x_parents_list),
                                                                        tuple(new_x_parents_list))
                    y_spore_delta = self.local_scorer.local_score_delta(Y,
                                                                        tuple(original_y_parents_list),
                                                                        tuple(new_y_parents_list))

                    # Join the score deltas and the operation deltas
                    action_score_delta = x_score_delta + y_spore_delta

                # Operation checked:
                # Compute the final score for the checked operation
                operation_score = best_score + action_score_delta

                # If the action improves the score, store it and all required information
                if operation_score > current_best_score:
                    # Best score and delta
                    current_best_score = operation_score
                    score_delta = action_score_delta
                    action_taken = (action, (X, Y))

            # ALL ACTIONS TRIED

            # Check if an operation was chosen
            if action_taken:

                # Apply the chosen operation
                operation, (X, Y) = action_taken

                if operation == "add":
                    dag.add_edge(X, Y)
                    add_actions += 1
                elif operation == "remove":
                    dag.remove_edge(X, Y)
                    remove_actions += 1
                elif operation == "invert":
                    dag.invert_edge(X, Y)
                    invert_actions += 1

                # Store the best score
                best_score = current_best_score

            # Update the metrics

            # Number of iterations
            iterations += 1

            # Action taken
            if action_taken:
                action_str, (origin, destination) = action_taken
            else:
                # In the last iteration, no actions are taken
                action_str = "None"
                origin = "None"
                destination = "None"

            # Time taken
            new_time_taken = time() - initial_time
            time_taken_delta = new_time_taken - time_taken
            time_taken = new_time_taken

            # Actions performed (true computations and all computations including cache lookups)
            current_total_checks = self.local_scorer.local_score.cache_info().hits + \
                                   self.local_scorer.local_score.cache_info().misses
            current_computed_checks = self.local_scorer.local_score.cache_info().misses

            total_checks_delta = current_total_checks - total_checks
            computed_checks_delta = current_computed_checks - computed_checks

            total_checks = current_total_checks
            computed_checks = current_computed_checks

            # Print and log the required information as applicable
            iteration_key_data = {
                "Action taken": (action_str, None, False),
                "Origin node": (origin, None, True),
                "Destination node": (destination, None, True)
            }

            iteration_extra_data = {
                "Current score": (best_score, None, False),
                "Score delta": (score_delta, None, True),
                "Total score values checked (including cache lookups)": (total_checks, None, False),
                "Total score values checked delta": (total_checks_delta, None, True),
                "Computed score values": (computed_checks, None, False),
                "Computed score values delta": (computed_checks_delta, None, True),
                "Total time taken": (time_taken, "secs", False),
                "Time taken in this iteration": (time_taken_delta, "secs", True)
            }

            self.results_logger.write_data_row([iterations, action_str, origin, destination,
                                                best_score, score_delta, computed_checks, computed_checks_delta,
                                                total_checks, total_checks_delta, time_taken, time_taken_delta])
            self.results_logger.print_iteration_data(iteration_key_data, iteration_extra_data,
                                                     verbosity=verbose, minimum_verbosity=3, minimum_extra_verbosity=4)

        # END OF THE LOOP - DAG FINALIZED
        # METRICS COMPUTATION AND STORAGE

        # Create an empty DAG to compute comparative scores
        empty_dag = ExtendedDAG(self.nodes)

        # Create a bayesian network based on the currently learned DAG with the dataset used
        # and for log likelihood scoring
        current_bn = dag.to_bayesian_network(self.data, self.bayesian_network.states)

        # Local score
        empty_score = self.local_scorer.global_score(empty_dag)
        empty_diff = best_score - empty_score
        empty_percent = percentage_difference(empty_score, best_score)

        # Total actions
        total_actions = add_actions + remove_actions + invert_actions

        # Average Markov blanket
        average_markov = average_markov_blanket(dag)

        # The following metrics require an original DAG or Bayesian Network as an argument
        if self.bayesian_network is not None:

            # Score of the original bayesian network
            original_score = self.local_scorer.global_score(self.bayesian_network)
            original_diff = best_score - original_score
            original_percent = percentage_difference(original_score, best_score)

            # Average Markov blanket difference
            original_markov = average_markov_blanket(self.bayesian_network)
            markov_diff = average_markov - original_markov
            markov_percent = percentage_difference(original_markov, average_markov)

            # Structural moral hamming distance (SMHD)
            smhd = structural_moral_hamming_distance(self.bayesian_network, dag)
            empty_smhd = structural_moral_hamming_distance(self.bayesian_network, empty_dag)
            smhd_diff = empty_smhd - smhd
            smhd_percent = percentage_difference(smhd, empty_smhd)

            # Log likelihood of sampled data
            # Sample data from the original bayesian network
            sampled_data = BayesianModelSampling(self.bayesian_network). \
                forward_sample(log_likelihood_size, show_progress=False)

            # Check the log likelihood for both original and new DAG
            log_likelihood = log_likelihood_score(current_bn, sampled_data)
            original_log_likelihood = log_likelihood_score(self.bayesian_network, sampled_data)
            log_likelihood_diff = log_likelihood - original_log_likelihood
            log_likelihood_percent = percentage_difference(original_log_likelihood, log_likelihood)

        # Write all the data into a dictionary that will be returned
        final_statistics = {
            "Final score": (best_score, None, False),
            "Empty graph score": (empty_score, None, False),
            "Score improvement (empty)": (empty_diff, None, True),
            "Score improvement percentage (empty)": (empty_percent, "%", True)
        }

        if self.bayesian_network:
            final_statistics.update({
                "Original graph score": (original_score, None, False),
                "Score improvement (original)": (original_diff, None, True),
                "Score improvement percentage (original)": (original_percent, "%", True)
            })

        final_statistics.update({
            "Time taken": (time_taken, "secs", False),
            "Total actions performed": (total_actions, None, False),
            "Additions": (add_actions, None, True),
            "Removals": (remove_actions, None, True),
            "Inversions": (invert_actions, None, True),
            "Total score values checked (including cache lookups)": (total_checks, None, False),
            "Computed score values": (computed_checks, None, True),
            "Average Markov blanket size": (average_markov, None, False)
        })

        if self.bayesian_network:
            final_statistics.update({
                "Original graph average Markov blanket size": (original_markov, None, False),
                "Markov blanket size difference": (markov_diff, None, True),
                "Markov blanket size percentage difference": (markov_percent, "%", True),
                "SMHD": (smhd, None, False),
                "Empty graph SMHD": (empty_smhd, None, True),
                "SMHD difference": (smhd_diff, None, True),
                "SMHD percentage difference": (smhd_percent, "%", True),
                "Log likelihood": (log_likelihood, None, False),
                "Original graph log likelihood": (original_log_likelihood, None, True),
                "Log likelihood difference": (log_likelihood_diff, None, True),
                "Log likelihood percentage difference": (log_likelihood_percent, "%", True)
            })

        # Log the final results
        self.results_logger.write_comment_block("FINAL RESULTS", final_statistics,
                                                verbosity=verbose, minimum_verbosity=1)

        # Log the obtained DAG
        self.results_logger.write_dag_block(dag, verbosity=verbose, minimum_verbosity=5)

        # If a path is specified to store the resulting DAG, the DAG will be converted into BIF format and stored
        if self.dag_path:

            # Ensure that the path actually exists
            if not os.path.exists(self.dag_path):
                Path(self.dag_path).mkdir(parents=True, exist_ok=True)

            # Compute the name of the file
            full_dag_path = "{}/{}_{}.bif".format(self.dag_path, self.dag_name, initial_time)

            # Store the BIF
            BIFWriter(current_bn).write_bif(full_dag_path)

        return dag, final_statistics

    # LOGGING METHODS #

    def _write_column_names(self, column_names=("iteration", "action", "origin", "destination",
                                                "score", "score_delta",
                                                "total_checks", "total_checks_delta"
                                                "computed_checks", "computed_checks_delta",
                                                "time_taken", "time_taken_delta")):
        """
        Write the appropriate column names for the header

        Parameters
        ----------
        column_names: Iterable
            Names to write to the results logger
        """

        self.results_logger.write_data_row(column_names)
