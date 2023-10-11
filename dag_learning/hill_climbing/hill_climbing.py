# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import math
from time import time
import datetime
import os
from pathlib import Path

from tqdm import tqdm

from typing import Iterable

from dag_learning import BaseAlgorithm, find_legal_hillclimbing_operations
from dag_scoring import average_markov_blanket, structural_moral_hamming_distance, percentage_difference
from dag_architectures import ExtendedDAG

from pgmpy.sampling import BayesianModelSampling
from pgmpy.metrics import log_likelihood_score
from pgmpy.readwrite.BIF import BIFWriter


class HillClimbing(BaseAlgorithm):
    """
    `HillClimbing` implements a simple Greedy Search approach to Bayesian Network structure building.

    The algorithm works in a loop by trying all possible actions over the existing nodes (either adding
    a new edge to the DAG or removing or inverting an already existing one).

    These actions are evaluated by using a total score metric for the Bayesian Network based on the
    data provided, choosing the action that provides the biggest increase in local score.

    This loop is continued until no action improves the current score, at which point a fully constructed
    Bayesian Network based on the existing nodes and data is provided.

    This algorithm serves as a baseline, to which all other algorithms implemented will be compared to.
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
        self._write_iteration_data(0, iterations, "None", "None", "None",
                                   best_score, best_score, computed_checks, computed_checks,
                                   total_checks, total_checks, initial_time_taken, initial_time_taken)

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

            # If necessary, print the header
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
            self._write_iteration_data(verbose, iterations, action_str, origin, destination,
                                       best_score, score_delta, computed_checks, computed_checks_delta,
                                       total_checks, total_checks_delta, time_taken, time_taken_delta)

            # If necessary (debugging purposes), print the full list of nodes and edges
            if verbose >= 6:
                print("- Nodes: {}".format(list(dag.nodes)))
                print("- Edges: {}".format(list(dag.edges)))
                print("")

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

    # TODO Make method generic and possibly move to either utils or results logger
    def _write_iteration_data(self, verbose, iteration, action_performed, origin, destination,
                              score, score_delta, total_checks, total_checks_delta,
                              computed_checks, computed_checks_delta, time_taken, time_taken_delta):
        """
        Writes the iteration data, if applicable, into the results logger.

        If the verbosity is appropriate, the data is also printed as a console output.

        Parameters
        ----------
        verbose: int
            Verbosity of the program, as specified in "estimate_dag"
        iteration: int
            Current iteration of the algorithm
        action_performed: {"add", "remove", "invert"}
            Action chosen by the algorithm for the iteration
        origin: str
            Node of origin for the action
        destination: str
            Node of destination for the action
        score: float
            Total score after the iteration.
        score_delta: float
            Change in total score after the iteration.
        total_checks: int
            Total checks (including local score lookups in the cache) after the iteration.
        total_checks_delta: int
            Difference in total checks after the iteration.
        computed_checks: int
            Total computed checks (actions that needed to compute a new local score) after the iteration
        computed_checks_delta: int
            Difference in computed checks after the iteration
        time_taken: float
            Total time taken (in secs) after the iteration.
        time_taken_delta: float
            Time taken by the iteration.
        """

        # If a results logger exists, print the iteration data
        if self.results_logger:
            it_data = [iteration, action_performed, origin, destination,
                       score, score_delta, computed_checks, computed_checks_delta,
                       total_checks, total_checks_delta, time_taken, time_taken_delta]
            self.results_logger.write_data_row(it_data)

        # Depending on the verbosity level, print the information on the console

        # Verbosity 3: Action chosen
        if verbose >= 3:
            print("- Action taken: {}".format(action_performed))
            print("\t * Origin node: {}".format(origin))
            print("\t * Destination node: {}".format(destination))
            print("")

        # Verbosity 4: Iteration values
        if verbose >= 4:
            print("- Current score: {}".format(score))
            print("\t * Score delta: {}".format(score_delta))
            print("- Computed score checks: {}".format(computed_checks))
            print("\t * Computed score checks delta: {}".format(computed_checks_delta))
            print("- Total score checks: {}".format(total_checks))
            print("\t * Total score checks delta: {}".format(total_checks_delta))
            print("- Total time taken: {} secs".format(time_taken))
            print("\t * Time taken in this iteration: {} secs".format(time_taken_delta))
            print("")

        # Verbosity 6 checks (DAG nodes and edges) are purely for debug and are printed outside of this method
