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
                - 6: DAG is directly printed
        log_likelihood_size: int, default=1000
            Size of the data sample generated for the log likelihood score

        Returns
        -------
        ExtendedDAG
        """

        # LOCAL VARIABLE DECLARATION #

        #################################################
        # Metrics used to evaluate the learning process #
        #################################################

        # Iterations performed
        iterations: int = 0

        # Total operations checked and operations that needed new local score calculations
        total_operations: int = 0
        computed_operations: int = 0

        # Number of operations performed of each type
        add_operations: int = 0
        remove_operations: int = 0
        invert_operations: int = 0

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

        # If results logging is used, write the initial header and the column names
        if self.results_logger:
            self._write_header(initial_time)
            self._write_column_names()

        # Compute the initial score
        # It is assumed that none of these scores will have been computed before
        for node in tqdm(list(dag.nodes), desc="Initial scoring", disable=(verbose < 4)):
            # Compute the score for each node
            best_score += self.local_scorer.local_score(node, tuple(dag.get_parents(node)))

            # Update the metrics
            computed_operations += 1
            total_operations += 1

        # Log the initial iteration (iteration 0) - this data should not be printed on screen
        initial_time_taken = time() - initial_time
        self._write_iteration_data(0, iterations, "None", "None", "None",
                                   best_score, best_score, computed_operations, computed_operations,
                                   total_operations, total_operations, initial_time_taken, initial_time_taken)

        # If necessary, output the initial score
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

            # ALL OPERATIONS TRIED

            # Check if an operation was chosen
            if action_taken:

                # Apply the chosen operation
                operation, (X, Y) = action_taken

                if operation == "add":
                    dag.add_edge(X, Y)
                    add_operations += 1
                elif operation == "remove":
                    dag.remove_edge(X, Y)
                    remove_operations += 1
                elif operation == "invert":
                    dag.invert_edge(X, Y)
                    invert_operations += 1

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

            # Operations performed (true computations and all computations including cache lookups)
            current_computed_operations = self.local_scorer.local_score.cache_info().misses
            current_total_operations = self.local_scorer.local_score.cache_info().hits + \
                                       self.local_scorer.local_score.cache_info().misses

            computed_operations_delta = current_computed_operations - computed_operations
            total_operations_delta = current_total_operations - total_operations

            computed_operations = current_computed_operations
            total_operations = current_total_operations

            # Print and log the required information as applicable
            self._write_iteration_data(verbose, iterations, action_str, origin, destination,
                                       best_score, score_delta, computed_operations, computed_operations_delta,
                                       total_operations, total_operations_delta, time_taken, time_taken_delta)

            # If necessary (debugging purposes), print the full list of nodes and edges
            if verbose >= 6:
                print("- Nodes: {}".format(list(dag.nodes)))
                print("- Edges: {}".format(list(dag.edges)))
                print("")

        # END OF THE LOOP - DAG FINALIZED
        # METRICS COMPUTATION

        # Create an empty DAG to compute comparative scores
        empty_dag = ExtendedDAG(self.nodes)

        # Create a bayesian network based on the currently learned DAG with the dataset used
        # and for log likelihood scoring
        current_bn = dag.to_bayesian_network(self.data, self.bayesian_network.states)

        # Local score
        empty_score = self.local_scorer.global_score(empty_dag)
        score_diff = best_score - empty_score
        score_percent = percentage_difference(empty_score, best_score)

        # Total actions
        total_actions = add_operations + remove_operations + invert_operations

        # Average Markov blanket
        average_markov = average_markov_blanket(dag)

        # The following metrics require an original DAG or Bayesian Network as an argument
        if self.bayesian_network is not None:

            # Score of the original bayesian network
            original_score = self.local_scorer.global_score(self.bayesian_network)
            original_score_diff = best_score - original_score
            original_score_percent = percentage_difference(original_score, best_score)

            # Average Markov blanket difference
            original_markov = average_markov_blanket(self.bayesian_network)
            average_markov_diff = average_markov - original_markov
            average_markov_percent = percentage_difference(original_markov, average_markov)

            # Structural moral hamming distance (SMHD)
            computed_smhd = structural_moral_hamming_distance(self.bayesian_network, dag)
            empty_smhd = structural_moral_hamming_distance(self.bayesian_network, empty_dag)
            smhd_diff = empty_smhd - computed_smhd
            smhd_percent = percentage_difference(computed_smhd, empty_smhd)

            # Log likelihood of sampled data
            # Sample data from the original bayesian network
            sampled_data = BayesianModelSampling(self.bayesian_network). \
                forward_sample(log_likelihood_size, show_progress=False)

            # Check the log likelihood for both original and new DAG
            log_likelihood = log_likelihood_score(current_bn, sampled_data)
            original_log_likelihood = log_likelihood_score(self.bayesian_network, sampled_data)
            log_likelihood_diff = log_likelihood - original_log_likelihood
            log_likelihood_percent = percentage_difference(original_log_likelihood, log_likelihood)

            # Print the results
            self._write_final_results(dag, verbose, best_score, empty_score, score_diff, score_percent,
                                      total_actions, add_operations, remove_operations, invert_operations,
                                      computed_operations, total_operations, time_taken, average_markov,
                                      original_markov, average_markov_diff, average_markov_percent,
                                      computed_smhd, empty_smhd, smhd_diff, smhd_percent,
                                      log_likelihood, original_log_likelihood,
                                      log_likelihood_diff, log_likelihood_percent,
                                      original_score, original_score_diff, original_score_percent)

        # If no bayesian network is provided, print the data without its related statistics
        else:
            self._write_final_results(dag, verbose, best_score, empty_score, score_diff, score_percent,
                                      total_actions, add_operations, remove_operations, invert_operations,
                                      computed_operations, total_operations, time_taken, average_markov)

        # If a path is specified to store the resulting DAG, the DAG will be converted into BIF format and stored
        if self.dag_path:

            # Ensure that the path actually exists
            if not os.path.exists(self.dag_path):
                Path(self.dag_path).mkdir(parents=True, exist_ok=True)

            # Compute the name of the file
            full_dag_path = "{}/{}_{}.bif".format(self.dag_path, self.dag_name, initial_time)

            # Store the BIF
            BIFWriter(current_bn).write_bif(full_dag_path)

        return dag

    # LOGGING METHODS #

    def _write_header(self, timestamp):
        """
        Writes the header of the log file. The header is a list of comments (started with the # character)
        that specifies:

            - Algorithm used (HillClimbing) and hyperparameters
            - The time of the experiment (in timestamp)
            - The time of the experiment (in date format)

        Parameters
        ----------
        timestamp: float
            Timestamp of the experiment start
        """

        self.results_logger.write_line("########################################\n")

        # Write the experiment info (hyperparameters)
        self.results_logger.write_line("# EXPERIMENT ###########################\n\n")
        self.results_logger.write_line("# * Algorithm used: HillClimbing\n")
        self.results_logger.write_line("#\t - Score method used: {}\n".format(self.score_type))

        # Depending on the scorer used, different attributes might be shown
        if self.score_type == "bdeu":
            self.results_logger.write_line("#\t - Equivalent sample size: {}\n\n".format(self.local_scorer.esz))

        # Write the dataset info
        self.results_logger.write_line("# * Dataset used: {}\n\n".format(self.results_logger.file_name))

        # Write the timestamps
        self.results_logger.write_line("# * Timestamp: {}\n".format(timestamp))
        self.results_logger.write_line("# * Date: {}\n\n".format(datetime.datetime.fromtimestamp(timestamp)))

        # Write the final indication
        self.results_logger.write_line("# Iterations are found below\n")
        self.results_logger.write_line("########################################\n\n")

    def _write_column_names(self):
        """
        Write the appropriate column names for the header, those being:

            - Iteration
            - Action performed (addition, removal, inversion)
            - Origin node
            - Destination node
            - Score (total)
            - Score (delta)
            - Newly computed operations (total)
            - Newly computed operations (delta)
            - Total operations (total)
            - Total operations (delta)
            - Time taken (total)
            - Time taken (delta)
        """

        # Prepare the list
        headers = ["iteration", "action", "origin", "destination",
                   "score", "score_delta",
                   "comp_operations", "comp_operations_delta",
                   "total_operations", "total_operations_delta",
                   "time_taken", "time_taken_delta"]

        self.results_logger.write_data_row(headers)

    def _write_iteration_data(self, verbose, iteration, action_performed, origin, destination,
                              score, score_delta, comp_operations, comp_operations_delta,
                              total_operations, total_operations_delta, time_taken, time_taken_delta):
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
        comp_operations: int
            Total computed operations (operations that needed to compute a new local score) after the iteration
        comp_operations_delta: int
            Difference in computed operations after the iteration
        total_operations: int
            Total operations (including local score lookups in the cache) after the iteration.
        total_operations_delta: int
            Difference in total operations after the iteration.
        time_taken: float
            Total time taken (in secs) after the iteration.
        time_taken_delta: float
            Time taken by the iteration.
        """

        # If a results logger exists, print the iteration data
        if self.results_logger:
            it_data = [iteration, action_performed, origin, destination,
                       score, score_delta, comp_operations, comp_operations_delta,
                       total_operations, total_operations_delta, time_taken, time_taken_delta]
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
            print("- Computed score checks: {}".format(comp_operations))
            print("\t * Computed score checks delta: {}".format(comp_operations_delta))
            print("- Total score checks: {}".format(total_operations))
            print("\t * Total score checks delta: {}".format(total_operations_delta))
            print("- Total time taken: {} secs".format(time_taken))
            print("\t * Time taken in this iteration: {} secs".format(time_taken_delta))
            print("")

        # Verbosity 6 checks (DAG nodes and edges) is purely for debug and is printed outside of this method

    def _write_final_results(self, dag, verbose, score, empty_score, score_improvement, score_improvement_percent,
                             total_operations, add_operations, remove_operations, invert_operations,
                             computed_scores, total_scores, time_taken,
                             markov, original_markov=None, markov_difference=None, markov_difference_percent=None,
                             smhd=None, empty_smhd=None, smhd_difference=None, smhd_difference_percent=None,
                             log=None, original_log=None, log_difference=None, log_difference_percent=None,
                             original_score=None, original_score_difference=None,
                             original_score_difference_percent=None):
        """
        Write the final experiment results, if applicable, into the results logger.

        If the verbosity is appropriate, the data is also printed as a console output.

        Note that some arguments are optional (since some statistics require a Bayesian Network to be computed)

        Parameters
        ----------
        dag: ExtendedDAG
            Final DAG obtained
        verbose: int
            Verbosity of the program, as specified in "estimate_dag"
        score: float
            Total score of the final DAG
        empty_score: float
            Total score of an empty DAG
        score_improvement: float
            Score improvement between the final DAG and an empty DAG
        score_improvement_percent: float
            Score improvement (in percentage) between the final DAG and an empty DAG
        total_operations: int
            Total operations performed by the algorithm
        add_operations: int
            Total addition operations performed by the algorithm
        remove_operations: int
            Total removal operations performed by the algorithm
        invert_operations: int
            Total inversion operations performed by the algorithm
        computed_scores: int
            Total computed scores (without cache lookups)
        total_scores: int
            Total scores checked (including cache lookups)
        time_taken: float
            Time taken (in seconds) to perform the algorithm
        markov: float
            Average Markov blanket size of the final DAG
        original_markov: float, optional
            Average Markov blanket size of the original DAG
        markov_difference: float, optional
            Difference in Markov blanket sizes between the final and the original DAG
        markov_difference_percent: float, optional
            Difference (in percentage) in Markov blanket sizes between the final and the original DAG
        smhd: int, optional
            Structural moralized Hamming distance between the final DAG and the original DAG
        empty_smhd: int, optional
            Structural moralized Hamming distance between an empty DAG and the original DAG
        smhd_difference: int, optional
            Difference in SMHD between the final DAG and an empty DAG
        smhd_difference_percent: float, optional
            Difference (in percentage) in SMHD between the final DAG and an empty DAG
        log: float, optional
            Log likelihood of the sampled data having been generated from the new Bayesian Network
        original_log: float, optional
            Log likelihood of the sampled data having been generated from the original Bayesian Network
        log_difference: float, optional
            Difference in log likelihood between the final and the original DAG
        log_difference_percent: float, optional
            Difference (in percent) in log likelihood between the final and the original DAG
        original_score: float, optional
            Total score of the original DAG
        original_score_difference: float, optional
            Score improvement / difference between the original DAG and the obtained DAG
        original_score_difference_percent: float, optional
            Score improvement / difference (in percentage) between the original DAG and the obtained DAG
        """

        # If a results logger exists, print the iteration data
        if self.results_logger:
            self.results_logger.write_line("########################################\n")
            self.results_logger.write_line("# FINAL RESULTS \n\n")

            self.results_logger.write_line("# - Final score: {}\n".format(score))
            self.results_logger.write_line("#\t * Score of an empty graph: {}\n".format(empty_score))
            self.results_logger.write_line("#\t * Score improvement: {}\n".format(score_improvement))
            self.results_logger.write_line("#\t * Score improvement (%): {}%\n\n".format(score_improvement_percent))

            # If a bayesian network exists, also write the score for the original network
            if self.bayesian_network:
                self.results_logger.write_line("#\t * Score of the original graph: {}\n".format(original_score))
                self.results_logger.write_line("#\t * Score difference: {}\n".format(original_score_difference))
                self.results_logger.write_line(
                    "#\t * Score difference (%): {}%\n\n".format(original_score_difference_percent))

            self.results_logger.write_line("# - Time taken: {} secs\n\n".format(time_taken))

            self.results_logger.write_line("# - Total operations performed: {}\n".format(total_operations))
            self.results_logger.write_line("#\t * Additions: {}\n".format(add_operations))
            self.results_logger.write_line("#\t * Removals: {}\n".format(remove_operations))
            self.results_logger.write_line("#\t * Inversions: {}\n\n".format(invert_operations))

            self.results_logger.write_line("# - Computed scores: {}\n".format(computed_scores))
            self.results_logger.write_line("# - Total scores (including cache lookups): {}\n\n".format(total_scores))

            self.results_logger.write_line("# - Average Markov blanket size: {}\n".format(markov))

            # If a bayesian network exists, also write additional results
            if self.bayesian_network:
                self.results_logger.write_line(
                    "#\t * Original average Markov blanket size: {}\n".format(original_markov))
                self.results_logger.write_line("#\t * Markov difference: {}\n".format(markov_difference))
                self.results_logger.write_line("#\t * Markov difference (%): {}%\n\n".format(markov_difference_percent))

                self.results_logger.write_line("# - SMHD: {}\n".format(smhd))
                self.results_logger.write_line("#\t * Empty graph SMHD: {}\n".format(empty_smhd))
                self.results_logger.write_line("#\t * SMHD difference: {}\n".format(smhd_difference))
                self.results_logger.write_line("#\t * SMHD difference (%): {}%\n\n".format(smhd_difference_percent))

                self.results_logger.write_line("# - Log likelihood: {}\n".format(log))
                self.results_logger.write_line("#\t * Original model log likelihood: {}\n".format(original_log))
                self.results_logger.write_line("#\t * Log likelihood difference: {}\n".format(log_difference))
                self.results_logger.write_line(
                    "#\t * Log likelihood difference (%): {}%\n".format(log_difference_percent))

            self.results_logger.write_line("\n")
            self.results_logger.write_line("########################################\n")

            # Format the DAG edges and print them
            dag_edges = "# " + str(dag).replace("; ", "\n# ")
            self.results_logger.write_line("########################################\n")
            self.results_logger.write_line("# FINAL DAG OBTAINED \n\n")
            self.results_logger.write_line(dag_edges)
            self.results_logger.write_line("\n")
            self.results_logger.write_line("########################################\n")

        # If the verbosity is appropriate (1 or above), print the values on the console
        if verbose >= 1:
            print("\n FINAL RESULTS \n\n")

            print("- Final score: {}".format(score))
            print("\t * Score of an empty graph: {}".format(empty_score))
            print("\t * Score improvement: {}".format(score_improvement))
            print("\t * Score improvement (%): {}%\n".format(score_improvement_percent))

            # If a bayesian network exists, also write the score for the original network
            if self.bayesian_network:
                print("\t * Score of the original graph: {}".format(original_score))
                print("\t * Score difference: {}".format(original_score_difference))
                print("\t * Score difference (%): {}%\n".format(original_score_difference_percent))

            print("- Time taken: {} secs\n".format(time_taken))

            print("- Total operations performed: {}".format(total_operations))
            print("\t * Additions: {}".format(add_operations))
            print("\t * Removals: {}".format(remove_operations))
            print("\t * Inversions: {}\n".format(invert_operations))

            print("- Computed scores: {}".format(computed_scores))
            print("- Total scores (including cache lookups): {}\n".format(total_scores))

            print("- Average Markov blanket size: {}".format(markov))

            # If a bayesian network exists, also write additional results
            if self.bayesian_network:
                print("\t * Original average Markov blanket size: {}".format(original_markov))
                print("\t * Markov difference: {}".format(markov_difference))
                print("\t * Markov difference (%): {}%\n".format(markov_difference_percent))

                print("- SMHD: {}".format(smhd))
                print("\t * Empty graph SMHD: {}".format(empty_smhd))
                print("\t * SMHD difference: {}".format(smhd_difference))
                print("\t * SMHD difference (%): {}%".format(smhd_difference_percent))

                print("- Log likelihood: {}".format(log))
                print("\t * Original model log likelihood: {}".format(original_log))
                print("\t * Log likelihood difference: {}".format(log_difference))
                print("\t * Log likelihood difference (%): {}%\n".format(log_difference_percent))

        # If the verbosity is appropriate (5 or above), print the edges on the console
        if verbose >= 5:
            print("\n FINAL NETWORK EDGES \n")

            dag_edges = "- " + str(dag).replace("; ", "\n- ")
            print(dag_edges)
