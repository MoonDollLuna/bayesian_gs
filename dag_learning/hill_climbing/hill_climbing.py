# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import math

from dag_learning import BaseAlgorithm, \
    find_legal_hillclimbing_operations, compute_average_markov_mantle, compute_smhd, compute_percentage_difference
from dag_architectures import ExtendedDAG

from time import time
import datetime
from tqdm import tqdm


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
    data: str, DataFrame or ndarray
        Dataset or location of the dataset from which the DAG will be built. The following can be specified:

        - A path to a .csv file
        - A DataFrame containing the data and variable names
        - A numpy Array containing the data

        If a numpy array is specified, the variable names MUST be passed as argument.
    nodes: list[str], optional
        List of ordered variable names contained within the data.
        This argument is ignored unless a numpy Array is given as data - in which case, it is mandatory.
    bayesian_network: BayesianNetwork or str, optional
        Bayesian Network (or path to the BIF file describing it) used for final measurements (like
        the log likelihood of the dataset)
    equivalent_sample_size: int, default=10
        Equivalent sample size used to compute BDeu scores.
    bdeu_score_method: {"forloop", "unique", "mask"}, default="unique"
        Method used to count state frequencies. Possible values:

            * "unique": np.unique over the sliced dataset
            * "forloop": Standard for loop
            * "mask": Masking to segment the dataset into smaller datasets with each parent state combinations

        "unique" should be used, other methods are kept for compatibility’s sake.
    results_path: str, optional
        Path to store the results logger file. If not specified, no logging will be done.
    input_file_name: str, optional
        Filename of the input data. Only used if data is not specified as a CSV and if results_path is not None.
    flush_frequency: int, default=300
        Time (in seconds) between results logger flushes / how often the file is written to.
    """

    # CONSTRUCTOR #

    def __init__(self, data, nodes=None, bayesian_network=None,
                 equivalent_sample_size=10, bdeu_score_method="unique",
                 results_path=None, input_file_name=None, flush_frequency=300):

        # Call the super constructor
        super().__init__(data, nodes, bayesian_network,
                         equivalent_sample_size, bdeu_score_method,
                         results_path, input_file_name, flush_frequency)

    # MAIN METHODS #

    def estimate_dag(self, starting_dag=None, epsilon=0.0001, max_iterations=1e6,
                     wipe_cache=False, verbose=0):
        """
        Performs Hill Climbing to find a local best DAG based on BDeU.

        Note that the found DAG may not be optimal, but good enough.

        If specified in the constructor, a log file is also created.

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
        verbose: int, default = 0
            Verbosity of the program, where:
                - 0: No information is printed
                - 1: Final results and iteration numbers are printed
                - 2: Progress bar for each iteration is printed
                - 3: Action taken for each step is printed
                - 4: Intermediate results for each step are printed
                - 5: Image of the final graph is printed
                - 6: DAG is directly printed

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

        # Total operations checked and operations that needed new BDeU calculations
        total_operations: int = 0
        computed_operations: int = 0

        # Number of operations performed of each type
        add_operations: int = 0
        remove_operations: int = 0
        invert_operations: int = 0

        # Initial time and current time taken by the algorithm
        initial_time: float = time()
        time_taken: float = 0.0

        # BDeU metrics #
        # Best BDeu score
        best_bdeu: float = 0.0

        # Delta BDeU (change per iteration)
        bdeu_delta: float = math.inf

        ##############################################
        # Metrics used to evaluate the resulting DAG #
        ##############################################

        # PARAMETER INITIALIZATION #

        # Store the DAG and, if necessary, create an empty one with the existing nodes
        if starting_dag:
            dag = starting_dag
        else:
            dag = ExtendedDAG(self.nodes)

        # If necessary, wipe out the BDeU cache
        if wipe_cache:
            self.score_cache.wipe_cache()

        # MAIN LOOP #

        # If results logging is used, write the initial header and the column names
        if self.results_logger:
            self._write_header(initial_time)
            self._write_column_names()

        # Compute the initial BDeU score
        # It is assumed that none of these scores will have been computed before
        for node in tqdm(list(dag.nodes), desc="Initial BDeU scoring", disable=(verbose < 4)):

            # Compute the BDeU for each node
            best_bdeu += self.bdeu_scorer.local_score(node, dag.get_parents(node))

            # Update the metrics
            computed_operations += 1
            total_operations += 1

        # Log the initial iteration (iteration 0) - this data should not be printed on screen
        initial_time_taken = time() - initial_time
        self._write_iteration_data(0, iterations, "None", "None", "None",
                                   best_bdeu, best_bdeu, computed_operations, computed_operations,
                                   total_operations, total_operations, initial_time_taken, initial_time_taken)

        # If necessary, output the initial BDeU score
        if verbose >= 4:
            print("Initial BDeU score: {}".format(best_bdeu))

        # Run the loop until:
        #   - The BDeU score improvement is not above the tolerance threshold
        #   - The maximum number of iterations is reached
        while iterations < max_iterations and bdeu_delta > epsilon:

            # Reset the delta and specify the currently taken action
            bdeu_delta = 0
            current_best_bdeu = best_bdeu
            action_taken = None

            # Create delta variables to store change in values
            computed_operations_delta = 0
            total_operations_delta = 0

            # Compute all possible actions for the current DAG
            actions = find_legal_hillclimbing_operations(dag)

            # Loop through all actions (using TQDM)
            for action, (X, Y) in tqdm(actions,
                                       desc=("= ITERATION {}: ".format(iterations + 1)),
                                       disable=(verbose < 2)):

                # If necessary, print the header
                if verbose == 1:
                    print("= ITERATION {}".format(iterations + 1))

                # Depending on the action, compute the hypothetical parents list and child
                # Addition
                if action == "add":

                    # Compute the hypothetical parents lists
                    original_parents_list = dag.get_parents(Y)
                    new_parents_list = original_parents_list + [X]

                    # Compute the BDeU delta and the possible new BDeU score
                    current_bdeu_delta, local_total_operations, local_operations_computed = \
                        self._compute_bdeu_delta(Y, original_parents_list, new_parents_list)

                    current_bdeu = best_bdeu + current_bdeu_delta
                    computed_operations_delta = local_operations_computed
                    total_operations_delta = local_total_operations

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
                    current_bdeu_delta, local_total_operations, local_operations_computed = \
                        self._compute_bdeu_delta(Y, original_parents_list, new_parents_list)

                    current_bdeu = best_bdeu + current_bdeu_delta
                    computed_operations_delta = local_operations_computed
                    total_operations_delta = local_total_operations

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
                    current_x_bdeu_delta, local_total_x_operations, local_x_operations_computed = \
                        self._compute_bdeu_delta(X, original_x_parents_list, new_x_parents_list)
                    current_y_bdeu_delta, local_total_y_operations, local_y_operations_computed = \
                        self._compute_bdeu_delta(Y, original_y_parents_list, new_y_parents_list)

                    current_bdeu_delta = current_x_bdeu_delta + current_y_bdeu_delta
                    current_bdeu = best_bdeu + current_bdeu_delta
                    computed_operations_delta = local_x_operations_computed + local_y_operations_computed
                    total_operations_delta = local_total_x_operations + local_total_y_operations

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
                    add_operations += 1
                elif operation == "remove":
                    dag.remove_edge(X, Y)
                    remove_operations += 1
                elif operation == "invert":
                    dag.invert_edge(X, Y)
                    invert_operations += 1

                # Store the best dag_scoring
                best_bdeu = current_best_bdeu

            # Update the metrics

            iterations += 1

            new_time_taken = time()
            time_taken_delta = new_time_taken - time_taken
            time_taken = new_time_taken - initial_time

            action_str, (origin, destination) = action_taken

            # Print and log the required information as applicable
            self._write_iteration_data(verbose, iterations, action_str, origin, destination,
                                       best_bdeu, bdeu_delta, computed_operations, computed_operations_delta,
                                       total_operations, total_operations_delta, time_taken, time_taken_delta)

            # If necessary (debugging purposes), print the full list of nodes and edges
            if verbose >= 6:
                print("- Nodes: {}".format(list(dag.nodes)))
                print("- Edges: {}".format(list(dag.edges)))
                print("")

        # END OF THE LOOP - DAG FINALIZED
        # METRICS COMPUTATION

        # TODO Learn a Bayesian Network for Log Likelihood and results storing

        # Create an empty DAG to compute comparative scores
        empty_dag = ExtendedDAG(self.nodes)

        # BDEU
        empty_bdeu = self.bdeu_scorer.global_score(empty_dag)
        bdeu_diff = best_bdeu - empty_bdeu
        bdeu_percent = compute_percentage_difference(empty_bdeu, best_bdeu)

        # Total actions
        total_actions = add_operations + remove_operations + invert_operations

        # Average Markov mantle
        average_markov = compute_average_markov_mantle(dag)

        # The following metrics require an original DAG or Bayesian Network as an argument
        if self.bayesian_network is not None:

            # Average Markov mantle difference
            original_markov = compute_average_markov_mantle(self.bayesian_network)
            average_markov_diff = average_markov - original_markov
            average_markov_percent = compute_percentage_difference(original_markov, average_markov)

            # Structural moral hamming distance (SMHD)
            smhd = compute_smhd(self.bayesian_network, dag)
            empty_smhd = compute_smhd(self.bayesian_network, empty_dag)
            smhd_diff = empty_smhd - smhd
            smhd_percent = compute_percentage_difference(smhd, empty_smhd)

            # Print the results
            self._write_final_results(verbose, best_bdeu, empty_bdeu, bdeu_diff, bdeu_percent,
                                      total_actions, add_operations, remove_operations, invert_operations,
                                      computed_operations, total_operations, time_taken, average_markov,
                                      original_markov, average_markov_diff, average_markov_percent,
                                      smhd, empty_smhd, smhd_diff, smhd_percent)

        # If no bayesian network is provided, print the data without its related statistics
        else:
            self._write_final_results(verbose, best_bdeu, empty_bdeu, bdeu_diff, bdeu_percent,
                                      total_actions, add_operations, remove_operations, invert_operations,
                                      computed_operations, total_operations, time_taken, average_markov)

        # If the verbosity is high enough, print the final DAG
        if verbose >= 5:
            dag.to_daft().show()

        # TODO STORE DAG
        return dag

    # AUXILIARY METHODS #

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
        if self.score_cache.has_score(node, original_parents):
            # BDeU score exists: retrieve it
            original_bdeu = self.score_cache.get_local_score(node, original_parents)

            operations += 1
        else:
            # BDeU score does not exist: compute it
            original_bdeu = self.bdeu_scorer.local_score(node, original_parents)
            self.score_cache.insert_local_score(node, original_parents, original_bdeu)

            operations += 1
            computed_operations += 1

        # NEW PARENTS
        # Check if the BDeU score already exists
        if self.score_cache.has_score(node, new_parents):
            # BDeU score exists: retrieve it
            new_bdeu = self.score_cache.get_local_score(node, new_parents)

            operations += 1
        else:
            # BDeU score does not exist: compute it
            new_bdeu = self.bdeu_scorer.local_score(node, new_parents)
            self.score_cache.insert_local_score(node, new_parents, new_bdeu)

            operations += 1
            computed_operations += 1

        # Compute the BDeU delta
        bdeu_delta = new_bdeu - original_bdeu

        return bdeu_delta, operations, computed_operations

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
        self.results_logger.write_line("#\t - Score method used: BDeu\n")
        self.results_logger.write_line("#\t - Frequency counting methodology: {}\n".format(self.bdeu_scorer.count_method))
        self.results_logger.write_line("#\t - Equivalent sample size: {}\n\n".format(self.bdeu_scorer.esz))

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
            - BDeu (total)
            - BDeu (delta)
            - Newly computed operations (total)
            - Newly computed operations (delta)
            - Total operations (total)
            - Total operations (delta)
            - Time taken (total)
            - Time taken (delta)
        """

        # Prepare the list
        headers = ["iteration", "action", "origin", "destination",
                   "bdeu", "bdeu_delta",
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
            Total computed operations (operations that needed to compute a new BDeu score) after the iteration
        comp_operations_delta: int
            Difference in computed operations after the iteration
        total_operations: int
            Total operations (including BDeu lookups in the cache) after the iteration.
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

    def _write_final_results(self, verbose, score, empty_score, score_improvement, score_improvement_percent,
                             total_operations, add_operations, remove_operations, invert_operations,
                             computed_scores, total_scores, time_taken,
                             markov, original_markov=None, markov_difference=None, markov_difference_percent=None,
                             smhd=None, empty_smhd=None, smhd_difference=None, smhd_difference_percent=None):
        """
        Write the final experiment results, if applicable, into the results logger.

        If the verbosity is appropriate, the data is also printed as a console output.

        Note that some arguments are optional (since some statistics require a Bayesian Network to be computed)

        Parameters
        ----------
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
            Average Markov mantle size of the final DAG
        original_markov: float, optional
            Average Markov mantle size of the original DAG
        markov_difference: float, optional
            Difference in Markov mantle sizes between the final and the original DAG
        markov_difference_percent: float, optional
            Difference (in percentage) in Markov mantle sizes between the final and the original DAG
        smhd: int, optional
            Structural moralized Hamming distance between the final DAG and the original DAG
        empty_smhd: int, optional
            Structural moralized Hamming distance between an empty DAG and the original DAG
        smhd_difference: int, optional
            Difference in SMHD between the final DAG and an empty DAG
        smhd_difference_percent: float, optional
            Difference (in percentage) in SMHD between the final DAG and an empty DAG
        """

        # If a results logger exists, print the iteration data
        if self.results_logger:
            self.results_logger.write_line("########################################\n")
            self.results_logger.write_line("# FINAL RESULTS \n\n")

            self.results_logger.write_line("# - Final score: {}\n".format(score))
            self.results_logger.write_line("#\t * Score of an empty graph: {}\n".format(empty_score))
            self.results_logger.write_line("#\t * Score improvement: {}\n".format(score_improvement))
            self.results_logger.write_line("#\t * Score improvement (%): {}%\n\n".format(score_improvement_percent))

            self.results_logger.write_line("# - Time taken: {} secs\n\n".format(time_taken))

            self.results_logger.write_line("# - Total operations performed: {}\n".format(total_operations))
            self.results_logger.write_line("#\t * Additions: {}\n".format(add_operations))
            self.results_logger.write_line("#\t * Removals: {}\n".format(remove_operations))
            self.results_logger.write_line("#\t * Inversions: {}\n\n".format(invert_operations))

            self.results_logger.write_line("# - Computed scores: {}\n".format(computed_scores))
            self.results_logger.write_line("# - Total scores (including cache lookups): {}\n\n".format(total_scores))

            self.results_logger.write_line("# - Average Markov mantle size: {}\n".format(markov))

            # If a bayesian network exists, also write additional results
            if self.bayesian_network:
                self.results_logger.write_line(
                    "#\t * Original average Mankov mantle size: {}\n".format(original_markov))
                self.results_logger.write_line("#\t * Markov difference: {}\n".format(markov_difference))
                self.results_logger.write_line("#\t * Markov difference (%): {}%\n\n".format(markov_difference_percent))

                self.results_logger.write_line("# - SMHD: {}\n".format(smhd))
                self.results_logger.write_line("#\t * Empty graph SMHD: {}\n".format(empty_smhd))
                self.results_logger.write_line("#\t * SMHD difference: {}\n".format(smhd_difference))
                self.results_logger.write_line("#\t * SMHD difference (%): {}%\n".format(smhd_difference_percent))

            self.results_logger.write_line("\n")
            self.results_logger.write_line("########################################\n")

        # If the verbosity is appropriate (1 or above), print the values on the console
        if verbose >= 1:
            print("\n FINAL RESULTS \n\n")

            print("# - Final score: {}".format(score))
            print("#\t * Score of an empty graph: {}".format(empty_score))
            print("#\t * Score improvement: {}".format(score_improvement))
            print("#\t * Score improvement (%): {}%\n".format(score_improvement_percent))

            print("# - Time taken: {} secs\n".format(time_taken))

            print("# - Total operations performed: {}".format(total_operations))
            print("#\t * Additions: {}".format(add_operations))
            print("#\t * Removals: {}".format(remove_operations))
            print("#\t * Inversions: {}\n".format(invert_operations))

            print("# - Computed scores: {}".format(computed_scores))
            print("# - Total scores (including cache lookups): {}\n".format(total_scores))

            print("# - Average Markov mantle size: {}".format(markov))

            # If a bayesian network exists, also write additional results
            if self.bayesian_network:
                print("#\t * Original average Mankov mantle size: {}".format(original_markov))
                print("#\t * Markov difference: {}".format(markov_difference))
                print("#\t * Markov difference (%): {}%\n".format(markov_difference_percent))

                print("# - SMHD: {}".format(smhd))
                print("#\t * Empty graph SMHD: {}".format(empty_smhd))
                print("#\t * SMHD difference: {}".format(smhd_difference))
                print("#\t * SMHD difference (%): {}%".format(smhd_difference_percent))
