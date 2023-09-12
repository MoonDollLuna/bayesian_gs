# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #

import argparse
import json

# TODO - Hide imports behind entry point based on chosen options to speed up
from pgmpy.readwrite.BIF import BIFReader
from dag_learning import HillClimbing
from dag_architectures import ExtendedDAG

# DICTIONARIES #

# These dictionaries are used for simpler and faster lookup of all BNLearn bayesian networks
# This is assuming that the original file locations are used - modify the paths if necessary
# If a bayesian network is not included within BNLearn, the direct path to the file can also be provided.
bif_paths = {"asia": "./input/bif/small/asia.bif",  # SMALL
             "cancer": "./input/bif/small/cancer.bif",
             "earthquake": "./input/bif/small/earthquake.bif",
             "sachs": "./input/bif/small/sachs.bif",
             "survey": "./input/bif/small/survey.bif",
             "alarm": "./input/bif/medium/alarm.bif",  # MEDIUM
             "barley": "./input/bif/medium/barley.bif",
             "child": "./input/bif/medium/child.bif",
             "insurance": "./input/bif/medium/insurance.bif",
             "mildew": "./input/bif/medium/mildew.bif",
             "water": "./input/bif/medium/water.bif",
             "hailfinder": "./input/bif/large/hailfinder.bif",  # LARGE
             "hepar2": "./input/bif/large/hepar2.bif",
             "win95pts": "./input/bif/large/win95pts.bif",
             "andes": "./input/bif/very_large/andes.bif",  # VERY LARGE
             "diabetes": "./input/bif/very_large/diabetes.bif",
             "link": "./input/bif/very_large/link.bif",
             "munin1": "./input/bif/very_large/munin1.bif",
             "pathfinder": "./input/bif/very_large/pathfinder.bif",
             "pigs": "./input/bif/very_large/pigs.bif",
             "munin": "./input/bif/massive/munin.bif",  # MASSIVE
             "munin2": "./input/bif/massive/munin2.bif",
             "munin3": "./input/bif/massive/munin3.bif",
             "munin4": "./input/bif/massive/munin4.bif"}

# The format of the following dictionary assumes that Python's .format() method will be used to fill in the blanks:
#   - Dataset size (by default, only 10000)
#   - Dataset number (from 1 to 10)
dataset_paths = {"asia": "./input/csv/small/asia/{}/asia-{}_{}.csv",  # SMALL
                 "cancer": "./input/csv/small/cancer/{}/cancer-{}_{}.csv",
                 "earthquake": "./input/csv/small/earthquake/{}/earthquake-{}_{}.csv",
                 "sachs": "./input/csv/small/sachs/{}/sachs-{}_{}.csv",
                 "survey": "./input/csv/small/survey/{}/survey-{}_{}.csv",
                 "alarm": "./input/csv/medium/alarm/{}/alarm-{}_{}.csv",  # MEDIUM
                 "barley": "./input/csv/medium/barley/{}/barley-{}_{}.csv",
                 "child": "./input/csv/medium/child/{}/child-{}_{}.csv",
                 "insurance": "./input/csv/medium/insurance/{}/insurance-{}_{}.csv",
                 "mildew": "./input/csv/medium/mildew/{}/mildew-{}_{}.csv",
                 "water": "./input/csv/medium/water/{}/water-{}_{}.csv",
                 "hailfinder": "./input/csv/large/hailfinder/{}/hailfinder-{}_{}.csv",  # LARGE
                 "hepar2": "./input/csv/large/hepar2/{}/hepar2-{}_{}.csv",
                 "win95pts": "./input/csv/large/win95pts/{}/win95pts-{}_{}.csv",
                 "andes": "./input/csv/very_large/andes/{}/andes-{}_{}.csv",  # VERY LARGE
                 "diabetes": "./input/csv/very_large/diabetes/{}/diabetes-{}_{}.csv",
                 "link": "./input/csv/very_large/link/{}/link-{}_{}.csv",
                 "munin1": "./input/csv/very_large/munin1/{}/munin1-{}_{}.csv",
                 "pathfinder": "./input/csv/very_large/pathfinder/{}/pathfinder-{}_{}.csv",
                 "pigs": "./input/csv/very_large/pigs/{}/pigs-{}_{}.csv",
                 "munin": "./input/csv/massive/munin/{}/munin-{}_{}.csv",  # MASSIVE
                 "munin2": "./input/csv/massive/munin2/{}/munin2-{}_{}.csv",
                 "munin3": "./input/csv/massive/munin3/{}/munin3-{}_{}.csv",
                 "munin4": "./input/csv/massive/munin4/{}/munin4-{}_{}.csv"}

# ARGUMENT DECLARATION #

# Ensure that the entry point is safe (for multiprocessing)
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Performs iterations of a hill-climbing based DAG building algorithm in order to achieve a good enough "
                    "DAG based on the specified data.",
        epilog="Note that if the arguments are passed using \"-c\", the rest of the arguments will be ignored."
    )

    # Add all necessary arguments

    # JSON CONFIG #
    # If a JSON string is passed using config, the rest of the arguments will be ignored
    parser.add_argument("-c",
                        "--config",
                        help="JSON string including the hyperparameters for this script. All \" characters must be escaped."
                             " NOTE: If this argument is "
                             "specified, the JSON string arguments will be used instead of any other specified arguments.")

    # Dataset used - either a known BNLearn network or the path to a CSV file.
    csv_file = "asia"
    parser.add_argument("-ds",
                        "--dataset",
                        metavar="{asia, cancer, earthquake...} or path",
                        help="Dataset used to build the DAG on. Either an already existing BNLearn Bayesian Network "
                             "(such as Asia or Andes) or the path to a CSV file may be provided. If a BNLearn Bayesian "
                             "Network is used, a dataset number (between 1 and 10) and a dataset size (usually 10000) "
                             "must be specified to choose which CSV file to use.")

    # Dataset number (only required if a BNLearn dataset is specified)
    csv_number = 1
    parser.add_argument("-dsn",
                        "--dataset_number",
                        type=int,
                        choices=range(1, 11),
                        metavar="[1-10]",
                        help="ONLY REQUIRED IF A BNLEARN DATASET IS SPECIFIED. Which of the 10 available datasets for each "
                             "BNLearn bayesian networks should be used.")

    # Dataset size (only required if a BNLearn dataset is specified)
    csv_size = 10000
    parser.add_argument("-dss",
                        "-dataset_size",
                        type=int,
                        choices=[10000],
                        help="ONLY REQUIRED IF A BNLEARN DATASET IS SPECIFIED. Size (in instances) of the dataset used.")

    # BIF file used for statistics - either a known BNLearn network or the path to a BIF file.
    bif_file = "asia"
    parser.add_argument("-bif",
                        "--bif",
                        metavar="{asia, cancer, earthquake...} or path",
                        help="Bayesian Network used for DAG statistics after the hill climbing process. Either an already "
                             "existing BNLearn Bayesian Network (such as Asia or Andes) or "
                             "the path to a BIF file may be provided.")

    # Algorithm used - which version of Hill Climbing to use
    algorithm = "hillclimbing"
    parser.add_argument("-alg",
                        "--algorithm",
                        choices=["hillclimbing"],
                        help="Algorithm to perform.")

    # Scoring method used within the algorithm
    score_method = "bdeu"
    parser.add_argument("-s",
                        "--score",
                        choices=["bdeu"],
                        help="Scoring method used to measure the quality of the DAG during the algorithm.")

    # HILL CLIMBING SPECIFIC ARGUMENTS
    # Path to the starting DAG
    starting_dag = None
    parser.add_argument("-hcd",
                        "--hillclimbing_path",
                        help="Path to the starting DAG for Hill Climbing. If not specified, an empty DAG "
                             "will be used instead.")

    # Epsilon - minimum change in score required to accept the action
    epsilon = 0.0001
    parser.add_argument("-hce",
                        "--hillclimbing_epsilon",
                        type=float,
                        metavar="[hce >= 0.0]",
                        help="Minimum change in score required to accept an action during Hill Climbing.")

    # Maximum number of iterations
    max_iterations = 1e6
    parser.add_argument("-hci",
                        "--hillclimbing_iterations",
                        type=int,
                        metavar="[hci > 0]",
                        help="Maximum number of iterations performed during Hill Climbing.")

    # Size of the sample used for the log likelihood
    log_likelihood_size = 1000
    parser.add_argument("-hcl",
                        "--hillclimbing-loglikelihood",
                        type=int,
                        metavar="[hcl > 0]",
                        help="ONLY USED IF A BIF FILE IS SPECIFIED. Size (in instances) of the sample used for "
                             "log likelihood.")

    # Whether to wipe the cache (True) or not (False)
    wipe_cache = False
    parser.add_argument("-hcw",
                        "--hillclimbing_wipe",
                        action="store_true",
                        help="If specified, the score cache will be forcefully wiped before the Hill Climbing algorithm.")

    # Verbosity (between 0 and 6)
    verbose = 0
    parser.add_argument("-hcv",
                        "--hillclimbing_verbosity",
                        choices=range(0, 7),
                        type=int,
                        metavar="[0-6]",
                        help="Level of verbosity of the algorithm.")

    # BDEU SPECIFIC ARGUMENTS
    # Count method for frequency sampling within BDeu scores
    bdeu_count_method = "unique"
    parser.add_argument("-bdeuc",
                        "--bdeu_counting",
                        choices=["unique", "forloop", "mask"],
                        help="Frequency sampling method used for BDeu scoring. NOTE: \"unique\" is the most efficient "
                             "method available, and thus it is recommended.")

    # BDeu equivalent sample size
    bdeu_equivalent_sample_size = 10
    parser.add_argument("-bdeus",
                        "--bdeu_sample_size",
                        type=int,
                        metavar="[bdeus > 0]",
                        help="Equivalent sample size used for BDeu scoring.")

    # RESULTS LOGGING SPECIFIC ARGUMENTS
    # Path to store the results (without the file name)
    results_path = None
    parser.add_argument("-rp",
                        "--results_path",
                        help="Path where the results log should be stored. NOTE: The actual file name should NOT "
                             "be specified.")

    # Name of the output file (without extension)
    output_name = None
    parser.add_argument("-rn",
                        "--results_name",
                        help="Name of the results log file. If a CSV file was specified, this argument will be ignored.")

    # Flush frequency - how often the results log is updated
    flush_frequency = 300
    parser.add_argument("-rf",
                        "--results_flush",
                        type=int,
                        metavar="[rf > 0]",
                        help="Update frequency (in seconds) of the results log file.")

    # Resulting BIF path - Path where the resulting BIF file will be stored in
    resulting_bif_path = None
    parser.add_argument("-rb",
                        "--results_bif",
                        help="If specified, path where the resulting BIF file (resulting DAG plus estimated CPDs) will "
                             "be stored. NOTE: The actual file name should NOT be specified.")

    # ARGUMENT PARSING AND PRE-PROCESSING #

    # Parse the arguments and, if required, use the JSON string instead
    arguments = vars(parser.parse_args())
    if arguments["config"]:
        arguments = json.loads(arguments["config"])

    # Start parsing, sanitizing and pre-processing all present arguments

    # DATASET AND BIF #

    # Dataset arguments (number and size) are parsed before the actual dataset path, in order
    # to be able to use them if necessary
    if "dataset_number" in arguments:
        if arguments["dataset_number"]:
            csv_number = arguments["dataset_number"]

    if "dataset_size" in arguments:
        if arguments["dataset_size"]:
            csv_size = arguments["dataset_size"]

    if "bif" in arguments:
        if arguments["bif"]:
            # Directly extract the BIF path from the dictionary if appropriate,
            if arguments["bif"] in bif_paths:
                bif_file = bif_paths[arguments["bif"]]
            else:
                bif_file = arguments["bif"]

            # Convert the BIF path into an actual Bayesian Network
            bif_file = BIFReader(bif_file).get_model()

    if "dataset" in arguments:
        if arguments["dataset"]:
            # If a BNLearn dataset is specified, prepare the actual path to the CSV file
            if arguments["dataset"] in dataset_paths:
                csv_file = dataset_paths[arguments["dataset"]].format(csv_size, csv_size, csv_number)
            else:
                csv_file = arguments["dataset"]

    # ALGORITHM AND SCORE #

    if "algorithm" in arguments:
        if arguments["algorithm"]:
            algorithm = arguments["algorithm"]

    if "score" in arguments:
        if arguments["score"]:
            score_method = arguments["score"]

    if "hillclimbing_path" in arguments:
        if arguments["hillclimbing_path"]:
            # Path is converted into a DAG directly
            # TODO - Convert Extended DAG depending on type of algorithm
            starting_dag = ExtendedDAG.from_bayesian_network(BIFReader(arguments["hillclimbing_path"]).get_model())

    if "hillclimbing_epsilon" in arguments:
        if arguments["hillclimbing_epsilon"]:
            if arguments["hillclimbing_epsilon"] < 0.0:
                raise ValueError("HillClimbing Epsilon must be a positive number.")
            else:
                epsilon = arguments["hillclimbing_epsilon"]

    if "hillclimbing_iterations" in arguments:
        if arguments["hillclimbing_iterations"]:
            if arguments["hillclimbing_iterations"] <= 0:
                raise ValueError("HillClimbing Iterations must be a positive number.")
            else:
                max_iterations = arguments["hillclimbing_iterations"]

    if "hillclimbing_loglikelihood" in arguments:
        if arguments["hillclimbing_loglikelihood"]:
            if arguments["hillclimbing_loglikelihood"] <= 0:
                raise ValueError("HillClimbing log likelihood must be a positive number.")
            else:
                log_likelihood_size = arguments["hillclimbing_loglikelihood"]

    if "hillclimbing_wipe" in arguments:
        if arguments["hillclimbing_wipe"]:
            wipe_cache = True

    if "hillclimbing_verbosity" in arguments:
        if arguments["hillclimbing_verbosity"]:
            verbose = arguments["hillclimbing_verbosity"]

    # BDEU SCORE #

    if "bdeu_counting" in arguments:
        if arguments["bdeu_counting"]:
            bdeu_count_method = arguments["bdeu_counting"]

    if "bdeu_sample_size" in arguments:
        if arguments["bdeu_sample_size"]:
            if arguments["bdeu_sample_size"] <= 0:
                raise ValueError("BDeu equivalent sample size must be a positive number.")
            else:
                bdeu_equivalent_sample_size = arguments["bdeu_sample_size"]

    # RESULTS LOGGING #

    if "results_path" in arguments:
        if arguments["results_path"]:
            results_path = arguments["results_path"]

    if "results_name" in arguments:
        if arguments["results_name"]:
            output_name = arguments["results_name"]

    if "results_flush" in arguments:
        if arguments["results_flush"]:
            if arguments["results_flush"] <= 0:
                raise ValueError("Flush frequency must be a positive number.")
            else:
                flush_frequency = arguments["results_flush"]

    if "results_bif" in arguments:
        if arguments["results_bif"]:
            resulting_bif_path = arguments["results_bif"]

    # PARAMETER PRE-PROCESSING AND ALGORITHM EXECUTION

    if algorithm == "hillclimbing":

        # Create the Hill Climbing instance
        hill_climbing = HillClimbing(csv_file, nodes=None, bayesian_network=bif_file, score_method=score_method,
                                     results_path=results_path, output_file_name=output_name,
                                     flush_frequency=flush_frequency, resulting_bif_path=resulting_bif_path,
                                     bdeu_equivalent_sample_size=bdeu_equivalent_sample_size,
                                     bdeu_count_method=bdeu_count_method)

        # Perform Hill Climbing
        resulting_dag = hill_climbing.search(starting_dag=starting_dag, epsilon=epsilon,
                                             max_iterations=max_iterations, log_likelihood_size=log_likelihood_size,
                                             wipe_cache=wipe_cache, verbose=verbose)

    else:
        # TODO ADD MORE ALGORITHMS
        pass
