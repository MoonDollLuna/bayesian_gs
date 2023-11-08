# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #

import argparse
import re
from os.path import exists

# DICTIONARIES #

# These dictionaries are used for simpler and faster lookup of all BNLearn bayesian networks
# This is assuming that the original file locations are used - modify the paths if necessary
# If a bayesian network is not included within BNLearn, the direct path to the file can also be provided.

# List of valid, known BNLearn networks
bnlearn_networks = ["asia", "cancer", "earthquake", "sachs", "survey",
                    "alarm", "barley", "child", "insurance", "mildew", "water",
                    "hailfinder", "hepar2", "win95pts",
                    "andes", "diabetes", "link", "munin1", "pathfinder", "pigs",
                    "munin", "munin2", "munin3", "munin4"]

# Paths to the BIF file
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
        description="Performs iterations of a hill-climbing based DAG building algorithm in order "
                    "to achieve a good enough DAG based on the specified data.",
        epilog="Note that if the arguments are passed using \"-c\", the rest of the arguments will be ignored.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add all necessary arguments
    # Default values are directly added to the parser

    # JSON CONFIG #

    # If a JSON string is passed using config, the rest of the arguments will be ignored
    parser.add_argument("-c",
                        "--config",
                        metavar="json_string",
                        help="JSON string including the hyperparameters for this script. All \" characters "
                             "must be escaped. NOTE: If this argument is specified, the JSON string arguments "
                             "will be used instead of any other specified arguments.")

    # ARGUMENTS #

    # Dataset used - either a known BNLearn network or the path to a CSV file.
    parser.add_argument("-ds",
                        "--dataset",
                        metavar="path or {asia-[1,...,10]-10000, cancer-[1,...,10]-10000...}",
                        help="Dataset used to build the DAG on. Either an already existing BNLearn Bayesian Network "
                             "(such as Asia or Andes) or the path to a CSV file may be provided. If a BNLearn Bayesian "
                             "Network is used, a dataset number (between 1 and 10) and a dataset size (currently only "
                             "10000) must be specified to choose which pre-existing CSV file to use.",
                        default="asia-1-10000")

    # BIF file used for statistics - either a known BNLearn network or the path to a BIF file.
    parser.add_argument("-bif",
                        "--bif",
                        metavar="path or {asia, cancer, earthquake...}",
                        help="Path to the Bayesian Network used for DAG statistics after the learning process. "
                             "Either the path to a BIF file or an already existing BNLearn network (such as "
                             "'asia', 'cancer'...) can be provided.",
                        default="asia")

    # Algorithm used - which version of Hill Climbing to use
    parser.add_argument("-alg",
                        "--algorithm",
                        choices=["hillclimbing"],
                        help="Algorithm to perform.",
                        default="hillclimbing")

    # Scoring method used within the algorithm
    parser.add_argument("-s",
                        "--score",
                        choices=["bdeu", "bic", "aic", "ll"],
                        help="Scoring method used to measure the quality of the DAG during the algorithm.",
                        default="bdeu")

    # SCORE-SPECIFIC ARGUMENTS #

    # BDEU SPECIFIC ARGUMENTS
    # BDeu equivalent sample size
    parser.add_argument("-bdeus",
                        "--bdeu_sample_size",
                        type=int,
                        metavar="[bdeus > 0]",
                        help="Equivalent sample size used for BDeu scoring.",
                        default=10)

    # Verbosity (between 0 and 6)
    parser.add_argument("-v",
                        "--verbosity",
                        type=int,
                        metavar="[v >= 0]",
                        help="Level of verbosity of the algorithm. 0 refers to a silent algorithm",
                        default=6)

    # ALGORITHM ARGUMENTS

    # Path to the starting DAG
    parser.add_argument("-sdag",
                        "--starting_dag_path",
                        metavar="path",
                        help="Path to the starting DAG for the algorithm. If not specified, an empty DAG "
                             "will be used instead.",
                        default=None)

    # Epsilon - minimum change in score required to accept the action
    parser.add_argument("-eps",
                        "--epsilon",
                        type=float,
                        metavar="[hce >= 0.0]",
                        help="Minimum change in score required to accept an action during Hill Climbing.",
                        default=0.0001)

    # Maximum number of iterations
    parser.add_argument("-iter",
                        "--algorithm_iterations",
                        type=int,
                        metavar="[hci > 0]",
                        help="Maximum number of iterations performed during Hill Climbing.",
                        default=1e6)

    # Size of the sample used for the log likelihood
    parser.add_argument("-logs",
                        "--loglikelihood_sample_size",
                        type=int,
                        metavar="[hcl > 0]",
                        help="ONLY USED IF A BIF FILE IS SPECIFIED. "
                             "Size (in instances) of the sample used for log likelihood.",
                        default=1000)

    # RESULTS ARGUMENTS #

    # Path to store the results (without the file name)
    parser.add_argument("-rlp",
                        "--results_log_path",
                        help="Folder where the results log should be stored. If not specifed, no results logging "
                             "is done.",
                        metavar="path",
                        default=None)

    # Resulting BIF path - Path where the resulting BIF file will be stored in
    parser.add_argument("-rbp",
                        "--results_bif_path",
                        help="Folder where the resulting BIF file (resulting DAG plus estimated CPDs) will be stored. "
                             "If not specified, no BIF will be stored.",
                        metavar="path",
                        default=None)

    # Name of the output file (without extension)
    parser.add_argument("-rn",
                        "--results_file_name",
                        help="Name of the results log file. If a CSV file was specified, "
                             "this argument can be ignored by automatically using the CSV file name.",
                        metavar="name",
                        default=None)

    # Flush frequency - how often the results log is updated
    parser.add_argument("-rf",
                        "--results_flush_frequency",
                        type=int,
                        metavar="[rf > 0]",
                        help="Update frequency (in seconds) of the results log file.",
                        default=300)

    # ARGUMENT PARSING, PRE-PROCESSING AND EXECUTION #

    # Parse the arguments and, if required, use the JSON string instead
    # The values in the JSON string will be appended on top of the default values - to avoid missing values
    # for non-specified arguments in the JSON

    arguments = vars(parser.parse_args())
    if arguments.get("config"):
        import json
        arguments |= json.loads(arguments["config"])

    # Start parsing, sanitizing and pre-processing all present arguments

    # DATASET AND BIF #
    # Both the dataset path and the BIF paths are processed within the algorithm

    # Dataset - check if it's a path or a properly formatted option (with format "name-[1-10]-10000")
    bnlearn_regex = "|".join(bnlearn_networks)
    file_regex = re.search(r"(?P<dataset>{})-(?P<id>[1-9]|10)-(?P<size>10000)".format(bnlearn_regex),
                           arguments["dataset"])

    # If there is a match - load the appropriate path
    if file_regex:
        csv_path = dataset_paths[file_regex.group("dataset")].format(file_regex.group("size"),
                                                                     file_regex.group("size"),
                                                                     file_regex.group("id"))
        # Ensure that the CSV file actually exists
        if not exists(csv_path):
            raise ValueError(f"{arguments['dataset']} cannot find the CSV in the expected route.")
    else:
        # Check that the path actually exists - and if not, raise an exception
        csv_path = arguments["dataset"]
        if not exists(csv_path):
            raise ValueError(f"{csv_path} is either not a valid path or not a properly formatted BNLearn dataset.")

    # Bif - check if it's a BNLearn network or a path
    if bif_path := bif_paths.get(arguments["bif"]):
        # If it's a path, ensure that the BIF file actually exists
        if not exists(bif_path):
            raise ValueError(f"{arguments['bif']} cannot find the BIF in the expected route.")
    # If it's a path, ensure that the path actually exists - and if not, raise an exception
    else:
        # Check that the path actually exists - and if not, raise an exception
        bif_path = arguments["bif"]
        if not exists(csv_path):
            raise ValueError(f"{bif_path} is either not a valid path or not a properly formatted BNLearn dataset.")

    # ALGORITHM AND SCORE ARGUMENTS
    algorithm = arguments["algorithm"]
    score_method = arguments["score"]

    # BDeu specific arguments
    if score_method == "bdeu":
        bdeu_esz = arguments["bdeu_sample_size"]

    # VERBOSITY AND RESULTS LOGGING ARGUMENTS
    verbosity = arguments["verbosity"]

    results_log_path = arguments["results_log_path"]
    results_bif_path = arguments["results_bif_path"]
    results_file_name = arguments["results_file_name"]
    # Flush frequency must be a positive integer
    results_flush_frequency = arguments["results_flush_frequency"]
    if results_flush_frequency <= 0:
        raise ValueError("Results flush frequency must be a positive number.")

    # ALGORITHM SPECIFIC ARGUMENT PARSING AND EXECUTION #
    # Only parse the relevant arguments for the algorithm

    if algorithm == "hillclimbing":
        from dag_learning import HillClimbing

        starting_dag_path = arguments["starting_dag_path"]

        # All numeric values must be positive (or not negative for epsilon)
        epsilon = arguments["epsilon"]
        if epsilon < 0:
            raise ValueError("Epsilon cannot be a negative number.")

        iterations = arguments["algorithm_iterations"]
        if iterations <= 0:
            raise ValueError("The number of iterations must be a positive number.")

        loglikelihood_sample_size = arguments["loglikelihood_sample_size"]
        if loglikelihood_sample_size <= 0:
            raise ValueError("The number of iterations must be a positive number.")

        # Create the Hill Climbing instance and launch the experiment
        if score_method == "bdeu":
            hill_climbing = HillClimbing(csv_path, nodes=None, bayesian_network=bif_path, score_method=score_method,
                                         results_log_path=results_log_path, results_bif_path=results_bif_path,
                                         results_file_name=results_file_name,
                                         results_flush_freq=results_flush_frequency,
                                         bdeu_equivalent_sample_size=bdeu_esz)
        else:
            hill_climbing = HillClimbing(csv_path, nodes=None, bayesian_network=bif_path, score_method=score_method,
                                         results_log_path=results_log_path, results_bif_path=results_bif_path,
                                         results_file_name=results_file_name,
                                         results_flush_freq=results_flush_frequency)

        # Perform Hill Climbing
        resulting_dag = hill_climbing.search(starting_dag=starting_dag_path, epsilon=epsilon,
                                             max_iterations=iterations, log_likelihood_size=loglikelihood_sample_size,
                                             verbose=verbosity)

    else:
        # TODO ADD MORE ALGORITHMS
        raise NotImplementedError("Only HillClimbing is currently implemented")
