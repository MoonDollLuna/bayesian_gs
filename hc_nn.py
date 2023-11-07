# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #

import argparse
import re
from os.path import exists

# TODO - Hide imports behind entry point based on chosen options to speed up
from pgmpy.readwrite.BIF import BIFReader
from dag_learning import HillClimbing
from dag_architectures import ExtendedDAG

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

    # SCORE ARGUMENTS #

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

    # HILL CLIMBING ARGUMENTS #

    # Path to the starting DAG
    parser.add_argument("-hcd",
                        "--hillclimbing_dag",
                        metavar="path",
                        help="Path to the starting DAG for Hill Climbing. If not specified, an empty DAG "
                             "will be used instead.",
                        default=None)

    # Epsilon - minimum change in score required to accept the action
    parser.add_argument("-hce",
                        "--hillclimbing_epsilon",
                        type=float,
                        metavar="[hce >= 0.0]",
                        help="Minimum change in score required to accept an action during Hill Climbing.",
                        default=0.0001)

    # Maximum number of iterations
    parser.add_argument("-hci",
                        "--hillclimbing_iterations",
                        type=int,
                        metavar="[hci > 0]",
                        help="Maximum number of iterations performed during Hill Climbing.",
                        default=1e6)

    # Size of the sample used for the log likelihood
    parser.add_argument("-hcl",
                        "--hillclimbing-loglikelihood",
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

    # Name of the output file (without extension)
    parser.add_argument("-rn",
                        "--results_name",
                        help="Name of the results log file. If a CSV file was specified, "
                             "this argument can be ignored by automatically using the CSV file name.",
                        metavar="name",
                        default=None)

    # Flush frequency - how often the results log is updated
    parser.add_argument("-rf",
                        "--results_flush",
                        type=int,
                        metavar="[rf > 0]",
                        help="Update frequency (in seconds) of the results log file.",
                        default=300)

    # Resulting BIF path - Path where the resulting BIF file will be stored in
    parser.add_argument("-rbp",
                        "--results_bif_path",
                        help="Folder where the resulting BIF file (resulting DAG plus estimated CPDs) will be stored. "
                             "If not specified, no BIF will be stored.",
                        metavar="path",
                        default=None)

    # ARGUMENT PARSING AND PRE-PROCESSING #

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

    # Dataset - check if its a path or a properly formatted option (with format "name-[1-10]-10000")
    bnlearn_regex = "|".join(bnlearn_networks)
    file_regex = re.search(r"(?P<dataset>{})-(?P<id>[1-9]|10)-(?P<size>10000)", arguments["dataset"])

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

    # Bif - check if its a BNLearn network or a path
    if bif_path := bif_paths.get(arguments["bif"]):
        # If its a path, ensure that the BIF file actually exists
        if not exists(bif_path):
            raise ValueError(f"{arguments['bif']} cannot find the BIF in the expected route.")
    # If its a path, ensure that the path actually exists - and if not, raise an exception
    else:
        # Check that the path actually exists - and if not, raise an exception
        bif_path = arguments["bif"]
        if not exists(csv_path):
            raise ValueError(f"{bif_path} is either not a valid path or not a properly formatted BNLearn dataset.")

    # TODO - CONTINUE PARSING FROM HERE
    # TODO - ENSURE THAT EVERYTHING IS PROPERLY PARSED FROM THE BASE ALGORITHM

    # ALGORITHM AND SCORE #
    if arguments.get("algorithm"):
        algorithm = arguments["algorithm"]

    if arguments.get("score"):
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

    if "hillclimbing_verbosity" in arguments:
        if arguments["hillclimbing_verbosity"]:
            verbose = arguments["hillclimbing_verbosity"]

    # BDEU SCORE #

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
                                     bdeu_equivalent_sample_size=bdeu_equivalent_sample_size)

        # Perform Hill Climbing
        resulting_dag = hill_climbing.search(starting_dag=starting_dag, epsilon=epsilon, max_iterations=max_iterations,
                                             log_likelihood_size=log_likelihood_size, verbose=verbose)

    else:
        # TODO ADD MORE ALGORITHMS
        pass
