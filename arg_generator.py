# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# This script generates all possible combination of arguments to be tested as experiments (adapted to the
# OS that will be used) and stores them as JSON strings within a file.

# IMPORTS #
from itertools import product

# ARGUMENT DECLARATION #

# Helper structures #

# Size of each network
network_sizes = {"asia": "small",  # SMALL
                 "cancer": "small",
                 "earthquake": "small",
                 "sachs": "small",
                 "survey": "small",
                 "alarm": "medium",  # MEDIUM
                 "barley": "medium",
                 "child": "medium",
                 "insurance": "medium",
                 "mildew": "medium",
                 "water": "medium",
                 "hailfinder": "large",  # LARGE
                 "hepar2": "large",
                 "win95pts": "large",
                 "andes": "very_large",  # VERY LARGE
                 "diabetes": "very_large",
                 "link": "very_large",
                 "munin1": "very_large",
                 "pathfinder": "very_large",
                 "pigs": "very_large",
                 "munin": "massive",  # MASSIVE
                 "munin2": "massive",
                 "munin3": "massive",
                 "munin4": "massive"}

# WINDOWS PATHS #

# BIF file path, where {} represents:
#   1- Network size
#   2- Dataset name
bif_file_path = "./input/bif/{}/{}.bif"

# CSV file path, where {} represents:
#   1 - Network size
#   2 - Dataset name
#   3 - Dataset size
#   4 - Dataset name
#   5 - Dataset size
#   6 - Dataset ID
csv_file_path = "./input/csv/{}/{}/{}/{}-{}_{}.csv"

# Results log path, where {} represents:
#   1 - Dataset name
#   2 - Dataset size
#   3 - Algorithm used
#   4 - Scorer used
results_log_path = "./output/{}/{}/{}/{}/log/"

# Results BIF path, where {} represents:
#   1 - Dataset name
#   2 - Dataset size
#   3 - Algorithm used
#   4 - Scorer used
results_bif_path = "./output/{}/{}/{}/{}/bif/"

# Possible values #
# Network names
network_names = ["asia", "cancer", "earthquake", "sachs", "survey",
                 "alarm", "barley", "child", "insurance", "mildew", "water",
                 "hailfinder", "hepar2", "win95pts",
                 "andes", "diabetes", "link", "munin1", "pathfinder", "pigs",
                 "munin", "munin2", "munin3", "munin4"]

# Dataset IDs [1 - 10]
dataset_ids = list(range(1, 11))

# Dataset sizes
dataset_sizes = [10000]

# Algorithms
algorithms = ["hillclimbing"]

# Scoring methods
scoring_methods = ["bdeu", "ll", "bic", "aic"]

# FILE CREATION AND JSON CREATION#
# File is directly created and handled using the "with" Python interface
with open("experiment_list.txt", "w") as file:

    # Generate all possible combinations of arguments
    arguments_lists = [network_names, dataset_ids, dataset_sizes, algorithms, scoring_methods]
    arguments_combinations = list(product(*arguments_lists))

    # Create an appropriate string for each argument combination
    for network_name, dataset_id, dataset_size, algorithm, scoring_method in arguments_combinations:

        network_size = network_sizes[network_name]

        experiment_string = ("--dataset {} --bif {} --algorithm {} --score {} "
                             "--results_log_path {} --results_bif_path {}\n").format(
            csv_file_path.format(network_size, network_name, dataset_size, network_name, dataset_size, dataset_id),
            bif_file_path.format(network_size, network_name),
            algorithm,
            scoring_method,
            results_log_path.format(network_size, network_name, algorithm, scoring_method),
            results_bif_path.format(network_size, network_name, algorithm, scoring_method)
        )

        # Write the string arguments into the file
        file.write(experiment_string)
