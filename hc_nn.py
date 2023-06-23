# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #

import argparse

from pgmpy.readwrite.BIF import BIFReader
from dag_learning import HillClimbing

# DICTIONARIES #

# These dictionaries are used for simpler and faster lookup of all BNLearn bayesian networks
# This is assuming that the original file locations are used - modify the paths if necessary
# If a bayesian network is not included within BNLearn, the direct path to the file can also be provided.
bif_paths = {"asia": ".input/bif/small/asia.bif", # SMALL
             "cancer": ".input/bif/small/cancer.bif",
             "earthquake": ".input/bif/small/earthquake.bif",
             "sachs": ".input/bif/small/sachs.bif",
             "survey": ".input/bif/small/survey.bif",
             "alarm": ".input/bif/medium/alarm.bif", # MEDIUM
             "barley": ".input/bif/medium/barley.bif",
             "child": ".input/bif/medium/child.bif",
             "insurance": ".input/bif/medium/insurance.bif",
             "mildew": ".input/bif/medium/mildew.bif",
             "water": ".input/bif/medium/water.bif",
             "hailfinder": ".input/bif/large/hailfinder.bif", # LARGE
             "hepar2": ".input/bif/large/hepar2.bif",
             "win95pts": ".input/bif/large/win95pts.bif",
             "andes": ".input/bif/very_large/andes.bif", # VERY LARGE
             "diabetes": ".input/bif/very_large/diabetes.bif",
             "link": ".input/bif/very_large/link.bif",
             "munin1": ".input/bif/very_large/munin1.bif",
             "pathfinder": ".input/bif/very_large/pathfinder.bif",
             "pigs": ".input/bif/very_large/pigs.bif",
             "munin": ".input/bif/massive/munin.bif", # MASSIVE
             "munin2": ".input/bif/massive/munin2.bif",
             "munin3": ".input/bif/massive/munin3.bif",
             "munin4": ".input/bif/massive/munin4.bif"}

# The format of the following dictionary assumes that Python's .format() method will be used to fill in the blanks:
#   - Dataset size (by default, only 10000)
#   - Dataset number (from 1 to 10)
dataset_paths = {"asia": ".input/csv/small/asia/{}/asia-{}_{}.csv", # SMALL
                 "cancer": ".input/csv/small/cancer/{}/cancer-{}_{}.csv",
                 "earthquake": ".input/csv/small/earthquake/{}/earthquake-{}_{}.csv",
                 "sachs": ".input/csv/small/sachs/{}/sachs-{}_{}.csv",
                 "survey": ".input/csv/small/survey/{}/survey-{}_{}.csv",
                 "alarm": ".input/csv/medium/alarm/{}/alarm-{}_{}.csv", # MEDIUM
                 "barley": ".input/csv/medium/barley/{}/barley-{}_{}.csv",
                 "child": ".input/csv/medium/child/{}/child-{}_{}.csv",
                 "insurance": ".input/csv/medium/insurance/{}/insurance-{}_{}.csv",
                 "mildew": ".input/csv/medium/mildew/{}/mildew-{}_{}.csv",
                 "water": ".input/csv/medium/water/{}/water-{}_{}.csv",
                 "hailfinder": ".input/csv/large/hailfinder/{}/hailfinder-{}_{}.csv", # LARGE
                 "hepar2": ".input/csv/large/hepar2/{}/hepar2-{}_{}.csv",
                 "win95pts": ".input/csv/large/win95pts/{}/win95pts-{}_{}.csv",
                 "andes": ".input/csv/very_large/andes/{}/andes-{}_{}.csv", # VERY LARGE
                 "diabetes": ".input/csv/very_large/diabetes/{}/diabetes-{}_{}.csv",
                 "link": ".input/csv/very_large/link/{}/link-{}_{}.csv",
                 "munin1": ".input/csv/very_large/munin1/{}/munin1-{}_{}.csv",
                 "pathfinder": ".input/csv/very_large/pathfinder/{}/pathfinder-{}_{}.csv",
                 "pigs": ".input/csv/very_large/pigs/{}/pigs-{}_{}.csv",
                 "munin": ".input/csv/massive/munin/{}/munin-{}_{}.csv", # MASSIVE
                 "munin2": ".input/csv/massive/munin2/{}/munin2-{}_{}.csv",
                 "munin3": ".input/csv/massive/munin3/{}/munin3-{}_{}.csv",
                 "munin4": ".input/csv/massive/munin4/{}/munin4-{}_{}.csv"}

# USER ARGUMENTS #

# TODO POSSIBLY REMOVE

# These arguments can be passed as either:
#   - Arguments through console (parsed using ArgParse)
#   - A JSON config file (passing the path to the file)

# CSV file containing the dataset to use
# Either an existing BNLearn name or the path to a CSV file can be provided
csv_file = "asia"

# Number of the dataset (between 1 and 10)
# This is ignored it a CSV path is specified instead of an existing BNLearn bayesian network
csv_number = 1

# Size of the dataset
# By default, all provided datasets are of size 10000
csv_size = 10000

# BIF file to use to load a bayesian network
# Either an existing BNLearn file or the path to a BIF file can be provided
bif_file = "asia"

# Algorithm used
algorithm = "hillclimbing"

# Scoring method used within the algorithm
# Currently, only BDeu is available
score_method = "bdeu"

# BDEU SPECIFIC ARGUMENTS
# Frequency counting method
# (Note: it is recommended to use "unique" as it is the most efficient implementation")
bdeu_count_method = "unique"

# Equivalent sample size
bdeu_equivalent_sample_size = 10

# RESULTS LOGGING SPECIFIC ARGUMENTS
# Path where the results should be stored (WITHOUT THE FILE NAME)
results_path = None

# Name of the output file (without extension)
# If a CSV file is specified as input, this argument is ignored
output_name = None

# Flush frequency - How often is the results file flushed (in seconds)
# By default, the file is flushed every 5 minutes
flush_frequency = 300

# HILL CLIMBING SPECIFIC ARGUMENTS
# Starting DAG - if a path is specified, a starting DAG will be loaded
starting_dag = None

# Epsilon - minimum increase for local score for an action to be considered
epsilon = 0.0001

# Maximum number of iterations
max_iterations = 1e6

# Whether the score cache needs to be wiped or not
wipe_cache = False

# Level of verbosity
verbose = 0


# ARGUMENT PARSING #

# Create the parser
parser = argparse.ArgumentParser(
    description="Performs iterations of a hill-climbing based DAG building algorithm in order to achieve a good enough "
                "DAG based on the specified data.",
    epilog="Note that the arguments may be provided as a JSON string instead"
)

# Add all necessary arguments

# Dataset used - either a known BNLearn network or the path to a CSV file.
parser.add_argument("-ds",
                    "--dataset",
                    help="Dataset used to build the DAG on. Either an already existing BNLearn Bayesian Network "
                         "(such as Asia or Andes) or the path to a CSV file may be provided. If a BNLearn Bayesian "
                         "Network is used, a dataset number (between 1 and 10) and a dataset size (usually 10000) "
                         "must be specified to choose which CSV file to use.")

# Dataset number (only required if a BNLearn dataset is specified)
parser.add_argument("-dsn",
                    "--dataset-number",
                    type=int,
                    choices=range(1,11),
                    metavar="[1-10]",
                    help="ONLY REQUIRED IF A BNLEARN DATASET IS SPECIFIED. Which of the 10 available datasets for each "
                         "BNLearn bayesian networks should be used.")

# Dataset size (only required if a BNLearn dataset is specified)
parser.add_argument("-dss",
                    "-dataset-size",
                    type=int,
                    choices=[10000],
                    help="ONLY REQUIRED IF A BNLEARN DATASET IS SPECIFIED. Number of instances of the dataset used.")

# BIF file used for statistics - either a known BNLearn network or the path to a BIF file.
parser.add_argument("-bif",
                    "--bif",
                    help="Bayesian Network used for DAG statistics after the hill climbing process. Either an already "
                         "existing BNLearn Bayesian Network (such as Asia or Andes) or "
                         "the path to a BIF file may be provided.")

# Algorithm used - which version of Hill Climbing to use
parser.add_argument("-alg",
                    "--algorithm",
                    choices=["hillclimbing"],
                    help="Algorithm to perform.")

# Scoring method used within the algorithm
parser.add_argument("-s",
                    "--score",
                    choices=["bdeu"],
                    help="Scoring method used to measure the quality of the DAG during the algorithm.")

parser.print_help()