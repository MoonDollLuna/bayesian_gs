# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #

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

# These arguments can be passed as either:
#   - Arguments through console (parsed using ArgParse)
#   - A JSON config file (passing the path to the file)

# BIF file to use to load a bayesian network
# Either a known BNLearn file or the path to a BIF file can be provided
bif_file = "asia"

# CSV file containing the dataset to use
# Either a known BNLearn name or the path to a CSV file can be provided
csv_file = "asia"

# Size of the dataset
# By default, all provided datasets are of size 10000
csv_size = 10000

# Algorithm used
# Currently, only HillClimbing is available
algorithm = "HillClimbing"

# Scoring method used within the algorithm
# Currently, only BDeu is available
score_method = "bdeu"

# BDEU SPECIFIC ARGUMENTS
# Frequency counting method
# (Note: it is recommended to use "unique" as it is the most efficient implementation")
bdeu_count_method = "unique"

# Equivalent sample size
bdeu_equivalent_sample_size = 10

# ARGUMENT PARSING #

# TODO ADD PROPER ARGUMENT PARSING

# Read a BIF for extra stats
bif = BIFReader("./input/bif/small/asia.bif")
bn = bif.get_model()

# Create and launch the model (own)
hill_climbing = HillClimbing("./input/csv/small/asia/10000/asia-10000_1.csv", bayesian_network=bn)
dag = hill_climbing.estimate_dag(verbose=6)
