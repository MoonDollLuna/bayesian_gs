# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# This program generates several datasets from the specified .BIF files, and stores them into
# the specified folder locations as .CSV files to be reused easily.

###########
# IMPORTS #
###########

from pathlib import Path

from pgmpy.readwrite.BIF import BIFReader
from pgmpy.sampling import BayesianModelSampling

#############
# VARIABLES #
#############

# Modify them here to change dataset generation

# Bayesian networks to generate datasets from
# Specify the networks as {size: [list of networks]}
datasets = {"small": ["asia", "cancer", "earthquake", "sachs", "survey"],
            "medium": ["alarm", "barley", "child", "insurance", "mildew", "water"],
            "large": ["hailfinder", "hepar2", "win95pts"],
            "very_large": ["andes", "diabetes", "link", "munin1", "pathfinder", "pigs"],
            "massive": ["munin", "munin2", "munin3", "munin4"]}

# Format of the bayesian network files
# This code assumes that the files will be in BIF format, but others may be tried
bn_format = "bif"

# Number of datasets to generate from each bayesian network
dataset_count = 10

# Size of the datasets to generate
dataset_size = 10000

######################
# DATASET GENERATION #
######################

# Go through all bayesian network sizes
for size in datasets:

    # For each size, go through all bayesian networks
    for bn in datasets[size]:

        # Print the current BN
        print("\n={} ({})=\n".format(bn, size))

        # Generate the name of the new files
        file_name = "{}-{}".format(bn, dataset_size)

        # Create the path to store the datasets
        Path("./input/csv/{}/{}/{}/".format(size, bn, dataset_size)).mkdir(parents=True, exist_ok=True)

        # Parse the Bayesian Network
        bif = BIFReader("./input/{}/{}/{}.{}".format(bn_format, size, bn, bn_format))
        model = bif.get_model()

        # Repeat the process as specified
        for dataset_id in range(dataset_count):

            # Generate a dataset with the specified size
            data = BayesianModelSampling(model).forward_sample(size=dataset_size)

            # Convert the dataset to a CSV
            data.to_csv("./input/csv/{}/{}/{}/{}_{}.csv".format(size, bn, dataset_size, file_name, dataset_id + 1),
                        index=False)
