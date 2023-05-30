# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS
from dag_learning import HillClimbing

# TODO ADD PROPER ARGUMENT PARSING

# Create and launch the model (own)
hill_climbing = HillClimbing("./input/csv/small/asia/10000/asia-10000_1.csv")
dag = hill_climbing.estimate_dag(verbose=6)
