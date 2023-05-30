# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS
from pgmpy.readwrite.BIF import BIFReader

from dag_learning import HillClimbing

# TODO ADD PROPER ARGUMENT PARSING

# Read a BIF for extra stats
bif = BIFReader("./input/bif/small/asia.bif")
bn = bif.get_model()

# Create and launch the model (own)
hill_climbing = HillClimbing("./input/csv/small/asia/10000/asia-10000_1.csv", bayesian_network=bn)
dag = hill_climbing.estimate_dag(verbose=6)
