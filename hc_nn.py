# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS
from pgmpy.readwrite.BIF import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import HillClimbSearch
import time
# TODO MOVE DATA SAMPLING TO WITHIN THE CLASS

from dag_learning import HillClimbing

# TODO ADD PROPER ARGUMENT PARSING

# Parse an example BN
BIF = BIFReader("./input/bif/small/asia.bif")
model = BIF.get_model()

# Create and launch the model (own)
hill_climbing = HillClimbing(model, list(model.nodes), "./input/csv/small/asia/10000/asia-10000_1.csv")
dag = hill_climbing.estimate_dag(verbose=6)

# Create and launch the model (PGMPY)
#time = time.time()
#hill_climbing_pgmpy = HillClimbSearch(data)
#dag_pgmpy = hill_climbing_pgmpy.estimate(scoring_method="bdeuscore")
#print(time)
