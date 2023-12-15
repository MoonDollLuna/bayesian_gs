# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

"""
This module contains problem-agnostic structural learning algorithms, designed to learn
the structure of a Bayesian Network from data
"""

from .base_algorithm import BaseAlgorithm

from .hill_climbing.hill_climbing_utilities import find_legal_hillclimbing_operations
from .hill_climbing.hill_climbing import HillClimbing
from .hill_climbing.parallel_hill_climbing import ParallelHillClimbing
