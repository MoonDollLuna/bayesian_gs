# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

"""
This module contains utility methods related to computing statistics of the proposed algorithms.
"""


def average_markov_blanket(dag):
    """
    Given a directed acyclic graph (DAG), compute the average Markov blanket (the average number of
    parents, children and children's other parents than each node has)

    Parameters
    ----------
    dag: DAG
        A directed acyclic graph

    Returns
    -------
    float
    """

    # Get the list of nodes
    nodes = list(dag.nodes)

    # Get the total blanket size for all nodes
    total_mantle_size = sum([len(dag.get_markov_blanket(x)) for x in nodes])
    return total_mantle_size / len(nodes)


def structural_moral_hamming_distance(original_dag, obtained_dag):
    """
    Computes the Structural Moral Hamming Distance (SMHD) between two DAGs (an original one
    and the one obtained through a structural learning process)

    The Structural Moral Hamming Distance is the number of edges that differ between the moralized graphs
    of both DAGs. That means:
        - Edges that should be added to the obtained DAG to be like the original DAG
        - Edges that should be removed from the obtained DAG to be like the original DAG

    Parameters
    ----------
    original_dag: DAG
        A directed acyclic graph
    obtained_dag: DAG
        A directed acyclic graph

    Returns
    -------
    int
    """

    # Moralize both graphs
    moralized_original_dag = original_dag.moralize()
    moralized_obtained_dag = obtained_dag.moralize()

    # Get the set of edges between both DAGs
    original_dag_edges = set(list(moralized_original_dag.edges))
    obtained_dag_edges = set(list(moralized_obtained_dag.edges))

    # Compute the symmetric difference
    symmetric_difference = original_dag_edges ^ obtained_dag_edges

    return len(symmetric_difference)


def percentage_difference(original_score, new_score):
    """
    Given two values, computes the percentage difference between both values.

    Parameters
    ----------
    original_score: float
    new_score: float

    Returns
    -------
    float
    """

    # As a failsafe, if the original score is 0 the difference is considered to be infinite
    if original_score == 0:
        return "inf"

    return ((new_score - original_score) / abs(original_score)) * 100
