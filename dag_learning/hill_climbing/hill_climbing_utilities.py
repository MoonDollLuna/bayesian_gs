# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

"""
This files contains helper methods shared by all Hill Climbing implementations. These methods
are kept in a different file for ease of reuse.
"""

# IMPORTS #
from itertools import permutations
import networkx as nx
from pgmpy.base import DAG


# OPERATION COMPUTATION #

def find_legal_hillclimbing_operations(dag):
    # TODO LOOPS 
    """
    Given a DAG and a set of variables, find all legal operations, creating three sets containing:
        - All possible edges to add.
        - All possible edges to remove.
        - All possible edges to invert.

    All of these operations are included into a single set, where each element has shape:

        (operation [add, remove, invert], (origin node, destination node))

    This takes care of avoiding possible cycles and trying illegal operations during the hill climbing process.

    Parameters
    ----------
    dag: ExtendedDAG
        DAG over which the operations are tried

    Returns
    -------
    set
    """

    # Get the list of nodes from the DAG
    nodes = list(dag.nodes())

    # EDGE ADDITIONS #

    # Generate the initial set of possible additions (all possible permutations of nodes)
    add_edges = set([("add", permutation) for permutation in permutations(nodes, 2)])

    # Remove invalid edge additions
    # Remove existing edges
    add_edges = add_edges - set([("add", edge) for edge in list(dag.edges)])
    # Remove inverted edges that already exist
    add_edges = add_edges - set([("add", (Y, X)) for (X, Y) in list(dag.edges)])
    # Remove edges that can lead to a cycle
    add_edges = add_edges - set([("add", (X, Y)) for (_, (X, Y)) in add_edges if nx.has_path(dag, Y, X)])

    # EDGE REMOVALS #

    # Generate the initial set of possible removals (only the existing edges)
    remove_edges = set([("remove", edge) for edge in list(dag.edges)])

    # EDGE INVERSIONS

    # Generate the initial set of possible removals (only the existing edges)
    invert_edges = set([("invert", edge) for edge in list(dag.edges)])

    # Remove the edges that, when inverted, would lead to a cycle
    invert_edges = invert_edges - set([("invert", (X, Y)) for (_, (X, Y)) in invert_edges if not any(map(lambda path: len(path) > 2, nx.all_simple_paths(dag, X, Y)))])

    print(remove_edges)
    print(invert_edges)

    # Join all sets into a single set
    return add_edges | remove_edges | invert_edges


def compute_average_markov_mantle(dag):
    """
    Given a directed acyclic graph (DAG), compute the average Markov mantle (the average number of
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
    nodes = list(dag)

    # Get the total mantle size for all nodes
    total_mantle_size = sum([len(dag.get_markov_blanket(x)) for x in nodes])
    return total_mantle_size / len(nodes)


def compute_smhd(original_dag, obtained_dag):
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
