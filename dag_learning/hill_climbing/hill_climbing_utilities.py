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
    """
    Given a DAG, find all possible operations, those being:
        - All possible edges to add.
        - All possible edges to remove.
        - All possible edges to invert.

    All of these operations are included into a single list, where each element has shape:

        (operation [add, remove, invert], (origin node, destination node))

    This method filters all actions that may lead to illegal DAGs (cycles). This method also returns
    values as a list to ensure determinism within the algorithm.

    Parameters
    ----------
    dag: ExtendedDAG
        DAG over which the operations are tried

    Returns
    -------
    list[(str, (str, str))]
    """

    # Get the list of nodes and edges from the DAG
    nodes = list(dag.nodes)
    edges = list(dag.edges)

    # Create a set of existing edges, for faster lookup
    set_edges = set(edges)

    # EDGE ADDITIONS #

    # Starting with the initial set of possible additions (all possible permutations of nodes of size 2),
    # filter all additions that:
    #   - Already exist (directly or reversed)
    #   - Would cause a loop
    add_edges = [("add", (X, Y)) for X, Y in permutations(nodes, 2)
                 if _is_legal_addition(dag, set_edges, X, Y)]

    # EDGE REMOVALS #

    # The list of possible edge removals is the list of already existing edges
    # All edge removal operations are legal
    remove_edges = [("remove", edge) for edge in edges]

    # EDGE INVERSIONS

    # The list of possible edge inversions is the list of already existing edges that:
    #   - If inverted, do not cause a loop
    invert_edges = [("invert", (X, Y)) for X, Y in edges
                    if _is_legal_inversion(dag, X, Y)]

    # Extend the original list (add edges) to include the rest of lists
    add_edges.extend(remove_edges)
    add_edges.extend(invert_edges)

    return add_edges


def _is_legal_addition(dag, existing_edges, source, target):
    """
    Returns True if the edge between source and target is a legal edge within dag.

    A legal edge is an edge that:
        - Does not already exist within the dag.
        - Does not have an equivalent reversed edge within the dag.
        - Does not create a loop within the dag.

    Parameters
    ----------
    dag: DAG
        Directed Acyclic Graph on which the path is checked
    existing_edges: set
        Set of all the existing edges within the DAG
    source: str
        Starting node of the path
    target: str
        Ending node of the path

    Returns
    -------
    bool
    """

    # Check if the edge (or its reversed edge) already exists
    # Already existing edges are not legal
    if (source, target) in existing_edges or (target, source) in existing_edges:
        return False

    # If the edge does not exist, check if it would create a loop within the DAG
    # (a loop would be created if there is already a path between target and source)
    if nx.has_path(dag, target, source):
        return False

    # If all of these checks are passed, the path is legal
    return True


def _is_legal_inversion(dag, source, target):
    """
    Returns True if there is NOT a path in between source and target OTHER THAN the already existing path
    between source and target, and False otherwise.

    This method is used to check for possible loops during inversion in an efficient way. To do this, the
    existing edge between source and target is temporarily removed before calling the (efficient)
    NetworkX has_path method.

    Parameters
    ----------
    dag: DAG
        Directed Acyclic Graph on which the path is checked
    source: str
        Starting node of the path
    target: str
        Ending node of the path

    Returns
    -------
    bool
    """

    # Internally remove the already existing path
    # For sanity checking, we check for exceptions (if the edge does not exist, an exception will be raised)
    edge_removed = True
    try:
        dag.remove_edge(source, target)
    except nx.NetworkXError:
        edge_removed = False

    # Check for a path
    path_found = nx.has_path(dag, source, target)

    # If necessary, rebuild the edge
    if edge_removed:
        dag.add_edge(source, target)

    # The inversion is legal if the path is not found
    return not path_found


# STATISTIC COMPUTATIONS #

def compute_average_markov_blanket(dag):
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


def compute_percentage_difference(original_score, new_score):
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

    return ((new_score - original_score) / abs(original_score)) * 100
