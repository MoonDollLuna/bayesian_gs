import time
import networkx as nx
from pgmpy.readwrite.BIF import BIFReader
from dag_architectures import ExtendedDAG
from dag_learning import has_path_inversion

# Import an example network
BIF = BIFReader("./input/bif/massive/munin.bif")
model = BIF.get_model()

# Get the nodes and edges from the network into an ExtendedDAG
nodes = list(model.nodes)
edges = list(model.edges)

dag = ExtendedDAG(nodes)
dag.add_edges_from(edges)


# TIMING - FINDING PATH
start, end = edges[0]
initial_time = time.time()
print(nx.has_path(dag, start, end))
print(initial_time - time.time())

# TIMING - FIND PATH USING THE OTHER METHOD
start, end = edges[0]
initial_time = time.time()
print(has_path_inversion(dag, start, end))
print(initial_time - time.time())

# TIMING - ALL PATHS
start, end = edges[0]
initial_time = time.time()
print(any(map(lambda path: len(path) > 2, nx.all_simple_paths(dag, start, end))))
print(initial_time - time.time())