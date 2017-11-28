import operator
import time
import networkx as nx
import sys
sys.path.append('..')
from build_network import load_network, evaluate_metric

def run():
	network = load_network()
	network.remove_node(548)  # este gajo faltou as aulas...
	clustering_coefficients = nx.clustering(network, weight='weight')
	evaluate_metric("Clustering Coefficient", clustering_coefficients)

if __name__ == '__main__':
    run()