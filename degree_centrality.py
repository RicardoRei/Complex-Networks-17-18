import numpy as np
import networkx as nx
from build_network import load_network, load_nodes

network = load_network()

def run():

	'''
	Calculates the degree centrality
	'''

	degree_centrality  = nx.degree_centrality(network)

	print(degree_centrality)

	'''
	Finds the nodes with the maximum and minimum degree centrality, their IDs and their roles
	'''

	max_degree_centrality = max(degree_centrality.values())
	min_degree_centrality = min(degree_centrality.values())

	id_max_dc = list(degree_centrality.keys())[list(degree_centrality.values()).index(max_degree_centrality)]
	id_min_dc = list(degree_centrality.keys())[list(degree_centrality.values()).index(min_degree_centrality)]

	role_max_dc = load_nodes()[id_max_dc-1][1]
	role_min_dc = load_nodes()[id_min_dc-1][1]

	print("Maximum degree centrality: %0.4f, Id: %d and Role: %s" % (max_degree_centrality, id_max_dc, role_max_dc))
	print("Minimum degree centrality: %0.4f, Id: %d and Role: %s" % (min_degree_centrality, id_min_dc, role_min_dc))


if __name__ == '__main__':
    run()