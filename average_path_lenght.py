import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from BuildNetwork import load_network

network = load_network()

def main():
	apl_not_weighted = nx.average_shortest_path_length(network)
	apl_weighted = nx.average_shortest_path_length(network, weight="weight")

main()