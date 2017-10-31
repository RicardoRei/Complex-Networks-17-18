import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from build_network import load_network

network = load_network()

def run():
	apl_not_weighted = nx.average_shortest_path_length(network)
	apl_weighted = nx.average_shortest_path_length(network, weight="weight")
	print ("weighted Average Path Lenght: %0.4f \nAverage Path Lenght: %0.4f" % (apl_not_weighted, apl_weighted))

if __name__ == '__main__':
    run()
