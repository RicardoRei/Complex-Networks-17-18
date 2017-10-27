import numpy as np
import networkx as nx
from BuildNetwork import load_network

network = load_network()

def main():
	diameter = nx.diameter(network)
	print(diameter)

main()

'''
Diameter:

Diameter is 3.
'''