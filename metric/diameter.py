import numpy as np
import networkx as nx
import sys
sys.path.append('..')
from build_network import load_network

network = load_network()

def run():
	diameter = nx.diameter(network)
	print ("Diameter: " + str(diameter))

if __name__ == '__main__':
    run()

'''
Diameter:

Diameter is 3.
'''