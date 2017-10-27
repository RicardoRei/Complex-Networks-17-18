import numpy as np
import networkx as nx
from BuildNetwork import load_network

network = load_network()
<<<<<<< HEAD
def run():
=======

def main():
>>>>>>> be107558a76dee948bdaf732d19be1590ffa1f49
	diameter = nx.diameter(network)
	print(diameter)

if __name__ == '__main__':
    run()

'''
Diameter:

Diameter is 3.
'''