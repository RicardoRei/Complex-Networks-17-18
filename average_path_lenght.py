import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from infection_spread import load_network

network = load_network()

def run():
	apl_not_weighted = nx.average_shortest_path_length(network)
	apl_weighted = nx.average_shortest_path_length(network, weight="weight")

if __name__ == '__main__':
    run()

"""
Average Path Length (APL):

Average Path Lenght is 1.6219 considering no weights.
Average Path Lenght is 1.8700 considering weights.

The result without weights means that in average the distance between 2 pair is 1.6219.
(in portuguese) em media uma pessoa não tem contacto com todas as outras pessoas, mas para todas aquelas com quem não teve contacto existe 
alguem (das que teve contacto) que teve contacto com essa pessoa (com quem não teve contacto). 

(contact with ≈ 3/4 of the the total school population)

If we consider weights we are not looking for contacts but we are looking for Interactions which gives a clue of the size of the contacts too.
In average a person the distance between 2 pairs is 1.8700 Interactions. The fact that this number is higher than the one presented before is 
because the there is a lot of small contacts (> 3 Interactions for example). These metric not only measures the distance between 2 pair it will 
also multiply that distance by the weight (duration) of the links in the path between the pair.

APL shows that the network presents a small-world property.

"""