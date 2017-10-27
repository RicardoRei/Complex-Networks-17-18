import numpy as np
import networkx as nx
from build_network import load_network

network = load_network()

def run():
	eigenv_centr_not_w = nx.eigenvector_centrality(network)
	eigenv_centr_w = nx.eigenvector_centrality(network, weight='weight')

	print("Eigenvector Centrality not weighted: ", eigenv_centr_not_w)
	print("Eigenvector Centrality weighted: ", eigenv_centr_w)

	max_evc_not_w = max(eigenv_centr_not_w.values())
	max_evc_w = max(eigenv_centr_w.values())

	id_max_evc_not_w = list(eigenv_centr_not_w.keys())[list(eigenv_centr_not_w.values()).index(max_evc_not_w)]
	id_max_evc_w = list(eigenv_centr_w.keys())[list(eigenv_centr_w.values()).index(max_evc_w)]

	print("Maximum Eigenvector Centrality not weighted: ", max_evc_not_w)
	print("Maximum Eigenvector Centrality weighted: ", max_evc_w)

	print("ID of Maximum Eigenvector Centrality not weighted: ", id_max_evc_not_w)
	print("ID of Maximum Eigenvector Centrality weighted: ", id_max_evc_w)

if __name__ == '__main__':
    run()

'''
Maximum Eigenvector Centrality not weighted:  0.05848065844832825
Maximum Eigenvector Centrality weighted:  0.10778672661500492
ID of Maximum Eigenvector Centrality not weighted:  171
ID of Maximum Eigenvector Centrality weighted:  520
'''