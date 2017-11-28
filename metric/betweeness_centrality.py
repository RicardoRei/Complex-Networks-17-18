import networkx as nx
import sys
sys.path.append('..')
from build_network import load_network, evaluate_metric

def run():
	network = load_network()
	network.remove_node(548)  # este gajo faltou as aulas...
	betweeness_centrality = nx.betweenness_centrality(network, weight='weight')
	evaluate_metric("Betweeness Centrality", betweeness_centrality)

if __name__ == '__main__':
    run()