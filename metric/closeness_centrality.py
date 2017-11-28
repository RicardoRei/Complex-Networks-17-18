import operator
import networkx as nx
import sys
sys.path.append('..')
from build_network import load_network, load_nodes, evaluate_metric

def run():
    nodes = load_nodes()
    network = load_network()
    network.remove_node(548)  # este gajo faltou as aulas...
    closeness_centrality = nx.closeness_centrality(load_network())
    evaluate_metric("Closeness Centrality", closeness_centrality)


if __name__ == '__main__':
    run()