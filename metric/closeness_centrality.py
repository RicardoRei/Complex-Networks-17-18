import operator
import networkx as nx
import sys
sys.path.append('..')
from build_network import load_network, load_nodes, evaluate_metric

def run():
    nodes = load_nodes()
    closeness_centrality = nx.closeness_centrality(load_network())
    evaluate_metric("Closeness Centrality", closeness_centrality)


if __name__ == '__main__':
    run()