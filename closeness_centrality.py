import operator
import networkx as nx

from infection_spread import load_network, evaluate_metric

def run():
    closeness_centrality = nx.closeness_centrality(load_network())
    evaluate_metric("Closeness Centrality", closeness_centrality)


if __name__ == '__main__':
    run()