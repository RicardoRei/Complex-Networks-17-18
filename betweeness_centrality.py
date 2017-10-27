import networkx as nx

from infection_spread import load_network, evaluate_metric


def run():
    betweeness_centrality = nx.betweenness_centrality(load_network(), weight='weight')
    evaluate_metric("Betweeness Centrality", betweeness_centrality)

if __name__ == '__main__':
    run()