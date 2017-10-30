import operator
import time
import networkx as nx

from build_network import load_network, evaluate_metric

def run():

    clustering_coefficients = nx.clustering(load_network(), weight='weight')
    evaluate_metric("Clustering Coefficient", clustering_coefficients)

if __name__ == '__main__':
    run()