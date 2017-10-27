import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import operator

dataset_root_path = "flu-data" # corpora
parameters_path = '/dropoff=0/minimumDuration=1/deltaT=1620/staticWeightedEdgeList_at=1350_min=540_max=2159.txt'
addThenChop_path = dataset_root_path + '/edgeLists/durationCondition/' + 'addThenChop' + parameters_path
chopThenAdd_path = dataset_root_path + '/edgeLists/durationCondition/' + 'chopThenAdd' + parameters_path
chopThenCount_path = dataset_root_path + '/edgeLists/durationCondition/' + 'chopThenCount' + parameters_path
justChop_path = dataset_root_path + '/edgeLists/durationCondition/' + 'justChop' + parameters_path

"""
    Loads the nodes for the network
    Returns the sorted list of nodes
"""
def load_nodes():
    roles_file = open(dataset_root_path + "/roles.txt", 'r')
    roles = []

    for line in roles_file:
        id = int(line.split()[0])
        role = line.split()[1]
        roles.append((id, role))

    # TODO - Check why the hell the first 12 individuals have the same ID.
    # For now we eliminate the first 12 entries and get 789 roles shouldn't be 788 ?(655 + 73 + 55 + 5)

    roles = sorted(roles, key=lambda id: id)[12:]
    return roles


"""
    Loads the edges
    Returns the list of edges
"""
def load_edges(contact_strategy_path):
    strategy_file = open(contact_strategy_path, 'r')

    edges = []  # starting with addThenChop strategy to define the edges.

    for line in strategy_file:
        id1 = int(line.split()[0])
        id2 = int(line.split()[1])
        weight = int(line.split()[2])
        edge = (id1, id2, weight)

        edges.append(edge)

    return edges


"""
    Returns the NetworkX network structure loaded with the nodes and edges
    Parameter contact_strategy_path - "contact strategy" folder specified (Default = addThenChop)
"""
def load_network(contact_strategy_path=addThenChop_path):
    network = nx.Graph()

    nodes = load_nodes()
    edges = load_edges(contact_strategy_path)

    N = len(nodes)
    E = len(edges)

    for i in range(1, N + 1):
        network.add_node(i)

    network.add_weighted_edges_from(edges)

    return network


"""
    This Function receives the network graph and plots the degree histogram. 
"""
def plotDegreeHistogram(network):
    degree = nx.degree(network)
    values = map(lambda x: x[1], degree)
    plt.hist(list(values))
    plt.xlabel("Degree")
    plt.ylabel("Nodes")
    plt.title("Figure 1 -Degree distribution")
    plt.show()


"""
    This Function receives the network graph and plots the cumulative degree distribution.
    The Cumulative degree distribution cum_Pk represents the fraction of nodes with degree higher than k.
"""
def plotCumulativeDist(network):
    degree = nx.degree(network)
    values = map(lambda x: x[1], degree)
    max_degree = max(values)

    cum_Pk = [0] * (max_degree + 50)  # + 50 so we can see the limit going to zero.
    k_values = [0] * (max_degree + 50)
    for k in range(0, max_degree + 50):
        k_values[k] = k
        cum_Pk[k] = len([i for i in list(map(lambda x: x[1], degree)) if i >= k]) / len(network.nodes())

    plt.xlabel("Cumulative Degree Distribution")
    plt.ylabel("Degree")
    plt.title("Figure 2 - Cumulative Degree distribution")
    plt.plot(k_values, cum_Pk)
    plt.show()


"""
    This Function receives the network graph and plots the degree probability Pk.
    The degree probability Pk represents the probability of a node has a degree  equal to k.
"""
def plotDegreeProbability(network):
    degree = nx.degree(network)
    values = map(lambda x: x[1], degree)
    max_degree = max(values)

    Pk = [0] * (max_degree + 10)  # + 10 so we can see the limit going to zero.
    k_values = [0] * (max_degree + 10)
    for k in range(0, max_degree + 10):
        k_values[k] = k
        Pk[k] = len([i for i in list(map(lambda x: x[1], degree)) if i == k]) / len(network.nodes())

    plt.xlabel("Degree")
    plt.ylabel("Degree probability")
    plt.title("Figure 3 - Degree Probability")
    plt.plot(k_values, Pk, 'bo')
    plt.show()


"""
    This Function receives the network graph and plots Cumulative Distribution of Contacts time.
    The Contact Size probability Pk represents the probability of an individual has a contact of size k.
"""
def plotSizeOfContactsCumDist(network):
    weights = []
    for (u, v, w) in network.edges(data='weight'):
        weights.append(w)

    max_size = max(weights)
    cum_Pk = [0] * (max_size + 50)  # + 50 so we can see the limit going to zero.
    k_values = [0] * (max_size + 50)
    for k in range(0, max_size + 50):
        k_values[k] = k
        cum_Pk[k] = len([i for i in weights if i >= k]) / len(weights)

    plt.xlabel("Contact Size (Units: Interactions)")
    plt.ylabel("Cumulative Probability")
    plt.title("Figure 4 - Cumulative Distribution of Contacts time")
    plt.plot(k_values, cum_Pk, 'bo')
    plt.show()


def plotSizeOfContactsLogScale(network):
    weights = []
    for (u, v, w) in network.edges(data='weight'):
        weights.append(w)

    max_size = max(weights)
    Pk = [0] * (max_size + 10)  # + 10 so we can see the limit going to zero.
    k_values = [0] * (max_size + 10)
    for k in range(0, max_size + 10):
        k_values[k] = k
        Pk[k] = len([i for i in weights if i == k]) / len(weights)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Contact Size (Units: Interactions)")
    plt.ylabel("Probability")
    plt.title("Figure 5 - Degree Probability")
    plt.plot(k_values, Pk, 'bo')
    plt.show()



""" ANALYSIS """

network = load_network()
plotDegreeHistogram(network=network)  # figure 1
plotCumulativeDist(network=network)  # figure 2
# plotDegreeProbability(network=network) #NOT WHATS IS EXPECTED.... Power Law? # figure 3
plotSizeOfContactsCumDist(network=network)  # figure 4
plotSizeOfContactsLogScale(network=network)  # figure 5

degree = nx.degree(network)
degree_values = list(map(lambda x: x[1], degree))
average_degree = sum(degree_values) / len(degree_values)
print(average_degree)

degree = nx.degree(network, weight="weight")
degree_values = list(map(lambda x: x[1], degree))
average_weighted_degree = sum(degree_values) / len(degree_values)
print(average_weighted_degree)

# this takes a while...
# Running this 2 times will generate an error.
# network.remove_node(548) # should we remove this node? its not connected with any other.... it's a student...

# apl_not_weighted = nx.average_shortest_path_length(network)
# apl_weighted = nx.average_shortest_path_length(network, weight="weight")

# apl_weighted = 1.8700778513793304 but actually there is A LOT of edges with weight = 1

# this computatio takes a while

degree_centrality = nx.degree_centrality(network)

max_centrality = max(degree_centrality.values())
min_centrality = min(degree_centrality.values())

max_id = 0  # we want to know who is the most popular and whats the role associated.
min_id = 0
count = 0
for key, value in degree_centrality.items():
    if value == max_centrality:
        max_id = key
    if value == min_centrality:
        min_id = key

max_centrality_role = load_nodes()[max_id]
min_centrality_role = load_nodes()[min_id]

print("Max degree centrality: %0.4f, Id and Role: %s" % (max_centrality, max_centrality_role))
print("Min degree centrality: %0.4f, Id and Role: %s" % (min_centrality, min_centrality_role))

clustering_coefficient = nx.clustering(network, weight = 'weight')
clustering_coefficient_values = clustering_coefficient.values()

min_node, min_clustering_coefficient = min(enumerate(clustering_coefficient.values()), key = operator.itemgetter(1))
max_node, max_clustering_coefficient = max(enumerate(clustering_coefficient.values()), key = operator.itemgetter(1))