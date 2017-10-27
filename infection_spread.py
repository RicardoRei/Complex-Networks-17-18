import os
import networkx as nx
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
def load_network(contact_strategy_path = addThenChop_path):
    network = nx.Graph()

    nodes = load_nodes()
    edges = load_edges(contact_strategy_path)

    N = len(nodes)

    for i in range(1, N + 1):
        network.add_node(i)

    network.add_weighted_edges_from(edges)
    network.remove_node(548)  # este gajo faltou as aulas...

    return network

def print_node(id):
    return str(id) + "(Role Here)"

def evaluate_metric(metric_name, metric_dictionary):
    min_node, min_value = min(enumerate(metric_dictionary.values()), key = operator.itemgetter(1))
    max_node, max_value = max(enumerate(metric_dictionary.values()), key = operator.itemgetter(1))
    print("Average",metric_name,":", sum(metric_dictionary.values()) / len(metric_dictionary.values()))
    print("Node with lowest",metric_name, print_node(min_node), "-> (", min_value, ")")
    print("Node with highest",metric_name, print_node(max_node), "-> (", max_value, ")")
