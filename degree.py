import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from build_network import load_network

network = load_network()

"""
Average Degree  =  299.850
Average Weighted Degree = 5424.735

Ignoring the weight of a contact we can see by figure 1 that there is a great number of people with 300/350 contacts per day.

Figure 2 gives us an idea of the how likely is for a person to have more than k contacts per day. We can see having a small number of 
contacts is unlikely but having more than 340 contacts is very unlikely too.

but are all contacts equal? What is the distribution of the size of a contact?

In order to see the distribution of the size of a contact we need to look to the weights that represent the sum of all interactions between 2 persons.

figure 4 show us a very interesting thing... The relation between the size of a contact and his probability is a power law!
Which means that as the size of the contact increases the more rare it becames.

The γ in this power law is ≈ 0.667. γ value is typically in the range 2 < γ < 3 for a scale free network

"""

"""
    This Function receives the network graph and plots the degree histogram. 
"""
def plot_weighted_degree_histogram(network):
    degree = nx.degree(network, weight='weight')
    values = map(lambda x: x[1], degree)
    plt.hist(list(values))
    plt.xlabel("Weighted Degree")
    plt.ylabel("Number of Nodes")
    plt.title("Figure 1 - Weighted Degree distribution")
    plt.show()

"""
    This Function receives the network graph and plots the cumulative degree distribution.
    The Cumulative degree distribution cum_Pk represents the fraction of nodes with degree higher than k.
"""
def plot_weighted_cumulative_dist(network):
    degree = nx.degree(network, weight='weight')
    values = list(map(lambda x: x[1], degree))
    max_degree = max(values)
    values_range = len(values)

    cum_Pk = [0] * (max_degree + 50)  # + 50 so we can see the limit going to zero.
    k_values = [0] * (max_degree + 50)
    for k in range(0, max_degree + 50):
        k_values[k] = k
        cum_Pk[k] = len([i for i in values if i >= k]) / values_range

    plt.xlabel("Weighted Degree")
    plt.ylabel("Probability")
    plt.title("Figure 2 - Cumulative Weighted Degree distribution")
    plt.plot(k_values, cum_Pk)
    plt.show()

"""
    This Function receives the network graph and plots Cumulative Distribution of Contacts size or in other words the cumulative distribution
    of the number of interactions.

    The probability cum_Pk represents the probability of an individual has a interaction with size greater than k.
"""
def plot_interactions_cumulative_dist(network):
    weights = []
    for (u,v,w) in network.edges(data='weight'):
        weights.append(w)
    
    max_size = max(weights)
    cum_Pk = [0]*(max_size+50) # + 50 so we can see the limit going to zero.
    k_values = [0]*(max_size+50)
    for k in range(0, max_size+50):
        k_values[k] = k
        cum_Pk[k] = len([i for i in weights if i >= k]) / len(weights)
    
    plt.xlabel("Contact Size (Units: CPRs)")
    plt.ylabel("Cumulative Probability")
    plt.title("Figure 3 - Cumulative Distribution of Contacts time")
    data_points, = plt.plot(k_values, cum_Pk, 'bo', label='Data Points')
    powerLaw, = plt.plot([x**-0.667 for x in k_values if x != 0.0], 'k', label='Pk ≈ k^-γ')
    plt.legend(handles=[data_points, powerLaw])
    plt.show()



"""
    This Function receives the network graph and plots the Contacts size (number of interactions) in log scale.

    The probability Pk represents the probability of an individual has a interaction with size k.
"""
def plot_interactions_number_log_scale(network):
    weights = []
    for (u,v,w) in network.edges(data='weight'):
        weights.append(w)
    
    max_size = max(weights)
    Pk = [0]*(max_size+10) # + 10 so we can see the limit going to zero.
    k_values = [0]*(max_size+10)
    for k in range(0, max_size+10):
        k_values[k] = k
        Pk[k] = len([i for i in weights if i == k]) / len(weights)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Contact Size (Units: CPRs)")
    plt.ylabel("Probability")
    plt.title("Figure 4 - Degree Probability")
    plt.plot(k_values, Pk, 'bo')
    plt.show()

def run():
    plot_weighted_degree_histogram(network=network)
    plot_weighted_cumulative_dist(network=network)
    plot_interactions_cumulative_dist(network=network)
    plot_interactions_number_log_scale(network=network)

if __name__ == '__main__':
    run()

