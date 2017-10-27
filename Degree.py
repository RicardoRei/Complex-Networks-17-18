import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from infection_spread import load_network

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
    for (u,v,w) in network.edges(data='weight'):
        weights.append(w)
    
    max_size = max(weights)
    cum_Pk = [0]*(max_size+50) # + 50 so we can see the limit going to zero.
    k_values = [0]*(max_size+50)
    for k in range(0, max_size+50):
        k_values[k] = k
        cum_Pk[k] = len([i for i in weights if i >= k]) / len(weights)
    
    plt.xlabel("Contact Size (Units: Interactions)")
    plt.ylabel("Cumulative Probability")
    plt.title("Figure 4 - Cumulative Distribution of Contacts time")
    data_points, = plt.plot(k_values, cum_Pk, 'bo', label='Data Points')
    powerLaw, = plt.plot([x**-0.667 for x in k_values if x != 0.0], 'k', label='Pk ≈ k^-γ')
    plt.legend(handles=[data_points, powerLaw])
    plt.show()


def plotSizeOfContactsLogScale(network):
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
    plt.xlabel("Contact Size (Units: Interactions)")
    plt.ylabel("Probability")
    plt.title("Figure 5 - Degree Probability")
    plt.plot(k_values, Pk, 'bo')
    plt.show()

def main():
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

main()

