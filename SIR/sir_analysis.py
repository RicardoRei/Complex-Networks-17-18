import networkx as nx
import sys
sys.path.append('..')
from build_network import load_network
import numpy as np
import matplotlib.pyplot as plt
from metric.degree import cumulative_degree_dist

def probability_of_infection(beta, weight):
	prob_keep_susceptible = 1 - beta
	for i in range(1, weight):
		prob_keep_susceptible *= 1 - beta # 1 - prob. get infected
	return 1 - prob_keep_susceptible # prob. of being infected is the 1 - prob. of being healthy after k contacts

def random_vaccination(network, states, vaccinated_percentage, efficacy):
	initially_vaccinated = round(len(network.nodes())*vaccinated_percentage) if round(len(network.nodes())*vaccinated_percentage) > 0 else 1
	for sample in np.random.randint(0, len(states), initially_vaccinated):
		states[sample][0] = 2 if np.random.choice([1, 0], p=[efficacy, 1-efficacy]) == 1 else 0
	return states

def random_infection(network, states, infected_percentage):
	initially_infected = round(len(network.nodes())*infected_percentage) if round(len(network.nodes())*infected_percentage) > 0 else 1
	for sample in np.random.randint(0, len(states), initially_infected):
		if states[sample][0] == 0:
			states[sample][0] = 1
			states[sample][1] = 1
	return states

def vaccination_of_hubs(network, states, percentage, efficacy):
	centralities = nx.degree_centrality(network)
	sorted_centralities = sorted(centralities.items(), key=lambda item: (item[1], item[0]))

	i = 1
	while (i/len(network.nodes()) < percentage):
		node = sorted_centralities[-i][0]
		states[node-1][0] = 2 if np.random.choice([1, 0], p=[efficacy, 1-efficacy]) == 1 else 0
		i += 1

	return states

def sir_model_deterministic_recovery(network, infected_percentage, vaccinated_percentage, vaccine_efficacy, beta, days_to_recovery, period=15):
	# initialization
	states = np.zeros((len(network.nodes()), 2))
	states = random_vaccination(network, states, vaccinated_percentage, vaccine_efficacy)
	states = random_infection(network, states, infected_percentage)
	# compartiments for each day
	sir = np.zeros((period, 3))
	# simulations for each time step
	for i in range(period):
		for x in states:
			sir[i][int(x[0])] += 1
		states = sir_iteration_deterministic_recovery(network, states, beta, days_to_recovery)
	return sir

def sir_iteration_deterministic_recovery(network, states, beta, days_to_recovery):
	for node in network.nodes():
		if states[node-1][0] == 0:
			neighbors = nx.all_neighbors(network, node)
			for neighbor in neighbors:
				if states[neighbor-1][0] == 1:
					infection_prob = probability_of_infection(beta, network[node][neighbor]['weight'])
					states[node-1][0] = np.random.choice([0, 1], p=[1 - infection_prob, infection_prob])
					if states[node-1][0] == 1:
						break

		elif states[node-1][0] == 1:
			states[node-1][0] = 2 if states[node-1][1] >= days_to_recovery else 1
			states[node-1][1] += 1
			
	return states

def sir_model_with_recovery_rate(network, infected_percentage, beta, delta, period=15):
	# initialization
	states = np.zeros(len(network.nodes()))
	initial_infected = round(len(network.nodes())*infected_percentage) if round(len(network.nodes())*infected_percentage) > 0 else 1
	for sample in np.random.randint(0, len(states),initial_infected):
		states[sample] = 1

	# compartiments for each day
	sir = np.zeros((period, 3))

	# simulations for each time step
	for i in range(0, period):
		for x in states:
			sir[i][int(x)] += 1
		states = sir_iteration(network, states, beta, delta)
	return sir


def sir_iteration_recovery_rate(network, states, beta, delta):
	for node in network.nodes():
		if states[node-1] == 0:
			neighbors = nx.all_neighbors(network, node)
			for neighbor in neighbors:
				if states[neighbor-1] == 1:
					infection_prob = probability_of_infection(beta, network[node][neighbor]['weight'])
					states[node-1] = np.random.choice([0, 1], p=[1 - infection_prob, infection_prob])
					if states[node-1] == 1:
						break

		elif states[node-1] == 1:
			states[node-1] = np.random.choice([1, 2], p=[1 - delta, delta])

	return states

def plot_sir_evolution(sir):
	susceptible_percentage = list(map(lambda x: x/788, sir[:, 0]))
	infected_percentage = list(map(lambda x: x/788, sir[:, 1]))
	recovered_percentage = list(map(lambda x: x/788, sir[:, 2]))
	days = [i+1 for i in range(0, len(sir))]
	susceptibles, = plt.plot(days, susceptible_percentage, color='r', label='Susceptible Percentage')
	infected, = plt.plot(days, infected_percentage, color='b', label='Infected Percentage')
	recovered, = plt.plot(days, recovered_percentage, color='g', label='Recovered Percentage')
	plt.xlabel("Number of Days")
	plt.ylabel("Percentage")
	plt.title("Evolution of Influenza")
	plt.legend(handles=[susceptibles, infected, recovered])
	plt.show()

# FIXME
def plot_reproductive_number(network, infected_percentage, days_to_recovery, period):
	betas = np.arange(0.0, 0.0003, 0.00001)
	infected_fraction = [0]*len(betas)
	for i in range(0, len(betas)):
		infected_fraction[i] = sir_model_deterministic_recovery(network, infected_percentage, betas[i], days_to_recovery, period)\
								[period-1][2]/788

	plt.plot(betas, infected_fraction, color='r', label='Susceptible Percentage')
	plt.xlabel("Different Transmission Forces")
	plt.ylabel("Fraction of Infected")
	plt.title("Transmission Forces")
	plt.show()
	
def run():
	network = load_network()
	sir = sir_model_deterministic_recovery(network, 0.005, 0.6, 0.99, 0.0002, 5, 30)
	plot_sir_evolution(sir)
	#plot_reproductive_number(network, 0.005, 5, period=30)

if __name__ == '__main__':
    run()