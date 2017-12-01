import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys

from build_network import load_network

sys.path.append('..')

def xor(a, b):
    return (a and not b) or (not a and b)

class SIR:

    #	@brief:
    #			Constructor of the simulation
    def __init__(self, network):
        self.network = network
        self.N = len(network.nodes())
        self.nodes = range(self.N)
        self.node_states = np.zeros((self.N, 2))

    #	@brief:
    #			Function to reset the nodes with state Susceptible (0). Required to run another simulation with the same object.
    def reset_node_states(self):
        self.node_states = np.zeros((self.N, 2))

    #	@brief:
    #			Function to initialize the nodes of the network with the corresponding states.
    def initialize_node_states(self, infected_percentage, vaccinated_percentage, vaccine_effectiveness, vaccination_strategy):

        self.reset_node_states()

        if vaccination_strategy == "hubs":
            self.vaccinate_hubs(vaccinated_percentage, vaccine_effectiveness)
        elif vaccination_strategy == "random":
            self.vaccinate_randomly(vaccinated_percentage, vaccine_effectiveness)
        elif vaccination_strategy == "largest_neighbours":
            self.vaccinate_largest_neighbours(vaccinated_percentage, vaccine_effectiveness)
        else:
            raise AssertionError("Invalid Vaccination Strategy")
        self.infect_randomly(infected_percentage)

    #	@brief:
    #			Function that implements the vaccination strategy that randomly vaccinates a percentage of the nodes.
    def vaccinate_randomly(self, vaccinated_percentage, vaccine_effectiveness):
        number_of_vaccinated_nodes = round(self.N * vaccinated_percentage)
        count = 0
        while count < number_of_vaccinated_nodes:
            node = np.random.random_integers(self.N)-1
            if not self.node_is_recovered(node):
                self.vaccinate(node) if np.random.choice([1, 0], p=[vaccine_effectiveness, 1 - vaccine_effectiveness]) == 1 else 0
                count+=1
    
    #	@brief:
    #			Function that implements the vaccination strategy that vaccinates a certain percentage of the hubs in the network.
    def vaccinate_hubs(self, vaccinated_percentage, vaccine_effectiveness):
        degrees = nx.degree(self.network, weight='weight')
        sorted_degrees = sorted(degrees, key=lambda item: (item[1], item[0]))
        i = 0
        while (i / self.N < vaccinated_percentage):
            node_idx = sorted_degrees[-i][0] - 1
            self.vaccinate(node_idx) if np.random.choice([1, 0], p=[vaccine_effectiveness, 1 - vaccine_effectiveness]) == 1 else 0
            i += 1

    #	@brief:
    #			Function that returns for the node, its largest neighbour by weight
    def find_largest_neighbour(self, idx):

        nx_node_idx = idx + 1
        neighbours = nx.neighbors(self.network, nx_node_idx)

        largest = -sys.maxsize
        largest_neighbour = -1

        for node in neighbours:
            node_weight = self.network[nx_node_idx][node]['weight']
            #node_degree = self.network.degree(node, weight="weight")
            if node_weight >= largest:
                largest = node_weight
                largest_neighbour = node-1

        return largest_neighbour

    #	@brief:
    #			Function that implements the vaccination strategy that vaccinates the largest neighbours of a certain percentage of the hubs in the network.
    def vaccinate_largest_neighbours(self, n_percentage, vaccine_effectiveness):
        n = round(self.N * n_percentage)
        m = []

        nodes = []
        count = 0
        while count < n:
            node = np.random.random_integers(self.N)-1
            if node not in nodes:
                nodes.append(node)
                count += 1

        for node in nodes:
            largest_neighbour = self.find_largest_neighbour(node)
            if largest_neighbour not in m:
                m.append(largest_neighbour)

        for node in m:
            self.vaccinate(node) if np.random.choice([1, 0], p=[vaccine_effectiveness, 1 - vaccine_effectiveness]) == 1 else 0

        print (len(m))

    #	@brief:
    #			Function that randomly infects a percentage of the nodes in the network.
    def infect_randomly(self, infected_percentage):
        initially_infected = round(self.N * infected_percentage) \
            if round(self.N * infected_percentage) > 0 \
            else 1
        count = 0
        while count < initially_infected:
            node = np.random.random_integers(self.N)-1
            if self.node_is_susceptible(node):
                self.infect(node)
                count+=1

    #	@brief:
    #			Function to check if a certain node idx is Susceptible.
    def node_is_susceptible(self, idx):
        return self.node_states[idx][0] == 0

    #	@brief:
    #			Function that will check each neighbor of a certain node and if the neighbor is infected the node will became infected
    #			with a certain probability.
    def exposure(self, node, beta, new_states):
        node_neighbors = nx.all_neighbors(self.network, node + 1)
        for neighbor_idx, node_neighbor in enumerate(node_neighbors):
            if self.node_is_infected(neighbor_idx):
                probability_of_infection = self.probability_of_infection(beta, self.network[node + 1][node_neighbor][ 'weight'])
                new_states[node][0] = np.random.choice([0, 1], p = [1 - probability_of_infection, probability_of_infection])
                if self.node_is_infected(node):
                    new_states[node][1] = 0 # FIXME 0 or 1?
                    break
        return new_states

    #	@brief:
    #			Function that will compute the probability of an infection due to a contact with a certain weight for a certain beta.
    def probability_of_infection(self, beta, weight):
        return 1 - (1-beta)**weight  # prob. of being infected is the 1 - prob. of being healthy after k contacts

    #	@brief:
    #			Function that will change the state of a node from Susceptible (0) to Infected (1).
    def infect(self, node):
        self.node_states[node][0] = 1
        self.node_states[node][1] = 0

    #	@brief:
    #			Function to check if a certain node idx is Infected.
    def node_is_infected(self, idx):
        return self.node_states[idx][0] == 1

    #	@brief:
    #			Function to increment the number of days nodes are is in a certain State.
    def increment_status_days(self, node, states):
        states[node][1] += 1

    #	@brief:
    #			Function to check the status of the infection and recovers the node according to the recovery strategy.
    def recovery(self, node, recovery_strategy, new_states):
        if recovery_strategy(node):
            new_states[node][0] = 2
            new_states[node][1] = 0

    #	@brief:
    #			Function that checks if the node is infected for more than the normal days to recover and returns if the node should
    #			recover.
    def check_infection_threshold(self, node, days_to_recovery):
        return self.node_is_infected(node) and self.node_states[node][1] >= days_to_recovery

    #	@brief:
    #			Function that checks if the node infected is in the period of recuperation and with a given distribution (cumulative normal
    #			distribution the node recovers.
    def check_infection_range(self, node, days_to_recovery):
        from scipy.stats import norm
        probability_recovery = norm.cdf(self.node_states[node][1], loc=(days_to_recovery[0] + days_to_recovery[1])/2)
        return self.check_recovery_probability(node, probability_recovery)

    #	@brief:
    #			Function that checks with a certain probability if the node recovers.
    def check_recovery_probability(self, node, delta):
        return np.random.choice([1, 2], p=[1 - delta, delta]) == 2

    #	@brief:
    #			Function that vaccinates a certain node idx changing his state to recovered.
    def vaccinate(self, node):
        self.node_states[node][0] = 2
        self.node_states[node][1] = 0

    #	@brief:
    #			Function that checks if a certain node is recovered.
    def node_is_recovered(self, idx):
        return self.node_states[idx][0] == 2

    #	@brief:
    #			Function that runs the simulation.
    def run_simulation(self, iterations, infected_percentage, vaccinated_percentage, vaccine_effectiveness, vaccination_strategy, beta, delta=None, recovery_days=None):

        assert xor(delta==None, recovery_days==None), "Run with either delta or recovery days, but not with both at the same time"
        if delta != None:
            recovery_strategy = lambda node: self.check_recovery_probability(node, delta)
        elif type(recovery_days) != int:
            recovery_strategy = lambda node: self.check_infection_range(node, recovery_days)
        else:
            recovery_strategy = lambda node: self.check_infection_treshold(node, recovery_days)

        self.initialize_node_states(infected_percentage, vaccinated_percentage, vaccine_effectiveness, vaccination_strategy)
        simulation = np.zeros((iterations, 3))

        for t in range(iterations):
            for node_state in self.node_states: # Cycle that increments the number of individuals in a certain state for that time step.
                simulation[t][int(node_state[0])] += 1

            self.single_simulation_step(beta, recovery_strategy)

        return simulation

    def single_simulation_step(self, beta, recovery_strategy):
        new_states = np.array(self.node_states, copy=True)
        for node in self.nodes:
            self.increment_status_days(node, new_states)
            if self.node_is_susceptible(node):
                self.exposure(node, beta, new_states)
            elif self.node_is_infected(node):
                self.recovery(node, recovery_strategy, new_states)
        self.node_states = new_states


def plot_simulation(simulation):
    susceptible_percentage = list(map(lambda x: (x / 788)*100, simulation[:, 0]))
    infected_percentage = list(map(lambda x: (x / 788)*100, simulation[:, 1]))
    recovered_percentage = list(map(lambda x: (x / 788)*100, simulation[:, 2]))
    days = [i + 1 for i in range(0, len(simulation))]
    susceptibles, = plt.plot(days, susceptible_percentage, color='r', label='Susceptible Percentage')
    infected, = plt.plot(days, infected_percentage, color='b', label='Infected Percentage')
    recovered, = plt.plot(days, recovered_percentage, color='g', label='Recovered Percentage')
    plt.xlabel("Number of Days")
    plt.ylabel("Percentage")
    plt.title("Evolution of Influenza")
    plt.legend(handles=[susceptibles, infected, recovered])
    plt.show()


def main(args):

    if len(args) != 8:
        print('usage: '
            'sir.py '
            '<contact infection rate [0, 1]>'
            '<iterations> '
            '<number of infected individuals [0, 789]> '
            '<number of vaccinated individuals [0, 789]> '
            '<vaccine effectiveness [0, 1]> '
            '<vaccination strategy {random, hubs, largest_neighbours}> '
            '<minimum recovery days> '
            '<maximum recovery days> ')
        iterations = 30
        infected = 8
        vaccinated = 50
        vaccine_effectiveness = 0.95
        strategy = "largest_neighbours"
        recovery_days = (3, 9)
        beta = 0.003
    else:
        beta = float(args[0])
        iterations = int(args[1])
        infected = int(args[2])
        vaccinated = int(args[3])
        vaccine_effectiveness = float(args[4])
        strategy = args[5]
        recovery_days = (int(args[6]), int(args[7]))

    network = load_network()
    sir_system = SIR(network)

    simulation1 = sir_system.run_simulation(iterations,
        infected_percentage = infected / 789,
        vaccinated_percentage = vaccinated / 789,
        vaccine_effectiveness = vaccine_effectiveness,
        vaccination_strategy = strategy,
        recovery_days = recovery_days,
        beta=beta)

    plot_simulation(simulation1)
    print(simulation1)
    
    print("percentage of population infected over 30 days: %f" % ((simulation1[29][2] + simulation1[29][1] - 8 - simulation1[0][2])/788))
    print("number of infections: %d" % ((simulation1[29][2] + simulation1[29][1] - 8 - simulation1[0][2])))

if __name__ == '__main__':
    main(sys.argv[1:])