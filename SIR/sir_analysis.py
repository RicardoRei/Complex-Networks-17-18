import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from build_network import load_network

def xor(a, b):
    return (a and not b) or (not a and b)

class SIR:

    def __init__(self, network):
        self.network = network
        self.N = len(network.nodes())
        self.nodes = range(self.N)
        self.node_states = np.zeros((self.N, 2))

    def reset_node_states(self):
        self.node_states = np.zeros((self.N, 2))
    def initialize_node_states(self, infected_percentage, vaccinated_percentage, vaccine_effectiveness, vaccinate_hubs):
        self.reset_node_states()
        if vaccinate_hubs:
            self.vaccinate_hubs(vaccinated_percentage, vaccine_effectiveness)
        else:
            self.vaccinate_randomly(vaccinated_percentage, vaccine_effectiveness)
        self.infect_randomly(infected_percentage)
    def vaccinate_randomly(self, vaccinated_percentage, vaccine_effectiveness):
        number_of_vaccinated_nodes = round(self.N * vaccinated_percentage) \
            if round(self.N * vaccinated_percentage) > 0 \
            else 1
        for node_idx in np.random.randint(0, self.N, number_of_vaccinated_nodes):
            self.recover_node(node_idx) if np.random.choice([1, 0], p=[vaccine_effectiveness, 1 - vaccine_effectiveness]) == 1 else 0
    def vaccinate_hubs(self, vaccinated_percentage, vaccine_effectiveness):
        centralities = nx.degree_centrality(self.network)
        sorted_centralities = sorted(centralities.items(), key=lambda item: (item[1], item[0]))
        i = 0
        while (i / self.N < vaccinated_percentage):
            node_idx = sorted_centralities[-i][0] - 1
            self.recover_node(node_idx) if np.random.choice([1, 0], p=[vaccine_effectiveness, 1 - vaccine_effectiveness]) == 1 else 0
            i += 1
    def infect_randomly(self, infected_percentage):
        initially_infected = round(self.N * infected_percentage) \
            if round(self.N * infected_percentage) > 0 \
            else 1
        for node_idx in np.random.randint(0, self.N, initially_infected):
            if self.node_is_susceptible(node_idx):
                self.infect_node(node_idx)

    def node_is_susceptible(self, idx):
        return self.node_states[idx][0] == 0
    def check_node_exposure(self, node, beta):
        node_neighbors = nx.all_neighbors(self.network, node + 1)
        for neighbor_idx, node_neighbor in enumerate(node_neighbors):
            if self.node_is_infected(neighbor_idx):
                probability_of_infection = self.probability_of_infection(beta, self.network[node + 1][node_neighbor][ 'weight'])
                self.node_states[node][0] = np.random.choice([0, 1], p = [1 - probability_of_infection, probability_of_infection])
                if self.node_is_infected(node):
                    break
    def probability_of_infection(self, beta, weight):
        probability_staying_susceptible = 1 - beta
        for i in range(1, weight):
            probability_staying_susceptible *= 1 - beta  # 1 - prob. get infected
        return 1 - probability_staying_susceptible  # prob. of being infected is the 1 - prob. of being healthy after k contacts

    def infect_node(self, idx):
        self.node_states[idx][0] = 1
        self.node_states[idx][1] = 1
    def node_is_infected(self, idx):
        return self.node_states[idx][0] == 1
    def another_day_infected(self, node):
        self.node_states[node][1] += 1
    def check_infection_status(self, node, recovery_strategy):
        if recovery_strategy(node):
            self.recover_node(node)
        else:
            self.another_day_infected(node)

    def node_has_rested(self, node, days_to_recovery):
        return self.node_is_infected(node) and self.node_states[node][1] >= days_to_recovery
    def node_recovers(self, node, delta):
        return np.random.choice([1, 2], p=[1 - delta, delta]) == 2

    def recover_node(self, idx):
        self.node_states[idx][0] = 2
    def node_is_recovered(self, idx):
        return self.node_states[idx][0] == 2

    def run_simulation(self, iterations, infected_percentage, vaccinated_percentage, vaccine_effectiveness, vaccinate_hubs, beta, delta=None, recovery_days=None):

        assert xor(delta==None, recovery_days==None), "Run with either delta or recovery days, but not with both at the same time"
        if delta != None:
            recovery_strategy = lambda node: self.node_recovers(node, delta)
        else:
            recovery_strategy = lambda node: self.node_has_rested(node, recovery_days)

        self.initialize_node_states(infected_percentage, vaccinated_percentage, vaccine_effectiveness, vaccinate_hubs)
        simulation = np.zeros((iterations, 3))

        for t in range(iterations):

            # TODO - O que fazem estas duas linhas? Criar um metodo com o nome da operacao e coloca-las as duas la dentro
            for node_state in self.node_states:
                simulation[t][int(node_state[0])] += 1
            self.single_simulation_step(beta, recovery_strategy)

        return simulation
    def single_simulation_step(self, beta, recovery_strategy):
        for node in self.nodes:
            if self.node_is_susceptible(node):
                self.check_node_exposure(node, beta)
            elif self.node_is_infected(node):
                self.check_infection_status(node, recovery_strategy)


def plot_simulation(simulation):
    susceptible_percentage = list(map(lambda x: x / 788, simulation[:, 0]))
    infected_percentage = list(map(lambda x: x / 788, simulation[:, 1]))
    recovered_percentage = list(map(lambda x: x / 788, simulation[:, 2]))
    days = [i + 1 for i in range(0, len(simulation))]
    susceptibles, = plt.plot(days, susceptible_percentage, color='r', label='Susceptible Percentage')
    infected, = plt.plot(days, infected_percentage, color='b', label='Infected Percentage')
    recovered, = plt.plot(days, recovered_percentage, color='g', label='Recovered Percentage')
    plt.xlabel("Number of Days")
    plt.ylabel("Percentage")
    plt.title("Evolution of Influenza")
    plt.legend(handles=[susceptibles, infected, recovered])
    plt.show()

def run():

    network = load_network()
    sir_system = SIR(network)

    # Should return error since the simulation is either ran with delta or recovery time
    #sir_system.run_simulation(iterations=30, infected_percentage=0.005, vaccinated_percentage=0.6, vaccine_effectiveness=0.99, vaccinate_hubs=False, beta=0.0002, delta=0.02, recovery_days=30)

    # Should return error since the simulation is either ran with delta or recovery time
    #sir_system.run_simulation(iterations=30, infected_percentage=0.005, vaccinated_percentage=0.6, vaccine_effectiveness=0.99, vaccinate_hubs=False, beta=0.0002)

    simulation1 = sir_system.run_simulation(iterations=30, infected_percentage=0.005, vaccinated_percentage=0.6, vaccine_effectiveness=0.99, vaccinate_hubs=False, beta=0.0002, recovery_days=30)
    simulation2 = sir_system.run_simulation(iterations=30, infected_percentage=0.005, vaccinated_percentage=0.6, vaccine_effectiveness=0.99, vaccinate_hubs=False, beta=0.0002, delta=0.02)

    simulation3 = sir_system.run_simulation(iterations=30, infected_percentage=0.005, vaccinated_percentage=0.6, vaccine_effectiveness=0.99, vaccinate_hubs=True, beta=0.0002,recovery_days=30)
    simulation4 = sir_system.run_simulation(iterations=30, infected_percentage=0.005, vaccinated_percentage=0.6, vaccine_effectiveness=0.99, vaccinate_hubs=True, beta=0.0002, delta=0.02)

    plot_simulation(simulation1)
    plot_simulation(simulation2)
    plot_simulation(simulation3)
    plot_simulation(simulation4)

if __name__ == '__main__':
    run()