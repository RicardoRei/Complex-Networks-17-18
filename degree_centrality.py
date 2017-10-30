import numpy as np
import networkx as nx
from build_network import load_network, load_nodes, get_roles

network = load_network()

def run():

	'''
	Calculates the degree centrality
	'''

	degree_centrality  = nx.degree_centrality(network)

	print(degree_centrality)

	'''
	Finds the nodes with the maximum and minimum degree centrality, their IDs and their roles
	'''

	max_degree_centrality = max(degree_centrality.values())
	min_degree_centrality = min(degree_centrality.values())

	id_max_dc = list(degree_centrality.keys())[list(degree_centrality.values()).index(max_degree_centrality)]
	id_min_dc = list(degree_centrality.keys())[list(degree_centrality.values()).index(min_degree_centrality)]

	role_max_dc = load_nodes()[id_max_dc-1][1]
	role_min_dc = load_nodes()[id_min_dc-1][1]

	print("Maximum degree centrality: %0.4f, Id: %d and Role: %s" % (max_degree_centrality, id_max_dc, role_max_dc))
	print("Minimum degree centrality: %0.4f, Id: %d and Role: %s" % (min_degree_centrality, id_min_dc, role_min_dc))

	'''
	Calculates the Average of Teachers' and Students' Degree Centrality
	'''

	roles = get_roles()
	dc_teachers, dc_students, dc_staff, dc_other = [], [], [], []

	for i in roles['teacher']:
		dc_teachers.append(degree_centrality[i])
	for j in roles['student']:
		dc_students.append(degree_centrality[j])
	for k in roles['staff']:
		dc_staff.append(degree_centrality[k])
	for l in roles['other']:
		dc_other.append(degree_centrality[l])

	avg_dc_teachers = np.mean(dc_teachers)
	avg_dc_students = np.mean(dc_students)
	avg_dc_staff = np.mean(dc_staff)
	avg_dc_other = np.mean(dc_other)

	print("Average of Teachers' Degree Centrality: ", avg_dc_teachers)
	print("Average of Students' Degree Centrality: ", avg_dc_students)
	print("Average of Staff's Degree Centrality: ", avg_dc_staff)
	print("Average of Others' Degree Centrality: ", avg_dc_other)


if __name__ == '__main__':
    run()


    '''
    Maximum degree centrality: 0.6696, Id: 171 and Role: student
	Minimum degree centrality: 0.0051, Id: 376 and Role: staff
	Average of Teachers' Degree Centrality:  0.223808114741
	Average of Students' Degree Centrality:  0.420312909202
	Average of Staff's Degree Centrality:  0.145339031997
	Average of Others' Degree Centrality:  0.195171537484
    '''