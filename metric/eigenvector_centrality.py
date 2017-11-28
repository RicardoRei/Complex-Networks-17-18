import numpy as np
import networkx as nx
import sys
sys.path.append('..')
from build_network import load_network, get_roles, load_nodes

network = load_network()
network.remove_node(548)  # este gajo faltou as aulas...

def run():

	'''
	Calculates the eigenvector centrality for weighted and not weighted newtork
	'''

	eigenv_centr_not_w = nx.eigenvector_centrality(network)
	eigenv_centr_w = nx.eigenvector_centrality(network, weight='weight')

	print("Eigenvector Centrality not weighted: ", eigenv_centr_not_w)
	print("Eigenvector Centrality weighted: ", eigenv_centr_w)

	'''
	Finds the node with the maximum eigenvector centrality for weighted and not weighted netowrk
	'''

	max_evc_not_w = max(eigenv_centr_not_w.values())
	max_evc_w = max(eigenv_centr_w.values())

	id_max_evc_not_w = list(eigenv_centr_not_w.keys())[list(eigenv_centr_not_w.values()).index(max_evc_not_w)]
	id_max_evc_w = list(eigenv_centr_w.keys())[list(eigenv_centr_w.values()).index(max_evc_w)]

	role_max_evc_not_w = load_nodes()[id_max_evc_not_w-1][1]
	role_max_evc_w = load_nodes()[id_max_evc_w-1][1]

	print("Maximum eigenvector centrality not weighted: %0.4f, Id: %d and Role: %s" % (max_evc_not_w, id_max_evc_not_w, role_max_evc_not_w))
	print("Maximum eigenvector centrality weighted: %0.4f, Id: %d and Role: %s" % (max_evc_w, id_max_evc_w, role_max_evc_w))

	'''
	Calculates the Average of Teachers' and Students' Eigenvector Centrality for weighted and not weighted netowrk
	'''

	roles = get_roles()
	evc_teachers_not_w, evc_students_not_w, evc_staff_not_w, evc_other_not_w = [], [], [], []

	for i in roles['teacher']:
		evc_teachers_not_w.append(eigenv_centr_not_w[i])
	for j in roles['student']:
		evc_students_not_w.append(eigenv_centr_not_w[j])
	for k in roles['staff']:
		evc_staff_not_w.append(eigenv_centr_not_w[k])
	for l in roles['other']:
		evc_other_not_w.append(eigenv_centr_not_w[l])

	avg_evc_teachers_not_w = np.mean(evc_teachers_not_w)
	avg_evc_students_not_w = np.mean(evc_students_not_w)
	avg_evc_staff_not_w = np.mean(evc_staff_not_w)
	avg_evc_other_not_w = np.mean(evc_other_not_w)

	print("Average of Teachers' Eigenvector Centrality not weighted: ", avg_evc_teachers_not_w)
	print("Average of Students' Eigenvector Centrality not weighted: ", avg_evc_students_not_w)
	print("Average of Staff's Eigenvector Centrality not weighted: ", avg_evc_staff_not_w)
	print("Average of Others' Eigenvector Centrality not weighted: ", avg_evc_other_not_w)

	evc_teachers_w, evc_students_w, evc_staff_w, evc_other_w = [], [], [], []

	for i in roles['teacher']:
		evc_teachers_w.append(eigenv_centr_w[i])
	for j in roles['student']:
		evc_students_w.append(eigenv_centr_w[j])
	for k in roles['staff']:
		evc_staff_w.append(eigenv_centr_w[k])
	for l in roles['other']:
		evc_other_w.append(eigenv_centr_w[l])

	avg_evc_teachers_w = np.mean(evc_teachers_w)
	avg_evc_students_w = np.mean(evc_students_w)
	avg_evc_staff_w = np.mean(evc_staff_w)
	avg_evc_other_w = np.mean(evc_other_w)

	print("Average of Teachers' Eigenvector Centrality weighted: ", avg_evc_teachers_w)
	print("Average of Students' Eigenvector Centrality weighted: ", avg_evc_students_w)
	print("Average of Staff's Eigenvector Centrality weighted: ", avg_evc_staff_w)
	print("Average of Others' Eigenvector Centrality weighted: ", avg_evc_other_w)


if __name__ == '__main__':
    run()

'''
Maximum eigenvector centrality not weighted: 0.0585, Id: 171 and Role: student
Maximum eigenvector centrality weighted: 0.1078, Id: 520 and Role: student
Average of Teachers' Eigenvector Centrality not weighted:  0.0190850898178
Average of Students' Eigenvector Centrality not weighted:  0.0371526980624
Average of Staff's Eigenvector Centrality not weighted:  0.0114173690018
Average of Others' Eigenvector Centrality not weighted:  0.0160573586792
Average of Teachers' Eigenvector Centrality weighted:  0.0188688195665
Average of Students' Eigenvector Centrality weighted:  0.0351413244388
Average of Staff's Eigenvector Centrality weighted:  0.00149859134629
Average of Others' Eigenvector Centrality weighted:  0.00452492018843
'''