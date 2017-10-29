import numpy as np
import networkx as nx
from build_network import load_network, get_roles

network = load_network()

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

	print("Maximum Eigenvector Centrality not weighted: ", max_evc_not_w)
	print("Maximum Eigenvector Centrality weighted: ", max_evc_w)

	print("ID of Maximum Eigenvector Centrality not weighted: ", id_max_evc_not_w)
	print("ID of Maximum Eigenvector Centrality weighted: ", id_max_evc_w)

	'''
	Calculates the Average of Teachers' and Students' Eigenvector Centrality for weighted and not weighted netowrk
	'''

	roles = get_roles()
	evc_teachers_not_w, evc_students_not_w = [], []

	for i in roles['teacher']:
		evc_teachers_not_w.append(eigenv_centr_not_w[i])
	for j in roles['student']:
		evc_students_not_w.append(eigenv_centr_not_w[j])

	avg_evc_teachers_not_w = np.mean(evc_teachers_not_w)
	avg_evc_students_not_w = np.mean(evc_students_not_w)

	print("Average of Teachers' Eigenvector Centrality not weighted: ", avg_evc_teachers_not_w)
	print("Average of Students' Eigenvector Centrality not weighted: ", avg_evc_students_not_w)

	evc_teachers_w, evc_students_w = [], []

	for i in roles['teacher']:
		evc_teachers_w.append(eigenv_centr_w[i])
	for j in roles['student']:
		evc_students_w.append(eigenv_centr_w[j])

	avg_evc_teachers_w = np.mean(evc_teachers_w)
	avg_evc_students_w = np.mean(evc_students_w)

	print("Average of Teachers' Eigenvector Centrality weighted: ", avg_evc_teachers_w)
	print("Average of Students' Eigenvector Centrality weighted: ", avg_evc_students_w)


if __name__ == '__main__':
    run()

'''
Maximum Eigenvector Centrality not weighted:  0.05848065844832825
Maximum Eigenvector Centrality weighted:  0.10778672661500492
ID of Maximum Eigenvector Centrality not weighted:  171
ID of Maximum Eigenvector Centrality weighted:  520
Average of Teachers' Eigenvector Centrality not weighted:  0.0190850898178
Average of Students' Eigenvector Centrality not weighted:  0.0371526980624
Average of Teachers' Eigenvector Centrality weighted:  0.0188688195665
Average of Students' Eigenvector Centrality weighted:  0.0351413244388
'''