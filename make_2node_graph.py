import matplotlib
matplotlib.use("pdf")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import click
import scipy as sp

def build_graph(net_name, powers, powers_disturb, alfas, to_plot):
'''
Create a Wattss-Strogatz graph where nodes are located initially in a ring connected to some amount neighbors and then connections are relinked with some probability pth.
INPUT:
net_name: <String> - Name of the network. 
powers: <List> (2 elements) - Default power at each node.
powers_disturb: <List> (2 elements) - Power at each node after a disturbance.
alfas: <List> (2 elements) - Damping at each node.
to_plot: <Boolean> - If want to plot the output graph.
OUTPUT:
A text file at Networks folder.
'''
#	powers = powers.replace("[", "")
#	powers = powers.replace("]", "")
#	powers = powers.replace(",", "")
#	powers = powers.split("_")
#	powers = [float(a_power) for a_power in powers]

#	powers_disturb = powers_disturb.replace("[", "")
#	powers_disturb = powers_disturb.replace("]", "")
#	powers_disturb = powers_disturb.replace(",", "")
#	powers_disturb = powers_disturb.split("_")
#	powers_disturb = [float(a_power) for a_power in powers_disturb]

#	alfas = alfas.replace("[", "")
#	alfas = alfas.replace("]", "")
#	alfas = alfas.replace(",", "")
#	alfas = alfas.split("_")
#	alfas = [float(an_alfa) for an_alfa in alfas]

	N = 2
	K = np.array([[0, 1], [1, 0]])

	IM_Grapho = nx.from_numpy_matrix(K)


	alf = list()
	P = list()
	link_list = list()

	for node_i in range(N):
		alf.append([node_i, alfas[node_i]])
		if powers[node_i] > 0:
			P.append([node_i, 1.0, powers[node_i], powers_disturb[node_i]])
		else:
			P.append([node_i, 0.0, powers[node_i], powers_disturb[node_i]])

	link_list = [[0, 1, 1], [1, 0, 1]]


	out_file = "Networks/" + net_name + "_.txt"

	file = open(out_file,"w") 
	file.write("{} {} \n".format(N, len(link_list))) # Nodes Links
	file.write("K \n")
	for i in range(len(link_list)):
		file.write("{} {} {} \n".format(link_list[i][0], link_list[i][1], link_list[i][2])) 
	file.write("P \n")
	for i in range(len(P)):
		file.write("{} {} {} {} \n".format(P[i][0], P[i][1], P[i][2], P[i][3])) 
	file.write("Alfa \n")
	for i in range(len(alf)):
		file.write("{} {} \n".format(alf[i][0], alf[i][1])) 	
	 
	file.close() 

	if (to_plot):
		fr = plt.figure(figsize=(8,8))
		ax1 = fr.add_subplot(111)
		big_gen_list = list()
		small_gen_list = list()
		consumer_list = list()

		P = np.array(P)
		big_power = np.max(P[:,2])

		for a_node in range(len(P)):
			if (P[a_node][2] < 0.0):
				consumer_list.append(a_node)
			else:
				big_gen_list.append(a_node)

		pos=nx.circular_layout(IM_Grapho)
		nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=big_gen_list, node_color='crimson', node_size=100, alpha=0.9, label = "Generator")
		nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=consumer_list, node_color='indigo', node_size=50, alpha=0.9, label = "Consumer")
		plt.legend(loc="best", scatterpoints=1)
		nx.draw_networkx_edges(IM_Grapho, pos, width=1.0,alpha=0.5)
		ax1.set_xticklabels('')
		ax1.set_yticklabels('')
		ax1.tick_params(axis='both', which='both', length = 0, bottom=False, top=False, labelbottom=False)
		plt.tight_layout()
		fr.savefig("Images/" + net_name + "_.pdf", bbox_inches='tight')
		plt.close()


#if __name__ == '__main__':
#	main()
