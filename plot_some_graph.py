import matplotlib
matplotlib.use("pdf")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import click
import scipy as sp
import os

@click.command()
@click.option("-net_name", default = "case9_sm_net_0")
@click.option("-case", default = "case9")
@click.option("-model", default = "sm")

def main(net_name, case, model):

	net_file = "Example_Cases/{}_{}_net_.txt".format(case, model)

	created_file = "Networks/{}_.txt".format(net_name)

	os.system("cp {} {}".format(net_file, created_file))

	net_name = net_file.replace("Example_Cases/", "")
	net_name = net_name.replace("_.txt", "")
	delt_d = 0.5

	lines = [line.rstrip('\n') for line in open(net_file,"r")]


	lin_indx = 0
	for line in lines:
		esin_lin = line.split(" ")
		if (lin_indx == 0):
			N = int(esin_lin[0])
			interacts = int(esin_lin[1])
			K = np.zeros((N,N))
			P = np.zeros((N,1))
			P_disturbed = np.zeros(N)
			alf = np.zeros(N)

		elif ((lin_indx > 1) and (lin_indx < interacts + 2)):
			ni = int(esin_lin[0])
			nj = int(esin_lin[1])
			K[ni][nj] = float(esin_lin[2])

		elif ((lin_indx > interacts + 2) and (lin_indx < interacts + N + 3)):
			ni = int(esin_lin[0])
			P[ni] = float(esin_lin[1])
			# P_disturbed[ni] = float(esin_lin[2])


		lin_indx = lin_indx + 1


	IM_Grapho = nx.from_numpy_matrix(K)

	fr = plt.figure(figsize=(8,8))
	ax1 = fr.add_subplot(111)
	big_gen_list = list()
	small_gen_list = list()
	consumer_list = list()

	P = np.array(P)

	big_power = np.max(P[:])

	for a_node in range(len(P)):
		if (P[a_node] < 0.0):
			consumer_list.append(a_node)
		elif ((P[a_node] == big_power) and (delt_d < 1.0)):
			big_gen_list.append(a_node)
		else:
			small_gen_list.append(a_node)

	
	# print(big_power)
	# print(big_gen_list, '\n')
	# print(small_gen_list, '\n')
	# print(consumer_list)


	pos=nx.spring_layout(IM_Grapho)
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=big_gen_list, node_color='crimson', node_size=100, alpha=0.9, label = "Big Generators")
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=small_gen_list, node_color='yellowgreen', node_size=70, alpha=0.9, label = "Small Generators")
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=consumer_list, node_color='indigo', node_size=50, alpha=0.9, label = "Consumers")
	plt.legend(loc="best", scatterpoints=1)
	nx.draw_networkx_edges(IM_Grapho, pos, width=1.0,alpha=0.5)
	ax1.set_xticklabels('')
	ax1.set_yticklabels('')
	ax1.tick_params(axis='both', which='both', length = 0, bottom=False, top=False, labelbottom=False)
	plt.tight_layout()
	fr.savefig("Images/" + net_name + "_.pdf", bbox_inches='tight')
	plt.close()


if __name__ == '__main__':
	main()