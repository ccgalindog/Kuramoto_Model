import matplotlib
matplotlib.use("pdf")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import click

@click.command()
@click.option("-nodes", default = 2)
@click.option("-consumers", default = 1)
@click.option("-net_name", default = "quasireg_net_0")
@click.option("-powers", default = "a")
@click.option("-powers_disturb", default = "a")
@click.option("-alfas", default = "a")
@click.option("-delt_d", default = 0.5)

def main(nodes, consumers, net_name, powers, powers_disturb, alfas, delt_d):

	assert (((int(np.sqrt(consumers)))**2) == consumers ), "Consumers value must be a perfect square number!!"


	powers = powers.replace("[", "")
	powers = powers.replace("]", "")
	powers = powers.replace(",", "")
	powers = powers.split("_")
	powers = [float(a_power) for a_power in powers]

	powers_disturb = powers_disturb.replace("[", "")
	powers_disturb = powers_disturb.replace("]", "")
	powers_disturb = powers_disturb.replace(",", "")
	powers_disturb = powers_disturb.split("_")
	powers_disturb = [float(a_power) for a_power in powers_disturb]


	alfas = alfas.replace("[", "")
	alfas = alfas.replace("]", "")
	alfas = alfas.replace(",", "")
	alfas = alfas.split("_")
	alfas = [float(an_alfa) for an_alfa in alfas]


	N = nodes
	consr_lim = int(np.sqrt(consumers))
	K = np.zeros((N,N))
	Ax = np.zeros((consr_lim, consr_lim))

	k1 = N - consumers 

	for i in range(consr_lim):
		for j in range(consr_lim):
			Ax[i][j] = k1
			k1 = k1 + 1

	for i in range(consr_lim):
		for j in range(consr_lim):
			pos1 = Ax[i][j]
			if (i+1 < consr_lim):
				pos2 = Ax[i+1][j]
				K[int(pos1)][int(pos2)] = 1
				K[int(pos2)][int(pos1)] = 1
			if (i-1 > 0):
				pos3 = Ax[i-1][j]
				K[int(pos1)][int(pos3)] = 1
				K[int(pos3)][int(pos1)] = 1
			if (j+1 < consr_lim):
				pos4 = Ax[i][j+1]
				K[int(pos1)][int(pos4)] = 1
				K[int(pos4)][int(pos1)] = 1	
			if (j-1 > 0):
				pos5 = Ax[i][j-1]
				K[int(pos1)][int(pos5)] = 1
				K[int(pos5)][int(pos1)] = 1	

	already_gens = list()

	for hi in range(N - consumers):
		locate_gen = (np.random.randint(1, consr_lim-1), np.random.randint(1, consr_lim-1))
		
		while locate_gen in already_gens:
			locate_gen = (np.random.randint(1, consr_lim-1), np.random.randint(1, consr_lim-1))
		already_gens.append(locate_gen)

		pos1 = Ax[locate_gen[0]][locate_gen[1]]

		K[hi][int(pos1)] = 1
		K[int(pos1)][hi] = 1

		pos2 = Ax[locate_gen[0] + 1][locate_gen[1]]

		K[hi][int(pos2)] = 1
		K[int(pos2)][hi] = 1

		pos3 = Ax[locate_gen[0]][locate_gen[1]+1]

		K[hi][int(pos3)] = 1
		K[int(pos3)][hi] = 1

		pos4 = Ax[locate_gen[0]+1][locate_gen[1]+1]

		K[hi][int(pos4)] = 1
		K[int(pos4)][hi] = 1


	alf = list()
	P = list()
	link_list = list()

	for node_i in range(N):
		alf.append([node_i, alfas[node_i]])
		if powers[node_i] > 0:
			P.append([node_i, 1.0, powers[node_i], powers_disturb[node_i]])
		else:
			P.append([node_i, 0.0, powers[node_i], powers_disturb[node_i]])
		for node_j in range(N):
			if (K[node_i][node_j] == 1):
				link_list.append([node_i, node_j, 1])


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



	IM_Grapho = nx.from_numpy_matrix(K)

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
		elif ((P[a_node][2] == big_power) and (delt_d < 1.0)):
			big_gen_list.append(a_node)
		else:
			small_gen_list.append(a_node)

	pos=nx.spring_layout(IM_Grapho)
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=big_gen_list, node_color='crimson', node_size=100, alpha=0.9, label = "Big Generators")
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=small_gen_list, node_color='yellowgreen', node_size=70, alpha=0.9, label = "Small Generators")
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=consumer_list, node_color='indigo', node_size=50, alpha=0.9, label = "Consumers")
	nx.draw_networkx_edges(IM_Grapho, pos, width=1.0,alpha=0.5)
	plt.legend(loc="best", scatterpoints=1)
	ax1.set_xticklabels('')
	ax1.set_yticklabels('')
	ax1.tick_params(axis='both', which='both', length = 0, bottom=False, top=False, labelbottom=False)
	plt.tight_layout()
	fr.savefig("Images/" + net_name + "_.pdf", bbox_inches='tight')
	plt.close()


if __name__ == '__main__':
	main()