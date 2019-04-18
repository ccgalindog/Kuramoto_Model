import matplotlib
matplotlib.use("pdf")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import click
import scipy as sp
from pypower.api import ppoption, runpf, printpf, makeYbus
import pypsa
from pypower.idx_gen import PG, QG, GEN_BUS
from pypower.idx_bus import PD, QD, VM, VA, BUS_I
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import grid_cases as gridcase

# In this file you find the following functions:
# build_2node_graph
# build_quasiregular_graph
# build_random_graph
# build_smallworld_graph
# build_gridcase_graph
# SP_model
# EN_model
# SM_model

#####################################################################################################################



def build_2node_graph(net_name, powers, powers_disturb, alfas, to_plot, to_write):
	'''
	Creates a 2 node network.
	'''
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


	if to_write:

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

	return K	
		
##################################################



def build_quasiregular_graph(nodes, consumers, net_name, powers, powers_disturb, alfas, delt_d, to_plot, to_write):
	'''
	Create a graph where consumers are located in a square lattice and generators are located randomly and connected to 4 nearest neighbours.
	INPUT:
	nodes: <Int> - Total amount of nodes.
	consumers: <Int> - Amount of consumers. Must be a perfect square number and must be lower than 'nodes'. 
	net_name: <String> - Name of the network. 
	powers: <List> - Default power at each node.
	powers_disturb: <List> - Power at each node after a disturbance.
	alfas: <List> - Damping at each node.
	delt_d: <Double> - Fraction of generator nodes that are assigned as 'small generators'.
	to_plot: <Boolean> - If want to plot the output graph.
	OUTPUT:
	A text file at Networks folder.
	'''
	assert (((int(np.sqrt(consumers)))**2) == consumers ), "Consumers value must be a perfect square number!!"
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

	R_Matrix = 5*np.random.rand(K.shape[0], K.shape[1])
	R_Matrix = R_Matrix + R_Matrix.T


	K = np.multiply( K,  R_Matrix)


	for node_i in range(N):
		alf.append([node_i, alfas[node_i]])
		if powers[node_i] > 0:
			P.append([node_i, 1.0, powers[node_i], powers_disturb[node_i]])
		else:
			P.append([node_i, 0.0, powers[node_i], powers_disturb[node_i]])
		for node_j in range(N):
			if (K[node_i][node_j] != 0):
				link_list.append([node_i, node_j, K[node_i][node_j]])


	if to_write:

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

	return K	
				
#####################################################################################



def build_random_graph(nodes, m_degree, net_name, powers, powers_disturb, alfas, delt_d, to_plot, to_write):
	'''
	Create a graph where consumers and generators are located randomly and connected with a mean node degree.
	INPUT:
	nodes: <Int> - Total amount of nodes.
	m_degree: <Int> - Mean node degree desired. 
	net_name: <String> - Name of the network. 
	powers: <List> - Default power at each node.
	powers_disturb: <List> - Power at each node after a disturbance.
	alfas: <List> - Damping at each node.
	delt_d: <Double> - Fraction of generator nodes that are assigned as 'small generators'.
	to_plot: <Boolean> - If want to plot the output graph.
	OUTPUT:
	A text file at Networks folder.
	'''
	N = nodes
	connected = False
	m_degree = m_degree/2
	da_mean_deg = m_degree - 3
	while (connected == False):
		K = np.zeros((N,N))
		for i in range(N):
			for j in range(N):
				if i != j:
					u = np.random.rand()
					if (u < (m_degree/N)):
						K[i][j] = 1
						K[j][i] = 1

		IM_Grapho = nx.from_numpy_matrix(K)
		connected = nx.is_connected(IM_Grapho)
		da_mean_deg = np.mean(np.sum(K, axis=0))

	print("Mean degree of graph: ", np.mean(np.sum(K, axis=0)))

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


	if to_write:				
		out_file = "Networks/" + net_name + "_.txt"
		file = open(out_file,"w") 
		file.write("{} {} \n".format(N, len(link_list))) # Nodes Links
		file.write("K \n")
		for i in range(len(link_list)):
			file.write("{} {} {} \n".format(link_list[i][0], link_list[i][1], link_list[i][2])) 
		file.write("P \n")
		for i in range(len(P)):
			file.write("{} {} {} {} \n".format(P[i][0], P[i][1], P[i][2],  P[i][3])) 
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
			elif ((P[a_node][2] == big_power) and (delt_d < 1.0)):
				big_gen_list.append(a_node)
			else:
				small_gen_list.append(a_node)
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

	return K

####################################################################################################################



def build_smallworld_graph(nodes, neighbors, pth, net_name, powers, powers_disturb, alfas, delt_d, to_plot, to_write):
	'''
	Create a Wattss-Strogatz graph where nodes are located initially in a ring connected to some amount neighbors and then connections are relinked with some probability pth.
	INPUT:
	nodes: <Int> - Total amount of nodes.
	neighbors: <Int> - Amount of initial neighbors for each node in the ring. 
	pth: <Double> - Rewiring probability.
	net_name: <String> - Name of the network. 
	powers: <List> - Default power at each node.
	powers_disturb: <List> - Power at each node after a disturbance.
	alfas: <List> - Damping at each node.
	delt_d: <Double> - Fraction of generator nodes that are assigned as 'small generators'.
	to_plot: <Boolean> - If want to plot the output graph.
	OUTPUT:
	A text file at Networks folder.
	'''
	N = nodes
	IM_Grapho = nx.connected_watts_strogatz_graph(nodes, neighbors, pth)
	K = nx.adjacency_matrix(IM_Grapho)
	K = K.todense()
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
			if (K[node_i, node_j] == 1):
				link_list.append([node_i, node_j, 1])


	if to_write:
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
			elif ((P[a_node][2] == big_power) and (delt_d < 1.0)):
				big_gen_list.append(a_node)
			else:
				small_gen_list.append(a_node)
		pos=nx.circular_layout(IM_Grapho)
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

	return K
		
############################################################################################################



def build_colombian_graph(net_name, to_plot, to_write):
	'''
	Create a Wattss-Strogatz graph where nodes are located initially in a ring connected to some amount neighbors and then connections are relinked with some probability pth.
	INPUT:
	nodes: <Int> - Total amount of nodes.
	neighbors: <Int> - Amount of initial neighbors for each node in the ring. 
	pth: <Double> - Rewiring probability.
	net_name: <String> - Name of the network. 
	powers: <List> - Default power at each node.
	powers_disturb: <List> - Power at each node after a disturbance.
	alfas: <List> - Damping at each node.
	delt_d: <Double> - Fraction of generator nodes that are assigned as 'small generators'.
	to_plot: <Boolean> - If want to plot the output graph.
	OUTPUT:
	A text file at Networks folder.
	'''
	K = np.loadtxt('params_COL/K_Colombia_pu.txt')
	P = np.loadtxt('params_COL/P_Colombia_pu.txt')
	#Alf = np.abs( np.loadtxt('params_COL/alf_Colombia_pu.txt') )

	Alf = 0.1*np.ones( P.shape )

	IM_Grapho = nx.from_numpy_matrix(K)

	N = len(P)
	link_list = np.sum(K != 0)


	if to_write:
		out_file = "Networks/" + net_name + "_.txt"
		file = open(out_file,"w") 
		file.write("{} {} \n".format(N, link_list)) # Nodes Links
		file.write("K \n")
		for i in range(len(P)):
			for j in range(len(P)):
				if ( K[i][j] != 0 ):
					file.write("{} {} {} \n".format( i, j, K[i][j] ))

		file.write("P \n")
		for i in range(len(P)):
			if (P[i] > 0):
				file.write("{} {} {} {} \n".format(i, 1, P[i], P[i])) 
			else:
				file.write("{} {} {} {} \n".format(i, 0, P[i], P[i])) 

		file.write("Alfa \n")
		for i in range(len(Alf)):
			file.write("{} {} \n".format(i, abs(Alf[i]))) 	
		file.close() 


	if (to_plot):
		fr = plt.figure(figsize=(8,8))
		ax1 = fr.add_subplot(111)
		big_gen_list = list()
		small_gen_list = list()
		consumer_list = list()

		for a_node in range(len(P)):
			if (P[a_node] <= 0.0):
				consumer_list.append(a_node)
			else:
				big_gen_list.append(a_node)

		pos=nx.spring_layout(IM_Grapho)
		nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=big_gen_list, node_color='crimson', node_size=100, alpha=0.9, label = "Big Generators")
		nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=consumer_list, node_color='indigo', node_size=50, alpha=0.9, label = "Consumers")
		plt.legend(loc="best", scatterpoints=1)
		nx.draw_networkx_edges(IM_Grapho, pos, width=1.0,alpha=0.5)
		ax1.set_xticklabels('')
		ax1.set_yticklabels('')
		ax1.tick_params(axis='both', which='both', length = 0, bottom=False, top=False, labelbottom=False)
		plt.tight_layout()
		fr.savefig("Images/" + net_name + "_.pdf", bbox_inches='tight')
		plt.close()

	return K	




#############################################################################################


def kuramoto_weight(x0, P, K, Gamm):
	dot_theta = P + np.sum( K * np.sin( np.repeat(np.array([x0]).T, len(x0), axis=1) - np.repeat(np.array([x0]), len(x0), axis=0) + Gamm), axis=0 )
	v_sqrd = np.linalg.norm(dot_theta)
	return v_sqrd

###############################

def SM_model(mpc2, est_dyn, Y0):
	x_d = est_dyn["x_d"]
	baseMVA = mpc2[0]["baseMVA"]
	gtb = np.unique(mpc2[0]["gen"][:, GEN_BUS]) #Generator indices
	ngt = len(gtb) # Amount generators
	allbus = mpc2[0]["bus"][:,BUS_I]
	tb = mpc2[0]["gen"][:, GEN_BUS]
	ltb = np.logical_not(np.in1d(allbus,gtb))
	Pi = np.concatenate((mpc2[0]["gen"][:,PG], -mpc2[0]["bus"][ltb,PD])) / baseMVA
	Qi = np.concatenate((mpc2[0]["gen"][:,QG], -mpc2[0]["bus"][ltb,QD])) / baseMVA
	V = mpc2[0]["bus"][:,VM] 
	phi = (np.pi*mpc2[0]["bus"][:,VA])/180
	E = ((V + Qi*x_d/V) + 1j*(Pi*x_d/V))*np.exp(1j*phi)
	ltb = np.where(ltb)[0]
	Y0gl = Y0[gtb,:]
	Y0gl = Y0gl[:,ltb]
	Y0lg = Y0[ltb,:]
	Y0lg = Y0lg[:,gtb]
	Y0ll = Y0[ltb,:]
	Y0ll = Y0ll[:,ltb]
	Plg = mpc2[0]["bus"][gtb, PD]/baseMVA
	Qlg = mpc2[0]["bus"][gtb, QD]/baseMVA
	Vg = mpc2[0]["bus"][gtb, VM]
	hi = Y0[gtb,:]
	hi = hi[:,gtb]
	Y0ggt = hi + csr_matrix((Plg/(Vg**2) - 1j*(Qlg/(Vg**2)), (range(ngt), range(ngt))), shape=(ngt,ngt))
	# Redefine Y0, define Yd, and apply Kron reduction to generate Y_SM.
	Y0n_a = np.concatenate((Y0ggt.todense(), Y0gl.todense()), axis = 1) 
	Y0n_b = np.concatenate((Y0lg.todense(), Y0ll.todense()), axis = 1) 
	Y0n = np.concatenate((Y0n_a, Y0n_b), axis = 0) 
	n = np.shape(mpc2[0]["bus"])[0]
	Yd = csr_matrix((1/(1j*x_d), (range(n), range(n))), shape=(n,n))
	Ydinv = np.linalg.inv(Y0n + Yd.todense())
	Y_SM = Yd.todense() - Yd.todense()*Ydinv*Yd.todense()
	aE = abs(E)
	A = Pi - (aE**2) * np.real(np.diag(Y_SM))
	K = np.diag(aE) * np.abs(Y_SM) * np.diag(aE)


	Gamm = np.zeros( Y_SM.shape )

	for i in range(Gamm.shape[0]):
		for j in range(Gamm.shape[1]):
			if (np.real(Y_SM[i,j]) == 0.0):
				Gamm[i][j] = 0.0
			else:
				Gamm[i][j] = np.angle(Y_SM[i,j]) - np.pi/2

	for i in range(Gamm.shape[0]):
		for j in range(Gamm.shape[1]):
			if (np.abs(Gamm[i,j]) < 1e-8):
				Gamm[i,j] = 0.0
	Node_Type = np.zeros(len(A))
	Node_Type[gtb] = 1.0

	return A, K, Gamm, phi, Node_Type

################################

def EN_model(mpc2, est_dyn, Y0):
	x_d = est_dyn["x_d"]
	baseMVA = mpc2[0]["baseMVA"]
	gtb = np.unique(mpc2[0]["gen"][:, GEN_BUS]) #Generator indices
	ngt = len(gtb) # Amount generators
	allbus = mpc2[0]["bus"][:,BUS_I]
	tb = mpc2[0]["gen"][:, GEN_BUS]
	ltb = np.logical_not(np.in1d(allbus,gtb))
	Pi = (mpc2[0]["gen"][:,PG]) / baseMVA
	Qi = (mpc2[0]["gen"][:,QG]) / baseMVA
	V = mpc2[0]["bus"][tb,VM] 
	phi = (np.pi*mpc2[0]["bus"][tb,VA])/180
	E = ((V + Qi*x_d[gtb]/V) + 1j*(Pi*x_d[gtb]/V))*np.exp(1j*phi)
	ltb = np.where(ltb)[0]
	nl = len(ltb)
	Y0gl = Y0[gtb,:]
	Y0gl = Y0gl[:,ltb]
	Y0lg = Y0[ltb,:]
	Y0lg = Y0lg[:,gtb]
	n = np.shape(mpc2[0]["bus"])[0]
	Yd = csr_matrix((1/(1j*x_d[gtb]), (range(ngt), range(ngt))), shape=(ngt,ngt))
	Plg = mpc2[0]["bus"][gtb, PD]/baseMVA
	Qlg = mpc2[0]["bus"][gtb, QD]/baseMVA
	Vg = mpc2[0]["bus"][gtb, VM]
	hi = Y0[gtb,:]
	hi = hi[:,gtb]
	Y0ggt = hi + csr_matrix((Plg/(Vg**2) - 1j*(Qlg/(Vg**2)), (range(ngt), range(ngt))), shape=(ngt,ngt))
	Pll = mpc2[0]["bus"][ltb, PD]/baseMVA
	Qll = mpc2[0]["bus"][ltb, QD]/baseMVA
	Vl = mpc2[0]["bus"][ltb, VM]
	Y0ll = Y0[ltb,:]
	Y0ll = Y0ll[:,ltb]
	Y0llt = Y0ll + csr_matrix((Pll/(Vl**2) - 1j*(Qll/(Vl**2)), (range(nl), range(nl))), shape=(nl,nl))
	Y0nn = np.copy(Yd.todense())
	Y0nr = csr_matrix((-1/(1j*x_d[gtb]), (range(ngt), range(ngt))), shape=(ngt,n))
	Y0rn = Y0nr.T
	Y0rr_a = np.concatenate(( Y0ggt.todense() - np.diag((np.array(np.sum(Y0nr.todense()[:,0:ngt], axis=0)))[0]), Y0gl.todense() ), axis = 1) 
	Y0rr_b = np.concatenate(( Y0lg.todense(), Y0llt.todense() ), axis = 1) 
	Y0rr = np.concatenate((Y0rr_a, Y0rr_b), axis = 0)
	Y_EN = Y0nn - (Y0nr*np.linalg.inv(Y0rr))*Y0rn
	aE = abs(E)
	A = Pi - (aE**2) * np.real(np.diag(Y_EN))
	K = np.diag(aE) * np.abs(Y_EN) * np.diag(aE)
	
	Gamm = np.zeros( Y_EN.shape )

	for i in range(Gamm.shape[0]):
		for j in range(Gamm.shape[1]):
			if (np.real(Y_EN[i,j]) == 0.0):
				Gamm[i][j] = 0.0
			else:
				Gamm[i][j] = np.angle(Y_EN[i,j]) - np.pi/2


	for i in range(Gamm.shape[0]):
		for j in range(Gamm.shape[1]):
			if (np.abs(Gamm[i,j]) < 1e-8):
				Gamm[i,j] = 0.0
	Node_Type = np.ones(len(A))
	return A, K, Gamm, phi, Node_Type

#########################################

def SP_model(mpc2, est_dyn, Y0):
	x_d = est_dyn["x_d"]
	baseMVA = mpc2[0]["baseMVA"]
	gtb = np.unique(mpc2[0]["gen"][:, GEN_BUS]) #Generator indices
	ngt = len(gtb) # Amount generators
	allbus = mpc2[0]["bus"][:,BUS_I]
	tb = mpc2[0]["gen"][:, GEN_BUS]
	ltb = np.logical_not(np.in1d(allbus,gtb))
	ngi = np.shape(mpc2[0]["gen"])[0]
	Pi = mpc2[0]["gen"][:,PG] / baseMVA
	Qi = mpc2[0]["gen"][:,QG] / baseMVA
	ltb = np.where(ltb)[0]
	nl = len(ltb)
	N = ngi + ngt + nl 
	V = mpc2[0]["bus"][tb,VM] 
	phi = (np.pi*mpc2[0]["bus"][tb,VA])/180
	E = ((V + Qi*x_d[gtb]/V) + 1j*(Pi*x_d[gtb]/V))*np.exp(1j*phi)
	Y0gg = Y0[gtb,:]
	Y0gg = Y0gg[:,gtb]
	Y0gl = Y0[gtb,:]
	Y0gl = Y0gl[:,ltb]
	Y0ll = Y0[ltb,:]
	Y0ll = Y0ll[:,ltb]
	Y0lg = Y0[ltb,:]
	Y0lg = Y0lg[:,gtb]
	Yd = csr_matrix((1/(1j*x_d[gtb]), (range(ngi), range(ngi))), shape=(ngi,ngi))
	Y_SP1 = np.concatenate(( Yd.todense(), -Yd.todense() ), axis = 1)
	Y_SP2 = np.concatenate(( Y_SP1, np.zeros((ngi, nl)) ), axis = 1)
	Y_SP1 = np.concatenate(( -Yd.todense(), Y0gg + Yd.todense() ), axis = 1)
	Y_SP3 = np.concatenate(( Y_SP1, Y0gl.todense() ), axis = 1)
	Y_SP1 = np.concatenate(( np.zeros((nl, ngi)), Y0lg.todense() ), axis = 1)
	Y_SP4 = np.concatenate(( Y_SP1, Y0ll.todense() ), axis = 1)
	Y_SP5 = np.concatenate(( Y_SP2, Y_SP3 ), axis = 0)
	Y_SP = np.concatenate(( Y_SP5, Y_SP4 ), axis = 0)
	Plg = mpc2[0]["bus"][gtb, PD]/baseMVA
	Qlg = mpc2[0]["bus"][gtb, QD]/baseMVA
	Vg = mpc2[0]["bus"][gtb, VM]
	Vl = mpc2[0]["bus"][ltb, VM]
	Pll = mpc2[0]["bus"][ltb, PD]/baseMVA
	Ai = np.concatenate(( Pi[0:ngi], (- Plg - np.square(Vg)*np.real(np.diag(Y0gg.todense() + Yd.todense()))) ), axis = 0)
	A = np.concatenate(( Ai, (- Pll - np.square(Vl)*np.real(np.diag(Y0ll.todense())))  ), axis = 0)
	de1 = np.concatenate(( abs(E), Vg), axis = 0)
	DE = csr_matrix( ( np.concatenate(( de1, Vl), axis = 0), (range(N), range(N)) ), shape=(N,N) )
	K = DE*np.abs(Y_SP)*DE
	aux1 = np.real(Y_SP) != 0
	aux2 = np.imag(Y_SP) != 0
	#Gamm = np.angle(Y_SP+1e-10) - (np.pi/2)*(aux1+aux2)



	Gamm = np.zeros( Y_SP.shape )

	for i in range(Gamm.shape[0]):
		for j in range(Gamm.shape[1]):
			if (np.real(Y_SP[i,j]) == 0.0):
				Gamm[i][j] = 0.0
			else:
				Gamm[i][j] = np.angle(Y_SP[i,j]) - (np.pi/2)*(aux1[i,j] + aux2[i,j])



	for i in range(Gamm.shape[0]):
		for j in range(Gamm.shape[1]):
			if (np.abs(Gamm[i,j]) < 1e-8):
				Gamm[i,j] = 0.0
	phi_3 = (np.pi*mpc2[0]["bus"][:,VA])/180
	phi = np.zeros(N)
	Node_Type = np.ones(N)
	for i in range(N):
		if (i < ngi):
			phi[i] = phi_3[i]
			Node_Type[i] = 1.0
		elif ((i >= ngi) and (i < ngi + ngt)):
			phi[i] = 0.0
			Node_Type[i] = 0.0
		else:
			phi[i] = phi_3[i-ngt]
			Node_Type[i] = 0.0
	return A, K, Gamm, phi, Node_Type

###############################################

def get_kuramoto_net(A, K_hat, Gamm, est_dyn, model, ref_freq, gtb):
	ref_freq = 2*np.pi*ref_freq
	H = est_dyn["H"]
	D = est_dyn["D"]
	NN = np.shape(K_hat)[0]
	all_indxs = np.arange(0,NN)
	ltb = np.logical_not(np.isin(all_indxs, gtb))
	if (model == "en"):
		H = H[gtb]
		D = D[gtb]
		P = 0.5*ref_freq*np.divide(A, H)
		alpha = 0.5*np.divide(D, H)
		Hif = np.repeat(np.array([H]).T, NN, axis = 1)
		K = 0.5*ref_freq*np.divide(K_hat, Hif)
	elif (model == "sm"):
		P = 0.5*ref_freq*np.divide(A, H)
		alpha = 0.5*np.divide(D, H)
		Hif = np.repeat(np.array([H]).T, NN, axis = 1)
		K = 0.5*ref_freq*np.divide(K_hat, Hif)		
	elif (model == "sp"):
		D2 = np.ones(NN)
		for i in range(len(D2)):
			if (i < (len(D))):
				D2[i] = D[i] 
			else:
				D2[i] = D2[i-1]
		D = np.copy(D2)
		P = np.zeros(NN)
		alpha = np.zeros(NN)
		K = np.zeros((NN, NN))
		Hif = np.repeat(np.array([H]).T, NN, axis = 1)
		P[gtb] = 0.5*ref_freq*np.divide(A[gtb], H[gtb])
		alpha[gtb] = 0.5*np.divide(D[gtb], H[gtb])
		for i in gtb:
			for j in range(NN):
				if (i != j):
					K[i,j] = 0.5*ref_freq*(K_hat[i, j]/ Hif[i, j])
		P[ltb] = ref_freq*np.divide(A[ltb], D[ltb])
		alpha[ltb] = 0.0
		Dif = np.repeat(np.array([D]).T, NN, axis = 1)
		ltb = np.where(ltb == 1)[0]
		for i in ltb:
			for j in range(NN):
				if (i != j):
					K[i,j] = ref_freq*(K_hat[i, j]/ Dif[i, j])
	return K, P, alpha, Gamm

####################################


def build_gridcase_graph(net_name, case, model, ref_freq, k_alt_ini, k_alt_fin, k_alt_step, mag_d, re_d, im_d, start_speed, to_plot, to_write):
	'''
	This function creates a graph from a real power grid given by a pypsa case 
	INPUT:
	net_name: <String> - Network name.
	case: <String> - Grid case name. Example: "case9".
	model: <String> - either "sm", "sp" or "en".
	ref_freq: <Double> - Reference frequency.
	k_alt_ini: <Double> - Initial disturbance for the Y_bus. 
	k_alt_fin: <Double> - Final disturbance for the Y_bus.
	k_alt_step: <Double> - Step taken for each disturbance on Y_bus.
	mag_d: <Double> - Number which multiplies the magnitude of the case branches.
	re_d: <Double> - Number which multiplies the real part of the case branches.
	im_d: <Double> - Number which multiplies the imaginary part of the case branches.
	start_speed: <String> - Initial condition for the angular velocity. Enter either "zeros" or "random".
	to_plot: <Boolean> - If want to plot the output graph.
	OUTPUT:
	A text file at Networks folder and a text file at Initial_States folder.
	'''
	if (case == "case9"):
		mpc, est_dyn = gridcase.case9(mag_d, re_d, im_d)
	mpc2 = runpf(mpc)	
	Y0, Yf, Yt = makeYbus(mpc2[0]['baseMVA'], mpc2[0]['bus'], mpc2[0]['branch'])


	k_act = k_alt_ini
	while (k_act < k_alt_fin):
		net_name = net_name + "_kinit_{0:.3g}".format(k_act) 
		Y0_now = k_act*Y0
		if (model == "en"):
			A, K_hat, Gamm, phi, Node_Type = EN_model(mpc2, est_dyn, Y0_now)
		elif (model == "sm"):
			A, K_hat, Gamm, phi, Node_Type = SM_model(mpc2, est_dyn, Y0_now)
		elif (model == "sp"):
			A, K_hat, Gamm, phi, Node_Type = SP_model(mpc2, est_dyn, Y0_now)    
		gtb = np.unique(mpc2[0]["gen"][:, GEN_BUS])
		K, Pi, alpha, Gamm = get_kuramoto_net(A, K_hat, Gamm, est_dyn, model, ref_freq, gtb) 
		if (model == "sm"):
			res = minimize(kuramoto_weight, phi, args = (Pi, K, Gamm), method='nelder-mead', options={'xtol': 1e-6, 'disp': True})
			phi = res.x
		N = np.shape(K)[0]
		link_list = list()
		gamma_list = list()
		P = [[i, Node_Type[i], Pi[i], Pi[i]] for i in range(len(Pi))]
		alf = [[i, alpha[i]] for i in range(len(alpha))]
		for node_i in range(N):
			for node_j  in range(N):
				if (node_j != node_i) and (K[node_i, node_j] != 0):
					link_list.append([node_i, node_j, K[node_i, node_j]])
				if (node_j != node_i) and (Gamm[node_i, node_j] != 0):	
					gamma_list.append([node_i, node_j, Gamm[node_i, node_j]])
		
		if to_write:

			out_file = "Networks/" + net_name + "_.txt"
			file = open(out_file,"w") 
			file.write("{} {} {} \n".format(N, len(link_list), len(gamma_list))) # Nodes Links
			file.write("K \n")
			for i in range(len(link_list)):
				file.write("{} {} {} \n".format(link_list[i][0], link_list[i][1], link_list[i][2])) 
			file.write("P \n")
			for i in range(len(P)):
				file.write("{} {} {} {} \n".format(P[i][0], P[i][1], P[i][2], P[i][3])) 
			file.write("Alfa \n")
			for i in range(len(alf)):
				file.write("{} {} \n".format(alf[i][0], alf[i][1])) 	
			file.write("Gamma \n")
			for i in range(len(gamma_list)):
				file.write("{} {} {} \n".format(gamma_list[i][0], gamma_list[i][1], gamma_list[i][2])) 
			file.close() 
			out_file = "Initial_States/initstate_" + net_name + "_.txt"
			file = open(out_file,"w") 
			for i in range(len(phi)):
				file.write("{}\n".format(phi[i])) 
			for i in range(len(phi)):
				if (start_speed == "zeros"):
					file.write("{}\n".format(0.0))
				elif (start_speed == "random"):
					file.write("{}\n".format(2*np.random.random_sample() - 1))
			file.close() 


		IM_Grapho = nx.from_numpy_matrix(K)
		fr = plt.figure(figsize=(8,8))
		ax1 = fr.add_subplot(111)
		big_gen_list = list()
		small_gen_list = list()
		consumer_list = list()
		big_power = np.max(Pi)
		if (to_plot):
			for a_node in range(len(Pi)):
				if (Pi[a_node] <= 0.0):
					consumer_list.append(a_node)
				else:
					small_gen_list.append(a_node)
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
		k_act = k_act + k_alt_step


	return K



