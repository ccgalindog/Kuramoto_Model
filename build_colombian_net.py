import numpy as np

def main():

	K = np.loadtxt('params_COL/K_Colombia.txt')
	P = np.loadtxt('params_COL/P_Colombia.txt')
	Alf = np.loadtxt('params_COL/alf_Colombia.txt')

	N = len(P)
	link_list = np.sum(K != 0)

	print(link_list)

	i = 0
	net_name = 'col_sm_net_{}'.format(i)

	print(P.shape, ' \n', Alf.shape)

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
		file.write("{} {} \n".format(i, Alf[i])) 	
	file.close() 



if __name__ == '__main__':
	main()