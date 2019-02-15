import numpy as np
from matplotlib import pyplot as plt

def potential_energy(P, K, phases):

	V = -np.dot( P, phases ) - np.sum( K * np.cos( (np.repeat(np.array([phases]), len(phases), axis=0) - np.repeat(np.array([phases]).T, len(phases), axis=1)) ), axis = None )
	return V

def modified_triangular_net():
	L = 200
	V = np.zeros( (L, L) )
	i = 0
	d3 = 0
	d4 = 0
	k_a = 8.0
	K = k_a * np.array( [ [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0] ] ) #extra at 0
	#K = k_a * np.array( [ [0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0] ] ) #extra at 1
	#K = k_a * np.array( [ [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0] ] ) #extra at 2

	P = np.array( [k_a*0.125, -k_a*0.125, k_a*0.0, k_a*0.0] )

	d1_range = np.linspace(-2*np.pi, 2*np.pi, L)
	d2_range = np.linspace(-2*np.pi, 2*np.pi, L)

	for d1 in d1_range:
		j = 0
		for d2 in d2_range:
			ds = [ d1, d2, d3, d4 ]
			V[i][j] = potential_energy(P, K, ds)
			j = j + 1
		i = i + 1

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	im = ax1.matshow( V, cmap = plt.cm.inferno, extent=[ d1_range[0], d1_range[-1], d2_range[0], d2_range[-1] ], origin = 'lower' )
	cbar = fig.colorbar(im, ax = ax1)
	# cbar.set_ticks(np.arange(-70, 50, 10))
	cbar.set_label(r'$V_{(\delta)}$')
	ax1.set_xlabel(r'$\delta_1$')
	ax1.xaxis.set_label_position('top')
	ax1.set_xticks([-2*np.pi, -1.5*np.pi, -np.pi, -0.5*np.pi, 0., 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
	ax1.set_xticklabels([r"$-2\pi$", r"$-\frac{3\pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
		"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
	ax1.set_yticks([-2*np.pi, -1.5*np.pi, -np.pi, -0.5*np.pi, 0., 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
	ax1.set_yticklabels([r"$-2\pi$", r"$-\frac{3\pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
		"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
	ax1.set_ylabel(r'$\delta_2$')
	ax1.set_aspect('equal')
	plt.savefig('Images/potential_energy_triangle_notng0.pdf')
	plt.show()	

def cyclic_triangular_net():
	L = 200
	V = np.zeros( (L, L) )
	i = 0
	d3 = 0
	k_a = 8.0
	K = k_a * np.array( [ [0, 1, 1], [1, 0, 1], [1, 1, 0] ] )
	P = np.array( [k_a*0.125, -k_a*0.125, k_a*0.0] )

	d1_range = np.linspace(-2*np.pi, 2*np.pi, L)
	d2_range = np.linspace(-2*np.pi, 2*np.pi, L)

	for d1 in d1_range:
		j = 0
		for d2 in d2_range:
			ds = [ d1, d2, d3 ]
			V[i][j] = potential_energy(P, K, ds)
			j = j + 1
		i = i + 1

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	im = ax1.matshow( V, cmap = plt.cm.inferno, extent=[ d1_range[0], d1_range[-1], d2_range[0], d2_range[-1] ], origin = 'lower' )
	cbar = fig.colorbar(im, ax = ax1)
	# cbar.set_ticks(np.arange(-70, 50, 10))
	cbar.set_label(r'$V_{(\delta)}$')
	ax1.set_xlabel(r'$\delta_1$')
	ax1.xaxis.set_label_position('top')
	ax1.set_xticks([-2*np.pi, -1.5*np.pi, -np.pi, -0.5*np.pi, 0., 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
	ax1.set_xticklabels([r"$-2\pi$", r"$-\frac{3\pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
		"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
	ax1.set_yticks([-2*np.pi, -1.5*np.pi, -np.pi, -0.5*np.pi, 0., 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
	ax1.set_yticklabels([r"$-2\pi$", r"$-\frac{3\pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
		"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
	ax1.set_ylabel(r'$\delta_2$')
	ax1.set_aspect('equal')
	plt.savefig('Images/potential_energy.pdf')
	plt.show()

	
def main():
	# Run triangular net example:
	#cyclic_triangular_net()
	modified_triangular_net()
	# res_file = "Results/out_case9_sm_net_0_mag_1.0_kinit_1_k_1.000000_.txt"
	# net_file = "Networks/case9_sm_net_0_mag_1.0_kinit_1_.txt"
	# res_file = "Results/out_rd_sm_net_0_deltd_0.5_400_k_1.000000_.txt"
	# net_file = "Networks/rd_sm_net_0_deltd_0.5_.txt"


	# x_data = np.loadtxt(res_file)
	# x = x_data[:,1:-2]
	# N = int((x.shape[1])/2)
	# t = x_data[:,0]

	# lines = [line.rstrip('\n') for line in open(net_file)]

	# k = 0
	# for a_line in lines:
	# 	if (k == 0):
	# 		nodes = int(a_line.split(" ")[0])
	# 		links = int(a_line.split(" ")[1])
	# 		K = np.zeros( (nodes, nodes) )
	# 		P = np.zeros( nodes )

	# 	elif ((k > 1) and (k < links + 2 )):
	# 		ni = int(a_line.split(" ")[0])
	# 		nj = int(a_line.split(" ")[1])
	# 		K[ni][nj] = float(a_line.split(" ")[2])

	# 	elif ((k > links + 2) and (k < links + nodes + 3 )):
	# 		ni = int(a_line.split(" ")[0])
	# 		P[ni] = float(a_line.split(" ")[2])
	# 	k = k + 1


	# V = np.zeros( x.shape[0] )
	# for s in range( x.shape[0] ):
	# 	V[s] = potential_energy(P, K, x[s][0:nodes])

	# plt.figure()
	# plt.plot(t, V, lw = 2, c = 'r')
	# plt.xlim([0, 5])
	# plt.grid()
	# plt.ylabel(r'$V_{(\delta_{(t)})}$')
	# plt.xlabel(r'$Time   \rm{[s]}$')
	# plt.show()


if __name__ == '__main__':
	main()