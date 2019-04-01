import matplotlib.style
matplotlib.style.use('classic')
#import matplotlib
#matplotlib.use("pdf")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import click
import scipy as sp
from pypower.api import ppoption, runpf, printpf, makeYbus
from pypower.idx_gen import PG, QG, GEN_BUS
from pypower.idx_bus import PD, QD, VM, VA, BUS_I
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import grid_cases as gridcase
from scipy.integrate import odeint
from scipy.interpolate import griddata
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import click
import time
from numba import jit

#@jit
def kuramoto_2nd_order( x, t, P, K, alfs ):
	N = int(len( P ))
	theta = x[:N]
	dot_theta = x[N:]
	dotdot_theta = np.zeros( 2*N )
	dotdot_theta[0:N] = dot_theta
	dotdot_theta[N:] = P - np.multiply( alfs, dot_theta ) + np.sum(K * np.sin( np.repeat( theta.reshape(N,1).T, N, axis=0 ) - np.repeat( theta.reshape(N,1), N, axis=1 ) ), axis=1)

	return dotdot_theta

#@jit
def synch_condition( K, w ):
	#M, N = K.shape()
	#for i in range(M):
	#	for j in range(N):
	#		if (K[i][j] > 0):
	#			K[i][j] = 1.0
	#		elif (K[i][j] < 0):
	#			K[i][j] = -1.0

	#degMat = np.diag( np.sum( A, 2 ) );
	#% Y el Laplaciano es
	#Lap = degMat - A;
	#K = np.array(K != 0, dtype = float)
	G = nx.from_numpy_matrix(K)
	L_dagger = np.linalg.pinv( nx.laplacian_matrix( G ).todense() )

	#print( nx.laplacian_matrix( G ).todense() )
	
	B = nx.incidence_matrix( G, oriented = True ).todense()

	theta_ss = np.matmul(L_dagger, w) 
	max_diff = np.linalg.norm( np.matmul( B.T, theta_ss.T ), np.inf )
	
	return theta_ss, max_diff


@click.command()
@click.option('--k_ij', default = 1.0, help = 'Amplification factor for coupling matrix.')
@click.option('--to_plot', default = False, help = 'Plot phase and phase velocity evolution.')
@click.option('--wrap_pi', default = True, help = 'Plot phase evolution in [-pi, pi] range.')


def main(k_ij, to_plot, wrap_pi):

	K = k_ij*np.loadtxt( 'params_COL/K_Colombia_pu.txt' )
	P = np.loadtxt( 'params_COL/P_Colombia_pu.txt' )
	#Alf = np.loadtxt( 'params_COL/alf_Colombia_pu.txt' )
	Alf = 0.1*np.ones( P.shape )
	N = len(P)
	points_max = 1000
	t_fin = 100 

	start_time = time.time()


	theta_ss, max_diff = synch_condition( K, P )

	print( max_diff, k_ij/max_diff )

	t = np.linspace(0, t_fin, points_max)

	x0 = np.concatenate((theta_ss, 0*theta_ss), axis=1).T
	y0 = x0.flatten()
	y0 = y0.tolist()
	y0 = y0[0]
	states = odeint( kuramoto_2nd_order, y0, t, args=(P, K, Alf) )

	if (wrap_pi):
		phases = ( states[:,0:N] + np.pi) % (2 * np.pi ) - np.pi
	else:
		phases = states[:,0:N]


	phases = phases[:, 0:50]
	phase_vels = states[:, N:N+50]
	end_time = time.time()

	if (to_plot):

		plt.figure()
		plt.plot(t, phases)
		plt.ylabel(r'$\theta$')
		plt.xlabel(r'$t ~~~ \rm{[s]}$')
		plt.grid()
		#plt.xlim([0,10])
		#plt.ylim([-20,20])
		plt.show()

		plt.figure()
		plt.plot(t, phase_vels)
		plt.ylabel(r'$\dot \theta$')
		plt.xlabel(r'$t ~~~ \rm{[s]}$')
		plt.grid()
		#plt.xlim([0,10])
		#plt.ylim([-20,20])
		plt.show()

	print( 'Execution time:', end_time - start_time )

if __name__ == '__main__':
	main()