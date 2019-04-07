import numpy as np
import matplotlib.style
matplotlib.style.use('classic')
#import matplotlib
#matplotlib.use("pdf")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.interpolate import griddata
from numba import cuda, autojit, jit, vectorize
from pylab import imshow, show
from timeit import default_timer as timer
import time

def kuramoto_2nd_order( t, x, P, K, alfs ):
	N = int(len( P ))
	theta = x[:N]
	dot_theta = x[N:]
	dotdot_theta = np.zeros( 2*N, np.float64 )
	dotdot_theta[0:N] = dot_theta
	dotdot_theta[N:] = P - np.multiply( alfs, dot_theta ) + np.sum(K * np.sin( np.repeat( theta.reshape(N,1).T, N, axis=0 ) - np.repeat( theta.reshape(N,1), N, axis=1 ) ), axis=1)

	return dotdot_theta


def kuramoto_run(x0, K, P, Alf, t_fin, ki, dinode, sim_indic, to_return):
	stim = time.time()
	N = len(P)
	solver = ode( kuramoto_2nd_order )
	solver.set_integrator('vode', method = 'bdf', order = 5, nsteps=3000)
	solver.set_f_params(P, K, Alf)
	solver.set_initial_value(x0, 0)

	i = 0
	states = []
	t = []
	while solver.successful() and solver.t < t_fin:
		solver.integrate(1, step=True) 
		states.append( solver.y )
		t.append(solver.t)
		i += 1

	t = np.array(t)
	states = np.array(states)
	
	plt.figure()
	plt.subplot(121)
	plt.plot(t, states[:,:N])
	plt.subplot(122)
	plt.plot(t, states[:,N:])
	plt.show()

	tot_datos = np.int(0.9*len(t))
	t = t[ tot_datos: ].reshape(-1,1)
	states = states[ tot_datos:, : ]


	states[:,0:N] = ( states[:, 0:N] + np.pi) % (2 * np.pi ) - np.pi
	#phase_vels = states[-1, N:]
	
	#end_state = np.concatenate( [phases, phase_vels] )
	

	filname = 'Results/out_col_k_{}_node_{}_instate_{}_.txt'.format(ki, dinode, sim_indic)
	np.savetxt( filname, np.concatenate( [t, states], axis = 1 ) )

	etim = time.time()
	print('Done: ', filname,'\n run_time: ', etim - stim)
	
	
	if to_return:
		return states[-1,:]


def synch_condition( K, w ):
	'''
	Returns an approximation to the critical coupling and to the steady state
	of the system, calculated from topological considerations.
	'''
	G = nx.from_numpy_matrix(K)
	L_dagger = np.linalg.pinv( nx.laplacian_matrix( G ).todense() )
	B = nx.incidence_matrix( G, oriented = True ).todense()
	theta_ss = np.matmul(L_dagger, w) 
	x0 = np.concatenate((theta_ss, 0*theta_ss), axis=1).T
	x0 = x0.flatten()
	x0 = x0.tolist()
	x0 = x0[0]
	k_crit = np.linalg.norm( np.matmul( B.T, theta_ss.T ), np.inf )

	return k_crit, x0


def single_node_bs( kth, K, P, Alf, t_fin, angs_rank, vels_rank, ci ):
	N = len(P)
	k_crit, x0 = synch_condition( K, P )
	x0 = kuramoto_run( x0, K, P, Alf, t_fin, ci, kth, -1, True )
	y0 = np.copy( x0 )
	state_returns = 0.0
	state_totals = len(angs_rank)*len(vels_rank)
	sim_indic = 0
	for ang_k in angs_rank:
		print( 'Progress: ', sim_indic/state_totals )
		for velang_k in vels_rank:
			y0[kth] = ang_k
			y0[kth+N] = velang_k
			kuramoto_run( y0, K, P, Alf, t_fin, ci, kth, sim_indic, False )
			sim_indic = sim_indic + 1
			#error_traj = np.linalg.norm( yfin - x0 )

			#error_traj = np.linalg.norm( yfin[N:] ) #magnitude velocity
			#print( error_traj)
			#if (error_traj < 1e-2):
			#	state_returns = state_returns + 1.0			

			#if (error_traj < 10):
			#	state_returns = state_returns + 1.0
				
	#BS = state_returns/state_totals
	


	#return BS, x0



def main():
	ci = 1.5
	K = ci*np.loadtxt( 'params_COL/K_Colombia_pu.txt' )
	P = np.loadtxt( 'params_COL/P_Colombia_pu.txt' )
	Alf = 0.1*np.ones( P.shape )


	t_fin = 200
	kth_node = 21
	angs_rank = np.linspace(-np.pi/2, np.pi/2, 3)
	vels_rank = np.linspace(-100, 100, 3)
	
	for kth_node in range( len(P) ):
		single_node_bs( kth_node, K, P, Alf, t_fin, angs_rank, vels_rank, ci)


if __name__ == '__main__':
	main()