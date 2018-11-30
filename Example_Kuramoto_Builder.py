import numpy as np
import os
from sklearn.utils import shuffle
import start_kuramoto_system

def main():
	# Parameters:
	boost_dir = "/home/cristian/boost_1_68_0"
	for num_simulation in range(1):

		type_net = "case9" # Pick either "sw" : Smallworld; "rd" : Random; "qr" : QuasiRegular; "2n" : Two-nodes
		dyn_model = "sp" # Pick either "sm": Synchronous Motor, "sp": Structure Preserving, "en": Effective Network
		
		# Defining Initial Conditions:
		# For give_initstate_list write "no" if you want to create new initial-state files. 
		# If you already have initstate_files you want to load, write a list ["path/initstate_file_0", "path/initstate_file_1", ...]
		give_initstate_list = "no" #["Initial_States/initstate_blahblah_.txt"] 

		# Next three lines only matter if you chose give_initstate_list = "no", otherwise write anything
		init_ang = "random" # Initial state for phases. Pick either "random" or "zeros"
		init_vel = "zeros" # Initial state for phase velocities. Pick either "random" or "zeros"
		num_init_files = 1 # How many different initial conditions want to try

		
		delt_d = 0.0 # Fraction of renewable generation. A number in the range [0, 1]
		if ((type_net == "sw") or (type_net == "rd") or (type_net == "qr")):
			net_name = "{}_{}_net_{}_deltd_{}".format(type_net, dyn_model, num_simulation, delt_d) # Name for the network
		else:
			net_name = "{}_{}".format(type_net, dyn_model) # Name for the network

		ref_freq = 60
		Po = 1 # Normalization factor for power units
		consumers = 100 # Amount of consumers
		Pc = -1*Po # Power drained by each consumer
		Psg = 2.5*Po # Power supplied by each small generator (renewable energy source)
		Pbg = 10*Po # Power supplied by each big generator (non-renewable energy source)
		mean_degree = 6 # Mean connectivity Degree - needed only for Random net
		pth = 0.1 # Rewiring probability - needed only for SmallWorld net
		neighbors = 4 # Closest neighbors - needed only for SmallWorld net
		tini = 0.0 # Initial simulation time
		tfin = 6000.0 # Final simulation time
		mx_step = 0.01 # Step size for integration
		steps_to_print = 10 # Print output data each "steps_to_print" simulation steps

		# If no disturbance needed in P for this simulation then choose t_disturb and t_recover > tfin
		t_disturb = 2000000.0 # Time at which a disturbance occurs in P of the network
		t_recover = 2000010.0 # Time at which P of the network recovers

		# To sweep for many values of coupling strength k: 
		# If you need to simulate only for one specific k then choose kfin = kini. 
		kini = 1.0	# Initial k strength
		kfin = 1.5	# Final k strength
		kstep = 1.0 # Steps of k strength to simulate


		## Enter your code here to define P and alfa as numpy arrays or use the default constructor given:

		damp = 1 # Assumming same alfa for every node
		N, P, alf = start_kuramoto_system.default_constructor(type_net, dyn_model, delt_d, consumers, Pc, Psg, Pbg, damp) 
		
		force = 1 # Strength of the perturbance applied equally to all consumers
		P_disturbed = start_kuramoto_system.disturbe_all_consumers(P, N, force)

		##

		print(" Nodes:",  N, "\n", "Sum of vector P:", np.sum(P))
		start_kuramoto_system.create_simulation_files(P, P_disturbed, alf, type_net, dyn_model, ref_freq, net_name, N, neighbors, pth, mean_degree, consumers, give_initstate_list, init_ang, init_vel, tini, tfin, steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, delt_d, num_init_files)
	start_kuramoto_system.get_to_run_file(boost_dir)
		
if __name__ == '__main__':
	main()
