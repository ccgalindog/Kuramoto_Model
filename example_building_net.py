import numpy as np
import start_kuramoto_system

def main():
	boost_dir = "/home/cristian/boost_1_68_0"
	for num_simulation in range(2, 4):
		type_net = "qr" # Pick either "sw" : Smallworld; "rd" : Random; "qr" : QuasiRegular; "2n" : Two-nodes
		dyn_model = "sm" # Pick either "sm": Synchronous Motor, "sp": Structure Preserving, "en": Effective Network
		init_ang = "random" # Initial state for phases. Pick either "random" or "zeros"
		init_vel = "zeros" # Initial state for phase velocities. Pick either "random" or "zeros"
		num_init_files = 1 # How many different initial conditions want to try
		tini = 0.0 # Initial simulation time
		tfin = 1000.0 # Final simulation time
		mx_step = 0.001 # Step size for integration
		steps_to_print = 100 # Print output data each "steps_to_print" simulation steps
		# If no disturbance needed in P for this simulation then choose t_disturb and t_recover > tfin
		t_disturb = 500.0 # Time at which a disturbance occurs in P of the network
		t_recover = 510.0 # Time at which P of the network recovers
		# To sweep for many values of coupling strength k: 
		# If you need to simulate only for one specific k then choose kfin = kini. 
		kini = 8.0 # Initial k strength
		kfin = 8.1 # Final k strength
		kstep = 0.2 # Steps of k strength to simulate
		force = 30 # Strength of the perturbance applied equally to all consumers
		
		to_plot_net = True


		if (type_net == "rd"):
			consumers = 100
			mean_degree = 6 # Mean connectivity Degree - needed only for Random net
			Po = 1 # Normalization factor for power units
			Pc = -1*Po # Power drained by each consumer
			Psg = 2.5*Po # Power supplied by each small generator (renewable energy source)
			Pbg = 10*Po # Power supplied by each big generator (non-renewable energy source)
			delt_d = 0.5 # Fraction of renewable generation. A number in the range [0, 1]
			damp = 1 # Assumming same alfa for every node
			if ((type_net == "sw") or (type_net == "rd") or (type_net == "qr")):
				net_name = "{}_{}_net_{}_deltd_{}".format(type_net, dyn_model, num_simulation, delt_d) # Name for the network
			else:
				net_name = "{}_{}".format(type_net, dyn_model) # Name for the network
				
			N, P, alf = start_kuramoto_system.default_constructor(type_net, dyn_model, consumers, delt_d, Pc, Psg, Pbg, damp) 

			P_disturbed = start_kuramoto_system.disturbe_all_consumers(P, N, force)
			start_kuramoto_system.create_simulation_files(type_net, dyn_model, net_name, init_ang, init_vel, tini, tfin, \
														  steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, \
														  num_init_files, to_plot_net, P = P, P_disturbed = P_disturbed, \
														  alf = alf, delt_d = delt_d, mean_degree = mean_degree, N = N)



		elif (type_net == "qr"):
			consumers = 100
			Po = 1 # Normalization factor for power units
			Pc = -1*Po # Power drained by each consumer
			Psg = 2.5*Po # Power supplied by each small generator (renewable energy source)
			Pbg = 10*Po # Power supplied by each big generator (non-renewable energy source)
			delt_d = 0.5 # Fraction of renewable generation. A number in the range [0, 1]
			damp = 1 # Assumming same alfa for every node
			if ((type_net == "sw") or (type_net == "rd") or (type_net == "qr")):
				net_name = "{}_{}_net_{}_deltd_{}".format(type_net, dyn_model, num_simulation, delt_d) # Name for the network
			else:
				net_name = "{}_{}".format(type_net, dyn_model) # Name for the network
				
			N, P, alf = start_kuramoto_system.default_constructor(type_net, dyn_model, consumers, delt_d, Pc, Psg, Pbg, damp) 

			P_disturbed = start_kuramoto_system.disturbe_all_consumers(P, N, force)
			start_kuramoto_system.create_simulation_files(type_net, dyn_model, net_name, init_ang, init_vel, tini, tfin, \
														  steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, \
														  num_init_files, to_plot_net, P = P, P_disturbed = P_disturbed, \
														  alf = alf, delt_d = delt_d, consumers = 100, N = N)

		elif (type_net == "sw"):
			consumers = 100
			pth = 0.1 # Rewiring probability - needed only for SmallWorld net
			neighbors = 4 # Closest neighbors - needed only for SmallWorld net
			Po = 1 # Normalization factor for power units
			Pc = -1*Po # Power drained by each consumer
			Psg = 2.5*Po # Power supplied by each small generator (renewable energy source)
			Pbg = 10*Po # Power supplied by each big generator (non-renewable energy source)
			delt_d = 0.5 # Fraction of renewable generation. A number in the range [0, 1]
			damp = 1 # Assumming same alfa for every node
			if ((type_net == "sw") or (type_net == "rd") or (type_net == "qr")):
				net_name = "{}_{}_net_{}_deltd_{}".format(type_net, dyn_model, num_simulation, delt_d) # Name for the network
			else:
				net_name = "{}_{}".format(type_net, dyn_model) # Name for the network
				
			N, P, alf = start_kuramoto_system.default_constructor(type_net, dyn_model, consumers, delt_d, Pc, Psg, Pbg, damp) 

			P_disturbed = start_kuramoto_system.disturbe_all_consumers(P, N, force)
			start_kuramoto_system.create_simulation_files(type_net, dyn_model, net_name, init_ang, init_vel, tini, tfin, \
														  steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, \
														  num_init_files, to_plot_net, P = P, P_disturbed = P_disturbed, \
														  alf = alf, delt_d = delt_d, consumers = 100, pth = pth, neighbors = neighbors, N = N)


		else:

			#for some_mag in (np.arange( 0.05, 4.0, 0.05 )):

			mag_d = 1.0
			re_d = 0.0 
			im_d = 1.0 
			ref_freq = 60

			if ((type_net == "sw") or (type_net == "rd") or (type_net == "qr")):
				net_name = "{}_{}_net_{}_deltd_{}".format(type_net, dyn_model, num_simulation, delt_d) # Name for the network
			else:
				net_name = "{}_{}_net_{}_mag_{}".format(type_net, dyn_model, num_simulation, mag_d) # Name for the network
			print(net_name)
			start_kuramoto_system.create_simulation_files(type_net, dyn_model, net_name, init_ang, init_vel, tini, tfin, \
														  steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, \
														  num_init_files, to_plot_net, mag_d = mag_d, re_d = re_d, im_d = im_d, ref_freq=ref_freq)


	start_kuramoto_system.get_to_run_file(boost_dir)



if __name__ == '__main__':
	main()