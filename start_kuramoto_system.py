import numpy as np
import os
from sklearn.utils import shuffle


def get_to_run_file(boost_dir):
'''
INPUT:
boost_dir: <String> - Location of Boost directory.
OUTPUT:
to_run.sh file which lists all the simulations that need to be run.
'''
	all_sets_files = glob("Sim_Settings/*.txt")
	flags_compiler = "c++ -O3 -Wall -ffast-math -std=c++11"
	functions_name = "kuramoto_functions.cpp"
	libout_name = "kuramoto_functions.o"
	file_name = "kuramoto_functions.cpp"
	main_name = "main_kuramoto.cpp"
	file_base_name = "main_kuramoto_grid"
	simulation_commands = list()
	os.system("{} -I {} -c {}".format(flags_compiler, boost_dir, functions_name))
	k = 0
	for sim_set_file in all_sets_files:
		file_base_name_new = file_base_name + "_{}".format(k)
		simulation_commands.append("{} -I {} {} {} -o {} && time ./{}>Results/Testing_out.txt {}".format(flags_compiler, boost_dir, libout_name, main_name, file_base_name_new, file_base_name_new, sim_set_file))
		k = k+1
	sims_2_run = open("to_run.sh", "w")
	for a_sim in simulation_commands:
		sims_2_run.write(a_sim + "\n")
	sims_2_run.close()

	
	
######################################################################################################



def create_system(net_file, initstate_file, sets_file, tini, tfin, steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, model):
'''
INPUT:
net_file: <String> - Filename of the Network file.
initstate_file: <String> - Filename of the Initial State file.
sets_file: <String> - Output filename for the Settings file.
tini: <Double> - Initial simulation time.
tfin: <Double> - Final simulation time.
steps_to_print: <Int> - Steps to simulate before printing output data. 
mx_step: <Double> - Integration step.
kini: <Double> - Initial coupling strength.
kfin: <Double> - Final coupling strength.
kstep: <Double> - Steps for the coupling strength sweep.
t_disturb: <Double> - Time at which a disturbance occurs in the power demand.
t_recover: <Double> - Time at which the system recovers from the disturbance.
model: <String> - Either "sm", "sp" or "en".
OUTPUT:
Settings file with the information of the simulation that will be run.
'''
	settings_file = open(sets_file, "w")
	settings_file.write("T_ini: {} \nT_fin: {} \nPrint_steps: {} \nT_max_step: {} \nT_disturb: {} \nT_recover: {} \n".format(tini, tfin, steps_to_print, mx_step, t_disturb, t_recover))
	settings_file.write("K_ini: {} \nK_fin: {} \nK_step: {} \n".format(kini, kfin, kstep))
	settings_file.write("Network_file: {} \n".format(net_file))
	settings_file.write("Initial_state_file: {} \n".format(initstate_file))
	settings_file.write("Model: {} \n".format(model))
	settings_file.close()

	
	
#######################################################################################################


	
def default_constructor(type_net, dyn_model, delt_d, consumers, Pc, Psg, Pbg, damp):

	if (type_net == "2n"):
		N = 2
		P = (np.zeros((1, N)))
		P[0,0] = Pbg
		P[0,1] = Pc
		alf = (damp*np.ones((1, N)))

	elif (type_net == "case9"): # Something by default, values won't matter
		if (dyn_model == "sm"):
			N = 9
		elif (dyn_model == "en"):
			N = 3
		elif (dyn_model == "sp"):
			N = 12
		P = (np.zeros((1, N)))
		P[0,0] = Pbg
		P[0,1] = Pc
		alf = (damp*np.ones((1, N)))	

	else:
		Nsg = int(round(-(delt_d*consumers*Pc)/Psg)) # Amount of small generators
		Nbg = int(round(-(consumers*Pc + Nsg*Psg)/Pbg)) # Amount of big generators
		N = consumers + Nbg + Nsg

		P = (np.zeros((1, N)))

		for k in range(N):
			if (k < Nbg):
				P[0,k] = Pbg
			elif (k < Nbg + Nsg):
				P[0,k] = Psg
			else:
				P[0,k] = Pc

		alf = (damp*np.ones((1, N)))

		if (type_net == "sw"): # For SmallWorld network shuffle the nodes
			am1 = P.shape[1]
			randomize = np.arange(am1)
			np.random.shuffle(randomize)
			P[0][:] = P[0][randomize]
			alf[0][:] = alf[0][randomize]

	return N, P, alf


def disturbe_all_consumers(P, N, force):
	P_disturbed = (np.zeros((1, N)))
	for i in range(P.shape[1]):
		if P[0, i] < 0:
			P_disturbed[0, i] = force*P[0, i]
		else:
			P_disturbed[0, i] = P[0, i]
	return P_disturbed



def create_simulation_files(P, P_disturbed, alf, type_net, dyn_model, ref_freq, net_name, N, neighbors, pth, mean_degree, consumers, give_initstate_list, init_ang, init_vel, tini, tfin, steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, delt_d, num_init_files):
	# Encoding P and alfa:
	P = P.tolist()[0]
	P_disturbed = P_disturbed.tolist()[0]
	alf = alf.tolist()[0]
	P = str(P)
	P = P.replace(" ", "_")
	P_disturbed = str(P_disturbed)
	P_disturbed = P_disturbed.replace(" ", "_")
	alf = str(alf)
	alf = alf.replace(" ", "_")

	# Construct simulation files:
	
	
	network_file = "Networks/" + net_name + "_.txt"

	if (type_net == "sw"):
		os.system("python3 make_smallworld_graph.py -nodes {} -neighbors {} -pth {} -net_name {} -powers {} -powers_disturb {} -alfas {} -delt_d {}".format(N, neighbors, pth, net_name, P, P_disturbed, alf, delt_d))
	elif (type_net == "rd"):
		os.system("python3 make_random_graph.py -nodes {} -m_degree {} -net_name {} -powers {} -powers_disturb {} -alfas {} -delt_d {}".format(N, mean_degree, net_name, P, P_disturbed, alf, delt_d))
	elif (type_net == "qr"):
		os.system("python3 make_quasiregular_graph.py -nodes {} -consumers {} -net_name {} -powers {} -powers_disturb {} -alfas {} -delt_d {}".format(N, consumers, net_name, P, P_disturbed, alf, delt_d))
	elif (type_net == "2n"):
		os.system("python3 make_2node_graph.py -net_name {} -powers {} -powers_disturb {} -alfas {}".format(net_name, P, P_disturbed, alf))
	elif (type_net[0:4] == "case"):
		os.system("python3 make_gridcase_graph.py -case {} -model {} -ref_freq {} -start_speed {} -k_alt_ini {} -k_alt_fin {} -k_alt_step {}".format(type_net, dyn_model, ref_freq, init_vel, kini, kfin, kstep))

		
	if (type_net[0:4] == "case"):
		k_actual = kini
		while (k_actual < kfin):
			net_name = type_net + "_kinit_{:.3g}_".format(k_actual) + dyn_model 
			network_file = "Networks/" + net_name + "_.txt"
			initstate_file = "Initial_States/initstate_" + net_name + "_.txt"
			settings_file = "Sim_Settings/set_" + net_name + "_.txt"
			os.system("python3 create_system.py -net_file {} -initstate_file {} -sets_file {} -tini {} -tfin {} -steps_to_print {} -mx_step {} -kini {} -kfin {} -kstep {} -t_disturb {} -t_recover {} -model {}".format(network_file, initstate_file, settings_file, tini, tfin, steps_to_print, mx_step, 1, 1, 1, t_disturb, t_recover, dyn_model))
			k_actual = k_actual + kstep


	else:
		if (give_initstate_list == "no"):
			for init_index in range(num_init_files):
				initstate_file = "Initial_States/initstate_" + net_name + "_{}_.txt".format(init_index)
				settings_file = "Sim_Settings/set_" + net_name + "_{}_.txt".format(init_index)
				os.system("python3 generate_init_state.py -nodes {} -init_ang {} -init_vel {} -initstate_file {}".format(N, init_ang, init_vel, initstate_file))
				os.system("python3 create_system.py -net_file {} -initstate_file {} -sets_file {} -tini {} -tfin {} -steps_to_print {} -mx_step {} -kini {} -kfin {} -kstep {} -t_disturb {} -t_recover {} -model {}".format(network_file, initstate_file, settings_file, tini, tfin, steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, dyn_model))
		else:
			for init_index in range(len(give_initstate_list)):
				initstate_file = give_initstate_list[init_index]
				settings_file = "Sim_Settings/set_" + net_name + "_{}_.txt".format(init_index)
				os.system("python3 create_system.py -net_file {} -initstate_file {} -sets_file {} -tini {} -tfin {} -steps_to_print {} -mx_step {} -kini {} -kfin {} -kstep {} -t_disturb {} -t_recover {} -model {}".format(network_file, initstate_file, settings_file, tini, tfin, steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, dyn_model))





def main():
	# Parameters:
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
		N, P, alf = default_constructor(type_net, dyn_model, delt_d, consumers, Pc, Psg, Pbg, damp) 
		
		force = 1 # Strength of the perturbance applied equally to all consumers
		P_disturbed = disturbe_all_consumers(P, N, force)

		##

		print(" Nodes:",  N, "\n", "Sum of vector P:", np.sum(P))
		create_simulation_files(P, P_disturbed, alf, type_net, dyn_model, ref_freq, net_name, N, neighbors, pth, mean_degree, consumers, give_initstate_list, init_ang, init_vel, tini, tfin, steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, delt_d, num_init_files)


if __name__ == '__main__':
	main()
