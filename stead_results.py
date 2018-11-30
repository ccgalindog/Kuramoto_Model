import matplotlib
matplotlib.use("pdf")
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import click

# In this file you find the functions:
# get_result
# get_mean_results

def get_result(result_file, stead_time, tim_step):
	my_data = [line.rstrip('\n') for line in open(result_file, "r")]
	x_data = np.loadtxt(my_data)
	x = x_data[:,1:-2]
	N = int((x.shape[1])/2)
	t = x_data[:,0]
	Re_r = x_data[:,-2]
	Im_r = x_data[:,-1]
	Mag_r = np.sqrt(np.square(Re_r) + np.square(Im_r))
	phases = ( x[:,0:N] + np.pi) % (2 * np.pi ) - np.pi
	phase_velocity = x[:,N:-1]
	stead_point = int(stead_time/tim_step)
	phase_velocity_sq = np.square(phase_velocity[stead_point:,:])
	v_inf = np.mean(np.mean(phase_velocity_sq, axis = 1), axis = 0)
	r_inf = np.mean(Mag_r[stead_point:])
	r_real_inf = np.mean(Re_r[stead_point:])
	r_imag_inf = np.mean(Im_r[stead_point:])
	return r_inf, r_real_inf, r_imag_inf, v_inf;

####################################################

def get_mean_results(std_time, t_step, steps, folders, mean_names, key_index):
	'''
	INPUT:
	std_time: <Double> - Time at which steady state time is taken.
	t_step: <Double> - time used as the integration step time.
	steps: <Int> - steps taken before printing when the simulation was run.
	folders: <List> - Folders that contain files to average. Each folder generates one mean file.
	mean_names: <List> - Same size as folders. Each element is the name to give to the mean file.
	key_index: <Int> - Position in the name of the result files after splitting with "_", such that it
			   contains the element to take as identifier for averaging.
	'''
	t_step = t_step*steps
	for folder, mean_name in zip(folders, mean_names):
		sets_files = sorted( glob( "{}/out_*.txt".format(folder) ) )
		mean_out = "{}/mean_results_{}_.txt".format(folder, mean_name)
		whole_results = open(mean_out, "w")
		for a_set_file in sets_files:
			r_all_inf = list()
			real_all_inf = list()
			imag_all_inf = list()
			k_all_inf = list()
			v_all_inf = list()
			r_inf_j, real_j, imag_j, v_inf_j = get_result(a_set_file, std_time, t_step)
			r_all_inf.append(r_inf_j)
			real_all_inf.append(real_j)
			imag_all_inf.append(imag_j)
			v_all_inf.append(v_inf_j)
			k_j = float(a_set_file.split("_")[key_index])
			k_all_inf.append(k_j)
			whole_results.write("{} {} {} {} {} \n".format(k_j, r_inf_j, real_j, imag_j, v_inf_j))

		whole_results.close()
