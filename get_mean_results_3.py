import matplotlib
matplotlib.use("pdf")
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import click


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


def main():
	std_time = 3000
	t_step = 0.01*10
	sets_files = sorted(glob("Results/out_*.txt"))


	mean_out = "Results/mean_results_case9_sp_.txt"

	whole_results = open(mean_out, "w")

	for a_set_file in sets_files:
		
		one_set_name = a_set_file.replace("Results/out_", "")
		one_set_name = one_set_name.replace("_.txt", "")

		print(one_set_name)


		r_all_inf = list()
		real_all_inf = list()
		imag_all_inf = list()
		k_all_inf = list()
		v_all_inf = list()

		# net_file = one_set_name.split("_ki_")[0]
		# mean_out = "Results/mean_results_" + net_file + "_.txt"
		

		r_inf_j, real_j, imag_j, v_inf_j = get_result(a_set_file, std_time, t_step)
		r_all_inf.append(r_inf_j)
		real_all_inf.append(real_j)
		imag_all_inf.append(imag_j)
		v_all_inf.append(v_inf_j)
		k_j = float(one_set_name.split("_k_")[1])
		k_all_inf.append(k_j)
		whole_results.write("{} {} {} {} {} \n".format(k_j, r_inf_j, real_j, imag_j, v_inf_j))

	whole_results.close()


if __name__ == '__main__':
	main()