import kuramoto_graphic
import stead_results

def main():
	result_file = "Results/out_qr_sm_net_0_deltd_0.5_k_1.000000_.txt"

	# # For time evolution:
	# stead_points = 200
	# wrap_pi = True
	# kuramoto_graphic.plot_time_evolution(result_file, stead_points, wrap_pi)

	# # For steady state average results:
	# folders = ["Results/RD/K_1.0"]
	# mean_names = ["RD"]
	# key_index = -2
	# stead_results.get_mean_results(10, 0.01, 10, folders, mean_names, key_index)

	# For Gifs:
	out_name = "To_Gif/some_gif.gif"
	stop_time = 30
	t_disturb = 170
	t_recover = 180
	jumps = 10
	nodes_to_plot = [1, 5, 10, 90]
	ki = 1.0
	kuramoto_graphic.build_circle_gif(result_file, out_name, stop_time, t_disturb, t_recover, jumps, nodes_to_plot, ki)
	

if __name__ == '__main__':
	main()