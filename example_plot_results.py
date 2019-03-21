import matplotlib.style
matplotlib.style.use('classic')
import kuramoto_graphic
import stead_results


def main():
	result_file = "Results/out_col_net_0_0_k_10.000000_.txt"

	# # For time evolution:
	# stead_points = 100
	# wrap_pi = True
	# kuramoto_graphic.plot_time_evolution(result_file, stead_points, wrap_pi)

	# # For steady state average results:
	folders = ["Results/"]
	mean_names = ["col"]
	key_index = -2
	stead_results.get_mean_results(70, 0.001, 100, folders, mean_names, key_index)

	# For Gifs:
	# out_name = "To_Gif/some_gif.gif"
	# stop_time = 30
	# t_disturb = 170
	# t_recover = 180
	# jumps = 10
	# nodes_to_plot = [1, 5, 10, 90]
	# ki = 1.0
	# kuramoto_graphic.build_circle_gif(result_file, out_name, stop_time, t_disturb, t_recover, jumps, nodes_to_plot, ki)
	

if __name__ == '__main__':
	main()