import matplotlib
matplotlib.use("pdf")
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import click
import os
import imageio

def main():

	result_file = "Results/out_quasiregular_net_99_k_8.000000_.txt"
	out_name = "To_Gif/my_second_gif.gif"
	stop_time = 340
	t_disturb = 170
	t_recover = 180
	jumps = 10
	nodes_to_plot = list(range(125))


	#####

	net_file = result_file.replace("Results/out_", "")
	da_k = net_file.split("k_")[1]
	da_k = float(da_k.split("_")[0])
	net_file = net_file.split("k_")[0]
	net_file = "Networks/" + net_file + ".txt"

	lines = [line.rstrip('\n') for line in open(net_file, "r")]

	interactions = int(lines[0].split(" ")[1])

	colors_to_plot = list()
	gimme_da_power = list()
	for node in nodes_to_plot:
		power_line = lines[interactions + 3 + node]
		gimme_da_power.append(float(power_line.split(" ")[1]))

	gimme_da_power = np.array(gimme_da_power)
	max_power = np.amax(gimme_da_power)

	for an_indx in range(len(nodes_to_plot)):
		if (gimme_da_power[an_indx] < 0):
			colors_to_plot.append("indigo")
		elif (gimme_da_power[an_indx] == max_power):
			colors_to_plot.append("crimson")
		else:
			colors_to_plot.append("yellowgreen")	



	nodes_to_plot = str(nodes_to_plot)
	nodes_to_plot = nodes_to_plot.replace(" ", "_")
	
	colors_to_plot = str(colors_to_plot)
	colors_to_plot = colors_to_plot.replace(" ", "_")

	

	os.system("python3 plot_oscillator_circle.py -result_file {} -nodes_to_plot {} -jumps {} -stop_time {} -t_disturb {} -t_recover {} -ki {} -colors_to_plot {}".format(result_file, nodes_to_plot, jumps, stop_time, t_disturb, t_recover, da_k, colors_to_plot))


	images_f = glob("To_Gif/*.png")
	c1 = 0

	for an_img in images_f:
		an_img = an_img.split(".")[0]
		an_img = float(an_img.split("/")[1])
		if an_img > c1:
			c1 = an_img

	print(c1)

	for ijk in range(int(c1)):
		if ijk < 10:
			os.system("mv To_Gif/{}.png To_Gif/000{}.png".format(ijk, ijk))
		elif ijk < 100:
			os.system("mv To_Gif/{}.png To_Gif/00{}.png".format(ijk, ijk))
		elif ijk < 1000:
			os.system("mv To_Gif/{}.png To_Gif/0{}.png".format(ijk, ijk))


	filenames = sorted(glob("To_Gif/*.png"))
	images = []
	for filename in filenames: 
		images.append(imageio.imread(filename))
	kargs = { 'duration': 0.15}

	
	imageio.mimsave(out_name, images, **kargs)



if __name__ == '__main__':
	main()