import matplotlib
matplotlib.use("pdf")
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import click
import os

@click.command()
@click.option("-result_file", default = "Results/out_smallworld_net_2_k_8.000000_.txt")
@click.option("-nodes_to_plot", default = "[0,_1,_5,_10]")
@click.option("-jumps", default = 10)
@click.option("-stop_time", default = 600)
@click.option("-t_disturb", default = 500)
@click.option("-t_recover", default = 505)
@click.option("-ki", default = 1.0)
@click.option("-colors_to_plot", default = "[white,_black]")

def main(result_file, nodes_to_plot, jumps, stop_time, t_disturb, t_recover, ki, colors_to_plot):

	nodes_to_plot = nodes_to_plot.replace("[", "")
	nodes_to_plot = nodes_to_plot.replace("]", "")
	nodes_to_plot = nodes_to_plot.replace(",", "")
	nodes_to_plot = nodes_to_plot.split("_")
	nodes_to_plot = [int(a_node) for a_node in nodes_to_plot]

	colors_to_plot = colors_to_plot.replace("[", "")
	colors_to_plot = colors_to_plot.replace("]", "")
	colors_to_plot = colors_to_plot.replace(",", "")
	colors_to_plot = colors_to_plot.split("_")


	x_data = np.loadtxt(result_file)
	x = x_data[:,1:-2]
	N = int((x.shape[1])/2)
	t = x_data[:,0]
	Re_r = x_data[:,-2]
	Im_r = x_data[:,-1]
	Mag_r = np.sqrt(np.square(Re_r) + np.square(Im_r))

	phases = ( x[:,0:N] + np.pi) % (2 * np.pi ) - np.pi
	phase_velocity = x[:,N:-1]


	colors = np.random.rand(N,3)

	
	m_indx = 0

	for my_time in range(len(t)):
		if ((my_time % jumps == 0) and (t[my_time] <= stop_time)) or ((2*my_time % jumps == 0) and (t[my_time] <= t_recover) and (t[my_time] >= t_disturb)):
			fig = plt.figure()
			ax = fig.add_subplot(111)

			circle1 = plt.Circle((0, 0), 1, color='y', alpha = 0.5)
			plt.gcf().gca().add_artist(circle1)
			i = 0
			for each_node in nodes_to_plot:

				x_act = np.cos(phases[my_time, each_node])
				y_act = np.sin(phases[my_time, each_node])

				plt.scatter(x_act, y_act, color = colors_to_plot[i], s = 50.0, label = r"$i$: {}".format(each_node))
				plt.plot([0.0, x_act], [0.0, y_act], color = colors_to_plot[i], lw = 2.0)
				i = i + 1
			ax.set_title(r"Time: %.2f $\rm{[s]}$     $\kappa = $ %.2f"%(t[my_time], ki))
			if ((t[my_time] >= t_disturb) and (t[my_time] <= t_recover)): 
				plt.text(0.8, 1.0, "Disturbance", size=10, ha="center", va="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
			ax.set_xlim([-1.2, 1.2])
			ax.set_ylim([-1.2, 1.2])
			ax.set_yticklabels([])
			ax.set_xticklabels([])
			plt.legend(loc='center left', numpoints = 1, bbox_to_anchor=(1, 0.5))
			ax.set_xlabel(r"$x$   $\rm{[a.u.]}$")
			ax.set_ylabel(r"$y$   $\rm{[a.u.]}$")
			ax.set_aspect("equal")
			plt.tight_layout()
			plt.savefig("To_Gif/{}.png".format(m_indx))
			plt.close()
			m_indx = m_indx + 1



if __name__ == '__main__':
	main()