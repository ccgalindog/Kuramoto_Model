import matplotlib
# matplotlib.use("pdf")
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import blended_transform_factory	
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
from glob import glob
import click
import os
import imageio
# In this file you find the functions:
# plot_time_evolution
# plot_phaseplane_2node
# plot_oscillator_circle
# plot_a_graph
# build_circle_gif



def plot_time_evolution(result_file, stead_points, wrap_pi):
	'''
	Get graphics of the time evolution of one system.
	INPUT:
	result_file: <String> - File name of the text file containing the results of a simulation.
	stead_points: <Int> - How many points to plot from the steady-state dynamics.
	wrap_pi: <Boolean> - You want the phase evolution beeing plotted in the range [-Pi, Pi] or not.
	'''
	outfile = result_file.split("/")[1]
	outfile = outfile.replace(".txt", "")
	result_file = open(result_file)
	x_data = np.loadtxt(result_file)
	x = x_data[:,1:-2]
	N = int((x.shape[1])/2)
	t = x_data[:,0]
	Re_r = x_data[:,-2]
	Im_r = x_data[:,-1]
	Mag_r = np.sqrt(np.square(Re_r) + np.square(Im_r))
	if (wrap_pi):
		phases = ( x[:,0:N] + np.pi) % (2 * np.pi ) - np.pi
	else:
		phases = x[:,0:N]
	phase_velocity = x[:,N:]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(t[:], Mag_r[:], color = "seagreen", label = r"$|r_{(t)}|$")
	ax.plot(t[:], Re_r[:], color = "midnightblue", label = r"$I\!Re [r_{(t)}]$")
	ax.plot(t[:], Im_r[:], color = "crimson", label = r"$I\!Im [r_{(t)}]$")
	plt.ylabel(r"$r_{(t)}$")
	plt.xlabel(r"Time $\rm{[s]}$")
	ax.legend(loc=1)
	plt.ylim([-1.1, 1.1])
	plt.grid()
	axins4 = inset_axes(ax, width="30%", height="34%", loc=4, borderpad=2.5)
	axins4.plot(t[-stead_points:], Mag_r[-stead_points:], color = "seagreen", label = r"$|r_{(t)}|$")
	axins4.plot(t[-stead_points:], Re_r[-stead_points:], color = "midnightblue", label = r"$I\!Re [r_{(t)}]$")
	axins4.plot(t[-stead_points:], Im_r[-stead_points:], color = "crimson", label = r"$I\!Im [r_{(t)}]$")
	loc = plticker.MultipleLocator(base=1.0) 
	axins4.xaxis.set_major_locator(loc)
	axins4.grid()
	plt.tight_layout()
	plt.savefig("Images/" + outfile + "_order_.pdf")
	plt.show()
	plt.close()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for jik in range(len(phases[0,:])):
		ax.plot(t[:], phases[:,jik], label = "Node: {}".format(jik))
	plt.legend(loc="best")
	plt.ylabel(r"$\theta_{(t)}$   $\rm{[rad]}$")
	plt.xlabel(r"Time $\rm{[s]}$")
	plt.grid()
	plt.tight_layout()
	plt.savefig("Images/" + outfile + "_phases_.pdf")
	plt.show()
	plt.close()

	nn1 = int(np.round(2*len(phase_velocity)/5))

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(t[:], phase_velocity[:][:])
	plt.ylabel(r"$\dot \theta_{(t)}$   $\rm{[\frac{rad}{s}]}$")
	plt.xlabel(r"Time $\rm{[s]}$")
	plt.grid()
	plt.savefig("Images/" + outfile + "_phasevels_.pdf")
	plt.show()
	plt.close()


	
####################################################################################################



def plot_phaseplane_2node(result_folder):
	'''
	Plot the phase plane for a 2-node network.
	INPUT:
	result_folder: <String> - Folder where all the result files are located.
	'''
	all_out_files = glob("{}/out*".format(result_folder))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for outfile in all_out_files:
		outfile2 = outfile.split("/")[-1]
		outfile2 = outfile2.replace(".txt", "")
		ki = float(outfile2.split("_")[-2])
		stead_points = 200
		transient_points = -1
		outfile = open(outfile)
		x_data = np.loadtxt(outfile)
		x = x_data[:,1:-2]
		N = int((x.shape[1])/2)
		t = x_data[:,0]
		Re_r = x_data[:,-2]
		Im_r = x_data[:,-1]
		Mag_r = np.sqrt(np.square(Re_r) + np.square(Im_r))
		phases = x[:,0:N]
		phase_velocity = x[:,N:]
		dif_phases = phases[:,0] - phases[:,1]
		dif_vels = phase_velocity[:,0] - phase_velocity[:,1]
		ax.plot(dif_vels, dif_phases, color = "seagreen")
	plt.text(7, 5, r"$K = {}$".format(ki), size=20, ha="center", va="center", bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
	plt.ylabel(r"$\Delta \theta$ $\rm{[rad]}$")
	plt.xlabel(r"$\Delta \chi$ $\rm{[rad/s]}$")
	plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
	plt.ylim([0, 2*np.pi])
	plt.xlim([-10, 10])
	plt.grid()
	plt.tight_layout()
	plt.savefig("Images/" + outfile2 + "_phasespace_.pdf")
	plt.close()



#######################################################################################################



def plot_oscillator_circle(result_file, nodes_to_plot, jumps, stop_time, t_disturb, t_recover, ki, colors_to_plot):
	'''
	Plot time evolution as points rotating in a unit circle.
	INPUT:
	result_file: <String> - Filename.
	nodes_to_plot: <List> - Nodes you want to watch.
	jumps: <Int> - Jumps taken in time to choose points to plot.
	stop_time: <Double> - When to stop plotting.
	t_disturb: <Double> - Time at which a disturbance occurred (if none, use t_disturb > stop_time).
	t_recover: <Double> - Time at which a disturbance finished (if none, use t_recover > stop_time).
	ki: <Double> - Coupling stregth used in the simulation.
	colors_to_plot: <List> - Color given to each node.
	OUTPUT:
	Some PNG images in the folder To_Gif, each one showing the state of the system at some point in time.
	'''
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
			
			
			
######################################################################################################################



def plot_a_graph(net_name, net_file):
	'''
	net_name: <String> - Name of the network.
	net_file: <String> - Filename of the network.

	'''
	created_file = "Networks/{}_.txt".format(net_name)
	os.system("cp {} {}".format(net_file, created_file))
	net_name = net_file.replace("Example_Cases/", "")
	net_name = net_name.replace("_.txt", "")
	delt_d = 0.5
	lines = [line.rstrip('\n') for line in open(net_file,"r")]
	lin_indx = 0
	for line in lines:
		esin_lin = line.split(" ")
		if (lin_indx == 0):
			N = int(esin_lin[0])
			interacts = int(esin_lin[1])
			K = np.zeros((N,N))
			P = np.zeros((N,1))
			P_disturbed = np.zeros(N)
			alf = np.zeros(N)
		elif ((lin_indx > 1) and (lin_indx < interacts + 2)):
			ni = int(esin_lin[0])
			nj = int(esin_lin[1])
			K[ni][nj] = float(esin_lin[2])
		elif ((lin_indx > interacts + 2) and (lin_indx < interacts + N + 3)):
			ni = int(esin_lin[0])
			P[ni] = float(esin_lin[1])
			# P_disturbed[ni] = float(esin_lin[2])
		lin_indx = lin_indx + 1
	IM_Grapho = nx.from_numpy_matrix(K)
	fr = plt.figure(figsize=(8,8))
	ax1 = fr.add_subplot(111)
	big_gen_list = list()
	small_gen_list = list()
	consumer_list = list()
	P = np.array(P)
	big_power = np.max(P[:])
	for a_node in range(len(P)):
		if (P[a_node] < 0.0):
			consumer_list.append(a_node)
		elif ((P[a_node] == big_power) and (delt_d < 1.0)):
			big_gen_list.append(a_node)
		else:
			small_gen_list.append(a_node)
	pos=nx.spring_layout(IM_Grapho)
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=big_gen_list, node_color='crimson', node_size=100, alpha=0.9, label = "Big Generators")
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=small_gen_list, node_color='yellowgreen', node_size=70, alpha=0.9, label = "Small Generators")
	nx.draw_networkx_nodes(IM_Grapho, pos, nodelist=consumer_list, node_color='indigo', node_size=50, alpha=0.9, label = "Consumers")
	plt.legend(loc="best", scatterpoints=1)
	nx.draw_networkx_edges(IM_Grapho, pos, width=1.0,alpha=0.5)
	ax1.set_xticklabels('')
	ax1.set_yticklabels('')
	ax1.tick_params(axis='both', which='both', length = 0, bottom=False, top=False, labelbottom=False)
	plt.tight_layout()
	fr.savefig("Images/" + net_name + "_.pdf", bbox_inches='tight')
	plt.close()



#################################################################################################################



def build_circle_gif(result_file, out_name, stop_time, t_disturb, t_recover, jumps, nodes_to_plot, ki):
	'''
	Takes every image inside To_Gif folder and makes a gif.
	INPUT:
	result_file: <String> - Name of file with results to plot.
	out_name: <String> - Name of the output Gif.
	stop_time: <Double> - Stop plotting at.
	t_disturb: <Double> - Time at which a disturbance occurred. If none use t_disturb > stop_time.
	t_recover: <Double> - Time at which the system recovers from a disturbance. If none use t_recover > stop_time.
	jumps: <Int> - How many integration steps to jump before building a shot of the system state.
	nodes_to_plot: <List> - Each element is a node that wants to be plotted.
	ki: <Double> - Value identifier for this simulation.
	'''
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

	plot_oscillator_circle(result_file, nodes_to_plot, jumps, stop_time, t_disturb, t_recover, ki, colors_to_plot)
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
	
	

####################################################################################################

