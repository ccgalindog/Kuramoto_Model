import matplotlib
# matplotlib.use("pdf")
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import blended_transform_factory	
import matplotlib.ticker as plticker
from glob import glob
import click

# In this file you find the functions:
# plot_time_evolution
#

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




