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

@click.command()
@click.option("-result_file", default = "Results/out_case9_sm_k_1.000000_.txt")


def main(result_file):

	outfile = result_file.split("/")[1]
	outfile = outfile.replace(".txt", "")

	stead_points = 200
	transient_points = -1
	result_file = open(result_file)
	x_data = np.loadtxt(result_file)

	x = x_data[:,1:-2]

	N = int((x.shape[1])/2)
	t = x_data[:,0]
	Re_r = x_data[:,-2]
	Im_r = x_data[:,-1]
	Mag_r = np.sqrt(np.square(Re_r) + np.square(Im_r))

	phases = ( x[:,0:N] + np.pi) % (2 * np.pi ) - np.pi

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
	loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
	axins4.xaxis.set_major_locator(loc)
	axins4.grid()
	# loc = plticker.MultipleLocator(base=6.0)
	# axins4.yaxis.set_major_locator(loc)
	# axins4.set_yticklabels([])
	plt.tight_layout()
	plt.savefig("Images/" + outfile + "_order_.pdf")
	plt.show()
	plt.close()

	print(phases.shape)

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


	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.plot(t, Im_r)
	# # ax.set_ylim([-25000,100])
	# plt.savefig("Images/" + outfile + "_demand_.pdf")
	# plt.close()

if __name__ == '__main__':
	main()