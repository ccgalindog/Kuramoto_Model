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

def main():

	all_out_files = glob("Results/K_1.1/out*")
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
	# loc = plticker.MultipleLocator(base=6.0)
	# axins4.yaxis.set_major_locator(loc)
	# axins4.set_yticklabels([])
	plt.tight_layout()
	# plt.show()
	plt.savefig("Images/" + outfile2 + "_phasespace_.pdf")
	plt.close()


if __name__ == '__main__':
	main()