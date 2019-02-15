import matplotlib.style
matplotlib.style.use('classic')
import matplotlib
matplotlib.use('pdf')
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import stead_results as ss
from mpl_toolkits.mplot3d import Axes3D



def main():
	files = glob("Results/out*")

	some_fs = list(np.random.choice(files, 10))
	some_fs.append('Results/out_rd_sm_net_0_deltd_0.5_1782_k_1.000000_.txt')
	print(some_fs)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	for a_file in some_fs:
		x_data = np.loadtxt(a_file)
		x = x_data[:,1:-2]
		N = int((x.shape[1])/2)
		#x = ( x[:,0:N] + np.pi) % (2 * np.pi ) - np.pi
		t = x_data[:,0]

		ax.plot( t, x[:,0], x[:,1] )

	ax.set_ylim([-2*np.pi, 2*np.pi])
	ax.set_zlim([-2*np.pi, 2*np.pi])
	plt.savefig('Images/3D.pdf')
	fig.show()

if __name__ == '__main__':
	main()