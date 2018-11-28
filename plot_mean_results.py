import matplotlib
matplotlib.use("pdf")
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from glob import glob


def main():

	deltas_d = [0.0]
	netkindes = ["case9"]

	colors = plt.cm.jet(np.linspace(0.2, 0.9,3))
	markers = ["*", "d", "o"]
	

	for netkind in netkindes:

		kk = 0

		fig = plt.figure()	
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)
		for deltd in deltas_d:

			mean_files = glob("Results/{}/mean_results_{}*".format(deltd, netkind))
			one_file = [line.rstrip('\n') for line in open(mean_files[0], "r")]

			outname = mean_files[0].replace("Results/{}/mean_results_".format(deltd), "")
			outname = outname.replace(".txt", "")
			outname = outname.split("_")[0]

			columns = len(mean_files)
			rows = len(one_file)

			complete_mean_rmag = np.zeros((rows, columns))
			complete_mean_rreal = np.zeros((rows, columns))
			complete_mean_rimag = np.zeros((rows, columns))
			complete_mean_v = np.zeros((rows, columns))
			complete_k = np.zeros((rows, 1))

			print(columns)
			for findex in range(columns):
				mean_file = mean_files[findex]
				dat_file = open(mean_file)
				print(mean_file)
				# outname = mean_files[findex].replace("Results/mean_results_", "")
				# outname = outname.replace(".txt", "")

				mislins = [line.rstrip('\n') for line in dat_file]

				for lindex in range(rows):
					a_line = mislins[lindex]
					some_k = float(a_line.split(" ")[0])
					some_rmag = float(a_line.split(" ")[1])
					some_real = float(a_line.split(" ")[2])
					some_imag = float(a_line.split(" ")[3])
					some_v = float(a_line.split(" ")[4])

					complete_mean_rmag[lindex, findex] = some_rmag
					complete_mean_rreal[lindex, findex] = some_real
					complete_mean_rimag[lindex, findex] = some_imag
					complete_mean_v[lindex, findex] = np.sqrt(some_v)
					complete_k[lindex] = some_k


			R_mag_inf_mean = np.mean(complete_mean_rmag, axis = 1)
			R_real_inf_mean = np.mean(complete_mean_rreal, axis = 1)
			R_imag_inf_mean = np.mean(complete_mean_rimag, axis = 1)
			V_inf_mean = np.mean(complete_mean_v, axis = 1)

			R_mag_inf_std = np.std(complete_mean_rmag, axis = 1)
			R_real_inf_std = np.std(complete_mean_rreal, axis = 1)
			R_imag_inf_std = np.std(complete_mean_rimag, axis = 1)
			V_inf_std = np.std(complete_mean_v, axis = 1)

			ax1.scatter(complete_k, R_mag_inf_mean, marker = markers[kk], color = colors[kk])
			# ax1.errorbar(complete_k, R_mag_inf_mean, yerr = R_mag_inf_std, color = colors[kk])
			
			ax2.scatter(complete_k, V_inf_mean, marker = markers[kk], color = colors[kk])
			# ax2.errorbar(complete_k, V_inf_mean, yerr = V_inf_std, color = colors[kk])
			
			ax3.scatter(complete_k, R_real_inf_mean, marker = markers[kk], color = colors[kk])
			# ax3.errorbar(complete_k, R_real_inf_mean, yerr = R_real_inf_std, color = colors[kk])

			ax4.scatter(complete_k, R_imag_inf_mean, marker = markers[kk], color = colors[kk])
			# ax4.errorbar(complete_k, R_imag_inf_mean, yerr = R_imag_inf_std, color = colors[kk])
			

			
			# ax1.plot(complete_k, R_mag_inf_mean, marker = markers[kk], ms = 4, color = colors[kk], label = r"$\Delta_p = {}$".format(deltd))
			# # # ax1.errorbar(complete_k, R_mag_inf_mean, yerr = R_mag_inf_std, color = colors[kk])
			
			# ax2.plot(complete_k, V_inf_mean, marker = markers[kk], ms = 4, color = colors[kk], label = r"$\Delta_p = {}$".format(deltd))
			# # # ax2.errorbar(complete_k, V_inf_mean, yerr = V_inf_std, color = colors[kk])
			# # ax2.set_ylim([0, 0.2])
			# ax3.plot(complete_k, R_real_inf_mean, marker = markers[kk], ms = 4, color = colors[kk], label = r"$\Delta_p = {}$".format(deltd))
			# # # ax3.errorbar(complete_k, R_real_inf_mean, yerr = R_real_inf_std, color = colors[kk])
			# # ax3.set_ylim([-0.3, 0.3])
			# ax4.plot(complete_k, R_imag_inf_mean, marker = markers[kk], ms = 4, color = colors[kk], label = r"$\Delta_p = {}$".format(deltd))
			# # # ax4.errorbar(complete_k, R_imag_inf_mean, yerr = R_imag_inf_std, color = colors[kk])
			

			kk = kk + 1

		ax1.legend(loc="best", numpoints = 1, markerscale = 1.3, fontsize = 10)
		ax2.legend(loc="best", numpoints = 1, markerscale = 1.3, fontsize = 10)
		ax3.legend(loc="best", numpoints = 1, markerscale = 1.3, fontsize = 10)
		ax4.legend(loc="best", numpoints = 1, markerscale = 1.3, fontsize = 10)
		ax1.set_ylabel(r"$|r_\infty|$")
		ax1.set_xlabel(r"$\frac{K}{P_o}$")
		ax1.set_ylim([0, 1])
		# ax2.set_ylim([0, 1])
		ax2.set_ylabel(r"$v_\infty$")
		# ax1.set_xlim([-2, 12])
		# ax2.set_xlim([-2, 12])
		# ax3.set_xlim([-2, 12])
		# ax4.set_xlim([-2, 12])
		ax2.set_xlabel(r"$\frac{K}{P_o}$")
		ax3.set_ylabel(r"$I\!Re [r_\infty]$")
		ax3.set_xlabel(r"$\frac{K}{P_o}$")
		# ax3.set_xlim([0, 2])
		ax3.set_ylim([-1.2, 1.2])
		ax4.set_ylabel(r"$I\!Im [r_\infty]$")
		ax4.set_xlabel(r"$\frac{K}{P_o}$")
		ax4.set_ylim([-1.2, 1.2])
		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		plt.tight_layout()
		plt.savefig("Images/" + outname + "_meanresults_{}_all_SM_loss.pdf".format(columns))
		plt.close()	


if __name__ == '__main__':
	main()