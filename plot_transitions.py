import matplotlib.style
matplotlib.style.use('classic')
import numpy as np
from matplotlib import pyplot as plt


def main():

	stead_results = np.loadtxt( 'Results/mean_col_sweep_BW.txt' )

	c_crit = 1.412

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot( stead_results[:,0], stead_results[:,1], linewidth = 2, marker = 'd', color = 'chartreuse' )
	ax.plot( [c_crit, c_crit], [0.2, 1.0], color = 'crimson', linestyle = '--', linewidth = 2 )
	ax.set_xticks( np.arange(0.0, 5.1, 0.5) )
	ax.annotate( r'$c^* \approx $' + r'${}$'.format(c_crit), xy = (c_crit, 0.8), xycoords = 'data', xytext = (2.0, 0.9), \
		textcoords = 'data', size = 15, va = 'center', ha = 'center', \
		bbox = dict(boxstyle='round', fc='w'), arrowprops = dict(arrowstyle='fancy', connectionstyle='arc3, rad = -0.2', fc = 'lightgray') )
	ax.set_xlabel(r'$c$')
	ax.set_ylabel(r"$|r_{(t)}|$")
	plt.grid()
	plt.savefig('Images_2/col_magr_bw.pdf')
	plt.show()
	plt.close()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot( stead_results[:,0], stead_results[:,2], linewidth = 2, marker = 'd', color = 'gold' )
	ax.plot( [c_crit, c_crit], [-0.2, 1.0], color = 'crimson', linestyle = '--', linewidth = 2 )
	ax.set_xticks( np.arange(0.0, 5.1, 0.5) )
	ax.annotate( r'$c^* \approx $'+ r'${}$'.format(c_crit), xy = (c_crit, 0.1), xycoords = 'data', xytext = (2.0, 0.3), \
		textcoords = 'data', size = 15, va = 'center', ha = 'center', \
		bbox = dict(boxstyle='round', fc='w'), arrowprops = dict(arrowstyle='fancy', connectionstyle='arc3, rad = -0.2', fc = 'lightgray') )
	ax.set_xlabel(r'$c$')
	ax.set_ylabel(r"$I\!Re [r_{(t)}]$")
	plt.grid()
	plt.savefig('Images_2/col_rer_bw.pdf')
	plt.show()
	plt.close()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot( stead_results[:,0], stead_results[:,3], linewidth = 2, marker = 'd', color = 'deepskyblue' )
	ax.plot( [c_crit, c_crit], [-2.0, np.max( stead_results[:,3] )], color = 'crimson', linestyle = '--', linewidth = 2 )
	ax.set_xticks( np.arange(0.0, 5.1, 0.5) )
	plt.ylim( [-2,14] )
	ax.annotate( r'$c^* \approx $' + r'${}$'.format(c_crit), xy = (c_crit, 4), xycoords = 'data', xytext = (2.0, 7), \
		textcoords = 'data', size = 15, va = 'center', ha = 'center', \
		bbox = dict(boxstyle='round', fc='w'), arrowprops = dict(arrowstyle='fancy', connectionstyle='arc3, rad = -0.2', fc = 'lightgray') )
	ax.set_xlabel(r'$c$')
	ax.set_ylabel(r'$v_{\infty}$' + '   ' + r'$\rm{[\frac{rad}{s}]}$')
	plt.grid()
	plt.savefig('Images_2/col_vinf_bw.pdf')
	plt.show()
	plt.close()

if __name__ == '__main__':
	main()