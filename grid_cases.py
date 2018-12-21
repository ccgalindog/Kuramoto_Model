import numpy as np

def case9(mag_d, re_d, im_d):
	"""Power flow data for 9 bus, 3 generator case.
	Please see L{caseformat} for details on the case file format.
	Based on data from Joe H. Chow's book, p. 70.
	@return: Power flow data for 9 bus, 3 generator case.
	"""
	ppc = {"version": '2'}

	##-----  Power Flow Data  -----##
	## system MVA base
	ppc["baseMVA"] = 100.0

	## bus data
	# bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
	ppc["bus"] = np.array([
		[0, 3, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
		[1, 2, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
		[2, 2, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
		[3, 1, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
		[4, 1, 125,  50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
		[5, 1, 90,    30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
		[6, 1, 0,    0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
		[7, 1, 100,    35, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
		[8, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9]
	])

	## generator data
	# bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
	# Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
	ppc["gen"] = np.array([
		[0, 0,   0, 300, -300, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[1, 163, 0, 300, -300, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[2, 85,  0, 300, -300, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	])

	## branch data
	# fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
	ppc["branch"] = np.array([
		[0, 3,     0*mag_d*re_d,   0.0576*mag_d*im_d,        0,  250, 250, 250, 0, 0, 1, -360, 360],
		[3, 4,  0.01*mag_d*re_d,    0.085*mag_d*im_d,   0.088*2, 250, 250, 250, 0, 0, 1, -360, 360],
		[3, 5, 0.017*mag_d*re_d,    0.092*mag_d*im_d,   0.079*2, 250, 250, 250, 0, 0, 1, -360, 360],
		[5, 8, 0.039*mag_d*re_d,     0.17*mag_d*im_d,   0.179*2, 150, 150, 150, 0, 0, 1, -360, 360],
		[2, 8,     0*mag_d*re_d,   0.0586*mag_d*im_d,         0, 300, 300, 300, 0, 0, 1, -360, 360],
		[7, 8, 0.0119*mag_d*re_d,  0.1008*mag_d*im_d,  0.1045*2, 150, 150, 150, 0, 0, 1, -360, 360],
		[6, 7, 0.0085*mag_d*re_d,   0.072*mag_d*im_d,  0.0745*2, 250, 250, 250, 0, 0, 1, -360, 360],
		[1, 6,      0*mag_d*re_d,  0.0625*mag_d*im_d,         0, 250, 250, 250, 0, 0, 1, -360, 360],
		[4, 6, 0.032*mag_d*re_d,    0.161*mag_d*im_d,   0.153*2, 250, 250, 250, 0, 0, 1, -360, 360]
	])

	##-----  OPF Data  -----##
	## area data
	# area refbus
	ppc["areas"] = np.array([
		[1, 5]
	])

	## generator cost data
	# 1 startup shutdown n x1 y1 ... xn yn
	# 2 startup shutdown n c(n-1) ... c0
	ppc["gencost"] = np.array([
		[2, 1500, 0, 3, 0.11,   5,   150],
		[2, 2000, 0, 3, 0.085,  1.2, 600],
		[2, 3000, 0, 3, 0.1225, 1,   335]
	])


	x_d = np.array([0.06, 0.12, 0.18, 1, 0.17, 0.26, 1.0, 0.2, 1.0])
	H = np.array([23.64, 6.4, 3.01, 0.1, 5.0, 3.6, 0.1, 4.0, 0.1])
	# H = 20*np.ones(9)
	D = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50])

	est_dyn = {"x_d": x_d, "H": H, "D": D}


	return ppc, est_dyn
  
  
