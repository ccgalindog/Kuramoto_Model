import numpy as np
from matplotlib import pyplot as plt

def main():

	x = np.linspace(-2*np.pi, 2*np.pi, 45)
	y = np.linspace(-2*np.pi, 2*np.pi, 45)
	X_space, Y_space = np.meshgrid( x, y )
	k = 0
	global_initstate_filename = "Initial_States/all_init_states.txt"
	global_initstate_file = open(global_initstate_filename, 'w')
	for xi in x:
		for yi in y:
			init_filename = "Initial_States/initstate_rd_sm_net_0_deltd_0.5_{}_.txt".format( k )
			state_list = [xi, yi, 0.0, 0.0, 0.0, 0.0]
			init_file = open( init_filename, 'w' )
			global_initstate_file.write("{}".format(k))
			for st in state_list:
				init_file.write( "{}\n".format( st ) )
				global_initstate_file.write( " {}".format( st ) )
			init_file.close()
			global_initstate_file.write("\n")
			k = k + 1
	global_initstate_file.close()

if __name__ == '__main__':
	main()