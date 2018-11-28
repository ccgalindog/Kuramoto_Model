import numpy as np
import click

@click.command()
@click.option("-nodes", default=5)
@click.option("-init_ang", default="random")
@click.option("-init_vel", default="zeros")
@click.option("-initstate_file", default="Initial_States/initstate_0.txt")

def main(nodes, init_ang, init_vel, initstate_file):


	if (init_ang == "random"):
		theta_0 = 2*np.pi*np.random.rand(nodes)

	if (init_ang == "zeros"):
		theta_0 = np.zeros(nodes)	

	if (init_vel == "random"):
		dot_theta_0 = 20*np.random.rand(nodes)

	if (init_vel == "zeros"):
		dot_theta_0 = np.zeros(nodes)

	x0 = np.concatenate((theta_0, dot_theta_0))

	file_init_state = open(initstate_file,"w")
	for an_x in x0:
		file_init_state.write("{} \n".format(an_x)) 
	file_init_state.close()


if __name__ == '__main__':
	main()