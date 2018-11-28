import matplotlib
matplotlib.use("pdf")
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import click
import os

@click.command()
@click.option("-boost_dir", default = "/home/cristian/boost_1_68_0") #Give the path to Boost C++ Library

def main(boost_dir):
	
	all_sets_files = glob("Sim_Settings/*.txt")
	flags_compiler = "c++ -O3 -Wall -ffast-math -std=c++11"
	functions_name = "kuramoto_functions.cpp"
	libout_name = "kuramoto_functions.o"
	file_name = "kuramoto_functions.cpp"
	main_name = "main_kuramoto.cpp"
	file_base_name = "main_kuramoto_grid"

	simulation_commands = list()
	os.system("{} -I {} -c {}".format(flags_compiler, boost_dir, functions_name))
	k = 0
	for sim_set_file in all_sets_files:

		file_base_name_new = file_base_name + "_{}".format(k)
		simulation_commands.append("{} -I {} {} {} -o {} && time ./{}>Results/Testing_out.txt {}".format(flags_compiler, boost_dir, libout_name, main_name, file_base_name_new, file_base_name_new, sim_set_file))

		k = k+1

	sims_2_run = open("to_run.sh", "w")
	for a_sim in simulation_commands:
		sims_2_run.write(a_sim + "\n")
	sims_2_run.close()

if __name__ == '__main__':
	main()