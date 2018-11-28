import numpy
from glob import glob
import os

def main():

	da_files = glob("Results/*.txt")

	for un_file in da_files:
		sim_index = int(un_file.split("_")[-4])
		mod_file = un_file.replace("net_{}".format(sim_index), "net_{}".format(sim_index+10))
		# print(mod_file)
		os.system("mv {} {}".format(un_file, mod_file))




if __name__ == '__main__':
	main()