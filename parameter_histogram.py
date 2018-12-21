import numpy as np
from matplotlib import pyplot as plt
from glob import glob

def main():
	plt.figure()
	the_files = glob("Networks/*.txt")
	for un_file in the_files:

		c_value = float(un_file.split("_")[-3])

		if ((c_value <= 1) and (c_value > 0)): 

			lines = [line.rstrip('\n') for line in open(un_file)]
			N = int(lines[0].split(" ")[0])
			K_links = int(lines[0].split(" ")[1])
			Gamma_links = int(lines[0].split(" ")[2])

			K_list = list()
			Alfa_list = list()
			

			for i in range(2, K_links+2):
				line = lines[i].split(" ")
				K_list.append( float(line[2]) )

			
			plt.hist(K_list, 100, label="$Y/Y_o = {}$".format(c_value))
	


	plt.legend(loc="best")
	plt.title("Distribution of $K_{ij}$ values")
	plt.grid()
	plt.show()
	plt.close()

	# for i in range(K_links+3, K_links+3)
	# 	line = lines[i]
	# 	K_list.append( float(line.split(" ")[2]) )	




if __name__ == '__main__':
	main()