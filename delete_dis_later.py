from glob import glob

def main():

	sets_files = glob('Sim_Settings/*')

	for each_file in sets_files:
		lines = [line.rstrip('\n') for line in open(each_file, "r")]

		lines[10] = 'Initial_state_file: Initial_States/initstate_case9_sm_mag_1.0_kinit_1__.txt'

		file = open(each_file, "w")

		for a_line in lines:
			file.write(a_line + '\n')
		file.close()

		print(lines[10])

if __name__ == '__main__':
	main()