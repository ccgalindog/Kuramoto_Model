import numpy as np
import click

@click.command()
@click.option("-net_file", default="Networks/network_0.txt")
@click.option("-initstate_file", default="Initial_States/initstate_0.txt")
@click.option("-sets_file", default="Sim_Settings/sim_settings_0.txt")
@click.option("-tini", default = 0.0)
@click.option("-tfin", default = 100.0)
@click.option("-steps_to_print", default = 100.0)
@click.option("-mx_step", default = 100.0)
@click.option("-kini", default = 1.0)
@click.option("-kfin", default = 1.0)
@click.option("-kstep", default = 1.0)
@click.option("-t_disturb", default = 105.0)
@click.option("-t_recover", default = 105.0)
@click.option("-model", default = "sm")

def main(net_file, initstate_file, sets_file, tini, tfin, steps_to_print, mx_step, kini, kfin, kstep, t_disturb, t_recover, model):

	settings_file = open(sets_file, "w")
	settings_file.write("T_ini: {} \nT_fin: {} \nPrint_steps: {} \nT_max_step: {} \nT_disturb: {} \nT_recover: {} \n".format(tini, tfin, steps_to_print, mx_step, t_disturb, t_recover))
	settings_file.write("K_ini: {} \nK_fin: {} \nK_step: {} \n".format(kini, kfin, kstep))
	settings_file.write("Network_file: {} \n".format(net_file))
	settings_file.write("Initial_state_file: {} \n".format(initstate_file))
	settings_file.write("Model: {} \n".format(model))
	settings_file.close()



if __name__ == '__main__':
	main()