#include <iostream>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <stdio.h>      
#include <math.h>   
#include <iomanip>
#include <complex>
#include <string>
#include <cmath>
#include <valarray>    
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>
#include "kuramoto_functions.h"
#include <new>

using namespace boost::numeric::odeint;
using namespace boost::numeric::ublas;

typedef boost::numeric::ublas::vector< double > state_type;


int main(int argc, char const *argv[])
{
	GLOBAL::mm = 0;
	std::string set_file = argv[1];

	kuramoto_system Kuramoto_i;

	std::ifstream sim_settings_file (set_file);
	Kuramoto_i = get_simulation_settings(sim_settings_file, Kuramoto_i);

	std::ifstream initstate_file (Kuramoto_i.input_initstate_file);
	state_type x = get_initial_state(initstate_file, Kuramoto_i);

	std::ifstream params_file (Kuramoto_i.input_network_file);
	Kuramoto_i = get_parameters(params_file, Kuramoto_i);

	state_type x0 = x;
	Kuramoto_i.input_network_file = Kuramoto_i.input_network_file.replace(Kuramoto_i.input_network_file.end()-4, Kuramoto_i.input_network_file.end(), "_");
	Kuramoto_i.input_initstate_file = Kuramoto_i.input_initstate_file.replace(Kuramoto_i.input_initstate_file.end()-4, Kuramoto_i.input_initstate_file.end(), "_");
	set_file = set_file.replace(set_file.end()-4, Kuramoto_i.input_initstate_file.end(), "");
	set_file = set_file.replace(0, 17, "");


	x = x0;

	if (Kuramoto_i.model == "sm"){

		std::vector<kuramoto_2nd_order> dif_systems;
		int obj_indix = 0;
		for ( double k_act = Kuramoto_i.k_ini; k_act <= Kuramoto_i.k_fin; k_act = k_act + Kuramoto_i.k_step ) 
		{
			Kuramoto_i.damage_key = 0;
			Kuramoto_i.update_K_matrix(k_act);
			std::string output_data_file = "Results/out_" + set_file + "k_" + std::to_string(k_act) + "_.txt";
			std::ofstream result_file;
			result_file.open(output_data_file);
			dif_systems.push_back(kuramoto_2nd_order(&result_file, Kuramoto_i));
			kuramoto_2nd_order dynamical_system = dif_systems[obj_indix];
			GLOBAL::mm = 0;
			dynamical_system.overload_iter_indx();
			run_ode_solver(dynamical_system, x, x0, set_file);
			dynamical_system.close_ofstream();
			obj_indix = obj_indix + 1;
		}
	}
	else {

		std::vector<kuramoto_mixed_order> dif_systems;
		int obj_indix = 0;

		for ( double k_act = Kuramoto_i.k_ini; k_act <= Kuramoto_i.k_fin; k_act = k_act + Kuramoto_i.k_step ) 
		{
			Kuramoto_i.damage_key = 0;
			Kuramoto_i.update_K_matrix(k_act);
			std::string output_data_file = "Results/out_" + set_file + "k_" + std::to_string(k_act) + "_.txt";
			std::ofstream result_file;
			result_file.open(output_data_file);
			dif_systems.push_back(kuramoto_mixed_order(&result_file, Kuramoto_i));
			kuramoto_mixed_order dynamical_system = dif_systems[obj_indix];
			GLOBAL::mm = 0;
			dynamical_system.overload_iter_indx();
			run_ode_solver(dynamical_system, x, x0, set_file);
			dynamical_system.close_ofstream();
			obj_indix = obj_indix + 1;
		}
	}
	
	std::cout << "Simulation finished !! \n" << "Network: " << Kuramoto_i.input_network_file << "\n";
	std::cout << "Initial State: " << Kuramoto_i.input_initstate_file << "\n";

	return 0;
}