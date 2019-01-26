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

// using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::ublas;

typedef boost::numeric::ublas::vector< double > state_type;


kuramoto_system::kuramoto_system()
{

};



void kuramoto_2nd_order::operator() ( const state_type &x , state_type &dxdt , const double t)
{	
	matrix<double> theta_rept (Kuramoto_i.N, Kuramoto_i.N);

	if (t == 0){GLOBAL::mm = 0;}

	Kuramoto_i.r_real = 0.0;
	Kuramoto_i.r_imag = 0.0;
	scalar_vector<double> ones_vec (Kuramoto_i.N, 1);

	Kuramoto_i.P_balance = inner_prod(Kuramoto_i.P, ones_vec);

	state_type theta_x = subrange(x, 0, Kuramoto_i.N);
	state_type dot_theta_x = subrange(x, Kuramoto_i.N, 2*Kuramoto_i.N);

	state_type cos_theta = apply_elementwise_vec(cos, theta_x);
	state_type sin_theta = apply_elementwise_vec(sin, theta_x);

	Kuramoto_i.r_real = (inner_prod(cos_theta, ones_vec))/Kuramoto_i.N;
	Kuramoto_i.r_imag = (inner_prod(sin_theta, ones_vec))/Kuramoto_i.N;

	for (int i = 0; i < Kuramoto_i.N; i = i + 1){
		row(theta_rept, i) = x[i]*ones_vec;
	}

	matrix<double> phas = trans(theta_rept) - theta_rept + Kuramoto_i.Gamm;
	phas = element_prod(Kuramoto_i.K, apply_elementwise(sin, phas));
	
	vector<double> sum_phases = prod(phas, ones_vec);

	subrange(dxdt, 0, Kuramoto_i.N) = dot_theta_x;
	subrange(dxdt, Kuramoto_i.N, 2*Kuramoto_i.N) = Kuramoto_i.P - element_prod(Kuramoto_i.alf, dot_theta_x) + sum_phases; 

	
	// OBSERVER:

	std::string out_string;
	if ((t > Kuramoto_i.t_disturb) && (Kuramoto_i.damage_key == 0))
	{
		for (int i = 0; i < Kuramoto_i.N; i++) {
			Kuramoto_i.P[i] = Kuramoto_i.P_disturbed[i];
		}
		Kuramoto_i.damage_key = 1;
	}

	if ((t > Kuramoto_i.t_recover) && (Kuramoto_i.damage_key == 1))
	{
		for (int i = 0; i < Kuramoto_i.N; i++) {
			Kuramoto_i.P[i] = Kuramoto_i.P_initial[i];
		}
		Kuramoto_i.damage_key = 2;
	}

	if (t == ((Kuramoto_i.tw)*(GLOBAL::mm)))
	{
		out_string = std::to_string(t) + "\t";
		write_ofstream(out_string);
		for (int ijk = 0; ijk < 2*Kuramoto_i.N; ijk++) {
			out_string = std::to_string(x[ijk]) + "\t";
			write_ofstream(out_string);
		}
		out_string = std::to_string(Kuramoto_i.r_real) + "\t" + std::to_string(Kuramoto_i.r_imag) + "\n";
		write_ofstream(out_string);

		GLOBAL::mm = GLOBAL::mm + 1;
		add_iter_indx();

	}

}


void kuramoto_mixed_order::operator() ( const state_type &x , state_type &dxdt , const double t)
{	
	if (t == 0){GLOBAL::mm = 0;}
	matrix<double> theta_rept (Kuramoto_i.N, Kuramoto_i.N);

	Kuramoto_i.r_real = 0.0;
	Kuramoto_i.r_imag = 0.0;
	scalar_vector<double> ones_vec (Kuramoto_i.N, 1);

	Kuramoto_i.P_balance = inner_prod(Kuramoto_i.P, ones_vec);

	state_type theta_x = subrange(x, 0, Kuramoto_i.N);
	state_type dot_theta_x = subrange(x, Kuramoto_i.N, 2*Kuramoto_i.N);

	state_type cos_theta = apply_elementwise_vec(cos, theta_x);
	state_type sin_theta = apply_elementwise_vec(sin, theta_x);

	Kuramoto_i.r_real = (inner_prod(cos_theta, ones_vec))/Kuramoto_i.N;
	Kuramoto_i.r_imag = (inner_prod(sin_theta, ones_vec))/Kuramoto_i.N;

	for (int i = 0; i < Kuramoto_i.N; i = i + 1){
		row(theta_rept, i) = x[i]*ones_vec;
	}

	matrix<double> phas = trans(theta_rept) - theta_rept + Kuramoto_i.Gamm;
	phas = element_prod(Kuramoto_i.K, apply_elementwise(sin, phas));
	
	vector<double> sum_phases = prod(phas, ones_vec);


	for (int i = 0; i < Kuramoto_i.N; i = i + 1){

		if (Kuramoto_i.Node_type[i] == 1.0){ // Generator
			dxdt[i] = dot_theta_x[i];
			dxdt[i + Kuramoto_i.N] = Kuramoto_i.P[i] - Kuramoto_i.alf[i]*dot_theta_x[i] + sum_phases[i];
		}
		else{ // Load
			dxdt[i] = Kuramoto_i.P[i] + sum_phases[i];
			dxdt[i + Kuramoto_i.N] = 0.0;
		}

	}	


	// OBSERVER:
	std::string out_string;
	if ((t > Kuramoto_i.t_disturb) && (Kuramoto_i.damage_key == 0))
	{
		for (int i = 0; i < Kuramoto_i.N; i++) {
			Kuramoto_i.P[i] = Kuramoto_i.P_disturbed[i];
		}
		Kuramoto_i.damage_key = 1;
	}

	if ((t > Kuramoto_i.t_recover) && (Kuramoto_i.damage_key == 1))
	{
		for (int i = 0; i < Kuramoto_i.N; i++) {
			Kuramoto_i.P[i] = Kuramoto_i.P_initial[i];
		}
		Kuramoto_i.damage_key = 2;
	}

	if (t == (Kuramoto_i.tw)*(GLOBAL::mm)) 
	{
		out_string = std::to_string(t) + "\t";
		write_ofstream(out_string);
		for (int ijk = 0; ijk < 2*Kuramoto_i.N; ijk++) {
			out_string = std::to_string(x[ijk]) + "\t";
			write_ofstream(out_string);
		}
		out_string = std::to_string(Kuramoto_i.r_real) + "\t" + std::to_string(Kuramoto_i.r_imag) + "\n";
		write_ofstream(out_string);

		GLOBAL::mm = GLOBAL::mm + 1;
	}



}


void kuramoto_1st_order::operator() ( const state_type &x , state_type &dxdt , const double t)
{	
	if (t == 0){GLOBAL::mm = 0;}

	matrix<double> theta_rept (Kuramoto_i.N, Kuramoto_i.N);

	Kuramoto_i.r_real = 0.0;
	Kuramoto_i.r_imag = 0.0;
	scalar_vector<double> ones_vec (Kuramoto_i.N, 1);

	Kuramoto_i.P_balance = inner_prod(Kuramoto_i.P, ones_vec);

	state_type theta_x = subrange(x, 0, Kuramoto_i.N);
	state_type dot_theta_x = subrange(x, Kuramoto_i.N, 2*Kuramoto_i.N);

	state_type cos_theta = apply_elementwise_vec(cos, theta_x);
	state_type sin_theta = apply_elementwise_vec(sin, theta_x);

	Kuramoto_i.r_real = (inner_prod(cos_theta, ones_vec))/Kuramoto_i.N;
	Kuramoto_i.r_imag = (inner_prod(sin_theta, ones_vec))/Kuramoto_i.N;

	for (int i = 0; i < Kuramoto_i.N; i = i + 1){
		row(theta_rept, i) = x[i]*ones_vec;
	}

	matrix<double> phas = trans(theta_rept) - theta_rept + Kuramoto_i.Gamm;
	phas = element_prod(Kuramoto_i.K, apply_elementwise(sin, phas));
	
	vector<double> sum_phases = prod(phas, ones_vec);

	subrange(dxdt, 0, Kuramoto_i.N) = Kuramoto_i.P + sum_phases;
	subrange(dxdt, Kuramoto_i.N, 2*Kuramoto_i.N) = state_type (Kuramoto_i.N); 

	// OBSERVER:

	std::string out_string;
	if ((t > Kuramoto_i.t_disturb) && (Kuramoto_i.damage_key == 0))
	{
		for (int i = 0; i < Kuramoto_i.N; i++) {
			Kuramoto_i.P[i] = Kuramoto_i.P_disturbed[i];
		}
		Kuramoto_i.damage_key = 1;
	}

	if ((t > Kuramoto_i.t_recover) && (Kuramoto_i.damage_key == 1))
	{
		for (int i = 0; i < Kuramoto_i.N; i++) {
			Kuramoto_i.P[i] = Kuramoto_i.P_initial[i];
		}
		Kuramoto_i.damage_key = 2;
	}

	if (t == (Kuramoto_i.tw)*(GLOBAL::mm)) 
	{
		out_string = std::to_string(t) + "\t";
		write_ofstream(out_string);
		for (int ijk = 0; ijk < 2*Kuramoto_i.N; ijk++) {
			out_string = std::to_string(x[ijk]) + "\t";
			write_ofstream(out_string);
		}
		out_string = std::to_string(Kuramoto_i.r_real) + "\t" + std::to_string(Kuramoto_i.r_imag) + "\n";
		write_ofstream(out_string);

		GLOBAL::mm = GLOBAL::mm + 1;
	}


}


kuramoto_2nd_order::kuramoto_2nd_order(std::ofstream* file_name, kuramoto_system G) : data_file(*file_name), Kuramoto_i(G){};

kuramoto_mixed_order::kuramoto_mixed_order(std::ofstream* file_name, kuramoto_system G) : data_file(*file_name), Kuramoto_i(G){};

kuramoto_1st_order::kuramoto_1st_order(std::ofstream* file_name, kuramoto_system G) : data_file(*file_name), Kuramoto_i(G){};


void kuramoto_system::add_iter_indx()
{
	this -> iter_indix += 1;
}
void kuramoto_system::overload_iter_indx()
{
	this -> iter_indix = 0;
}



void kuramoto_2nd_order::add_iter_indx()
{
	this -> iter_indix += 1;
}
void kuramoto_2nd_order::overload_iter_indx()
{
	this -> iter_indix = 0;
}

void kuramoto_1st_order::add_iter_indx()
{
	this -> iter_indix += 1;
}
void kuramoto_1st_order::overload_iter_indx()
{
	this -> iter_indix = 0;
}

void kuramoto_mixed_order::add_iter_indx()
{
	this -> iter_indix += 1;
}
void kuramoto_mixed_order::overload_iter_indx()
{
	this -> iter_indix = 0;
}



std::string kuramoto_2nd_order::get_type() {
	return this -> type;
}

kuramoto_system& kuramoto_2nd_order::get_system() {
	return this -> Kuramoto_i;
}

void kuramoto_2nd_order::write_ofstream(std::string da_line) 
{
	(this -> data_file) << da_line;
};

void kuramoto_2nd_order::close_ofstream() 
{
	(this -> data_file).close();
};



std::string kuramoto_mixed_order::get_type() {
	return this -> type;
}

kuramoto_system& kuramoto_mixed_order::get_system() {
	return this -> Kuramoto_i;
}


void kuramoto_mixed_order::write_ofstream(std::string da_line) 
{
	(this -> data_file) << da_line;
};

void kuramoto_mixed_order::close_ofstream() 
{
	(this -> data_file).close();
};



std::string kuramoto_1st_order::get_type() {
	return this -> type;
}

kuramoto_system& kuramoto_1st_order::get_system() {
	return this -> Kuramoto_i;
}


void kuramoto_1st_order::write_ofstream(std::string da_line) 
{
	(this -> data_file) << da_line;
};

void kuramoto_1st_order::close_ofstream() 
{
	(this -> data_file).close();
};




const std::vector<std::string> explode(const std::string& s, const char& c)
{
	std::string buff{""};
	std::vector<std::string> v;
	
	for(auto n:s)
	{
		if(n != c) buff+=n; else
		if(n == c && buff != "") { v.push_back(buff); buff = ""; }
	}
	if(buff != "") v.push_back(buff);
	
	return v;
}


kuramoto_system get_parameters(std::ifstream &params_file, kuramoto_system Kuramoto_i)
{
	double indk = 0;
	int n_nodes = 0.0, n_links = 0.0, ni, nj;
	std::string line;
	if (params_file.is_open())
	{
		while ( getline (params_file,line) )
		{
			std::vector<std::string> params_divided{explode(line, ' ')};
			if (indk == 0){
				n_nodes = std::strtof((params_divided[0]).c_str(),0);
				n_links = std::strtof((params_divided[1]).c_str(),0);
				Kuramoto_i.N = n_nodes;
				state_type P_first (Kuramoto_i.N);
				state_type Node_type_first (Kuramoto_i.N);
				state_type P_disturbed_first (Kuramoto_i.N);
				state_type P_initial_first (Kuramoto_i.N);
				state_type alf_first (Kuramoto_i.N);
				matrix<double> K_first (Kuramoto_i.N,Kuramoto_i.N);
				matrix<double> K_aux_first (Kuramoto_i.N,Kuramoto_i.N);
				matrix<double> Gamm_first (Kuramoto_i.N,Kuramoto_i.N);

				Kuramoto_i.P = P_first;
				Kuramoto_i.Node_type = Node_type_first;
				Kuramoto_i.P_disturbed = P_disturbed_first;
				Kuramoto_i.P_initial = P_initial_first;
				Kuramoto_i.alf = alf_first;
				Kuramoto_i.K = K_first;
				Kuramoto_i.K_aux = K_aux_first;
				Kuramoto_i.Gamm = Gamm_first;

			}

			else if ( (indk < 2 + n_links) && (indk > 1) ){
				ni = std::strtof((params_divided[0]).c_str(),0);
				nj = std::strtof((params_divided[1]).c_str(),0);
				Kuramoto_i.K(ni,nj) = std::strtof((params_divided[2]).c_str(),0);
				Kuramoto_i.K_aux(ni,nj) = std::strtof((params_divided[2]).c_str(),0);}

			else if ( (indk < 3 + n_links + n_nodes) && (indk > 2 + n_links) ){
				ni = std::strtof((params_divided[0]).c_str(),0);
				Kuramoto_i.Node_type[ni] = std::strtof((params_divided[1]).c_str(),0);
				Kuramoto_i.P[ni] = std::strtof((params_divided[2]).c_str(),0);
				Kuramoto_i.P_initial[ni] = Kuramoto_i.P[ni];
				Kuramoto_i.P_disturbed[ni] = std::strtof((params_divided[3]).c_str(),0);}

			else if ( (indk > 3 + n_links + n_nodes) && (indk < 4 + n_links + 2*n_nodes) ){
				ni = std::strtof((params_divided[0]).c_str(),0);
				Kuramoto_i.alf[ni] = std::strtof((params_divided[1]).c_str(),0);}

			else if ( indk > 4 + n_links + 2*n_nodes){
				ni = std::strtof((params_divided[0]).c_str(),0);
				nj = std::strtof((params_divided[1]).c_str(),0);
				Kuramoto_i.Gamm(ni,nj) = std::strtof((params_divided[2]).c_str(),0);
			}

			indk = indk + 1;
		}
		params_file.close();
	}
	return Kuramoto_i;

}

state_type get_initial_state(std::ifstream &initstate_file, kuramoto_system Kuramoto_i)
{
	std::vector<double> y;
	std::string line;
	if (initstate_file.is_open())
	{
		while ( getline (initstate_file,line) )
		{
			y.push_back(std::strtof((line).c_str(),0));
		}
		initstate_file.close();
	}
	int M = y.size();
	state_type x (M);
	for( int i = 0; i < M; i = i + 1 ) {
		x[i] = y[i];}
	return x;
}


kuramoto_system get_simulation_settings(std::ifstream &sim_settings_file, kuramoto_system Kuramoto_i)
{
	std::string line;
	int indik = 0;
	if (sim_settings_file.is_open())
	{
		while ( getline (sim_settings_file,line) )
		{
			std::vector<std::string> settings_divided{explode(line, ' ')};

			if (indik == 0){
				Kuramoto_i.tini = std::strtof((settings_divided[1]).c_str(),0);
			}
			else if (indik == 1){
				Kuramoto_i.tfin = std::strtof((settings_divided[1]).c_str(),0);
			}
			else if (indik == 2){
				Kuramoto_i.W = std::strtof((settings_divided[1]).c_str(),0);
			}
			else if (indik == 3){
				Kuramoto_i.tstep = std::strtof((settings_divided[1]).c_str(),0);
				Kuramoto_i.tw = Kuramoto_i.W*Kuramoto_i.tstep;
			}
			else if (indik == 4){
				Kuramoto_i.t_disturb = std::strtof((settings_divided[1]).c_str(),0);
			}
			else if (indik == 5){
				Kuramoto_i.t_recover = std::strtof((settings_divided[1]).c_str(),0);
			}
			else if (indik == 6){
				Kuramoto_i.k_ini = std::strtof((settings_divided[1]).c_str(),0);
			}
			else if (indik == 7){
				Kuramoto_i.k_fin = std::strtof((settings_divided[1]).c_str(),0);
			}
			else if (indik == 8){
				Kuramoto_i.k_step = std::strtof((settings_divided[1]).c_str(),0);
			}
			else if (indik == 9){
				Kuramoto_i.input_network_file = settings_divided[1];
			}
			else if (indik == 10){
				Kuramoto_i.input_initstate_file = settings_divided[1];
			}
			else if (indik == 11){
				Kuramoto_i.model = settings_divided[1];
			}
			indik = indik + 1;
		}

		sim_settings_file.close();
	}
	return Kuramoto_i;
}


void kuramoto_system::update_K_matrix(double ka) 
{
	 K = ka*K_aux; 
}
