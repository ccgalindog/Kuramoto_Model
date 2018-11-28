#ifndef KURAMOTO
#define KURAMOTO

#include "global_variables.h"

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


using namespace boost::numeric::odeint;
using namespace boost::numeric::ublas;


typedef boost::numeric::ublas::vector< double > state_type;

template <typename CALLABLE>
matrix<double> apply_elementwise(const CALLABLE& f, matrix<double>& a)
{
   matrix<double> result(a.size1(), a.size2());
   std::transform(a.data().begin(), a.data().end(), result.data().begin(), f);
   return result;
}

template <typename CALLABLE>
vector<double> apply_elementwise_vec(const CALLABLE& f, vector<double>& a)
{
   vector<double> result(a.size());
   std::transform(a.data().begin(), a.data().end(), result.data().begin(), f);
   return result;
}

struct kuramoto_system
{
public:
	state_type P;
	state_type P_disturbed;
	state_type P_initial;
	state_type Node_type;
	state_type alf;
	matrix<double> K;
	matrix<double> K_aux;
	matrix<double> Gamm;
	double r_real;
	double r_imag;
    int N;
    int W;
	int damage_key;
	std::string input_network_file;
	std::string input_initstate_file;
	std::string model;
	double tini;
	double tfin;
	double tstep;
	double tw;
	double t_disturb;
	double t_recover;
	double k_ini;
	double k_fin;
	double k_step;
	double P_balance;
	int iter_indix;
	kuramoto_system();
	void update_K_matrix(double ka);
	void add_iter_indx();
	void overload_iter_indx();


};


class kuramoto_2nd_order 
{
public:
	std::ofstream& data_file;
	kuramoto_system Kuramoto_i;
	std::string type;
	int iter_indix;	
	kuramoto_2nd_order(std::ofstream* file_name, kuramoto_system G);    
	void operator() ( const state_type &x , state_type &dxdt , const double t);
	void write_ofstream(std::string da_line);
	void close_ofstream();
	std::string get_type();
	kuramoto_system& get_system();
	void add_iter_indx();
	void overload_iter_indx();
};

class kuramoto_mixed_order 
{
public:
	std::ofstream& data_file;
	kuramoto_system Kuramoto_i;
	std::string type;
	int iter_indix;
	kuramoto_mixed_order(std::ofstream* file_name, kuramoto_system G);
	void operator() ( const state_type &x , state_type &dxdt , const double t);
	void write_ofstream(std::string da_line);
	void close_ofstream();
	std::string get_type();
	kuramoto_system& get_system();
	void add_iter_indx();
	void overload_iter_indx();
};


class kuramoto_1st_order 
{
public:
	std::ofstream& data_file;
	kuramoto_system Kuramoto_i;
	std::string type;
	int iter_indix;
	kuramoto_1st_order(std::ofstream* file_name, kuramoto_system G);
	void operator() ( const state_type &x , state_type &dxdt , const double t);
	void write_ofstream(std::string da_line);
	void close_ofstream();
	std::string get_type();
	kuramoto_system& get_system();
	void add_iter_indx();
	void overload_iter_indx();
};


const std::vector<std::string> explode(const std::string& s, const char& c);

kuramoto_system get_parameters(std::ifstream &params_file, kuramoto_system Kuramoto_i);

state_type get_initial_state(std::ifstream &initstate_file, kuramoto_system Kuramoto_i);

kuramoto_system get_simulation_settings(std::ifstream &sim_settings_file, kuramoto_system Kuramoto_i);

kuramoto_system update_K_matrix(double ka, kuramoto_system Kuramoto_i); 

template <typename T>
void run_ode_solver(T& dynamical_system, state_type& x, state_type x0, std::string set_file)
{
	kuramoto_system system = dynamical_system.get_system();
	GLOBAL::mm = 0;
	integrate_const( runge_kutta4< state_type >(), dynamical_system , x , system.tini , system.tfin , system.tstep);

}

#endif