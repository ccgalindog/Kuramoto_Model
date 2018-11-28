gcc := c++
flags := -O3 -Wall -ffast-math -std=c++11
boost := /home/cristian/boost_1_68_0


all: build run

build: main_kuramoto_grid_0

main_kuramoto_grid_0: main_kuramoto.cpp
	$(gcc) $(flags) -I $(boost) -c kuramoto_functions.cpp 
	$(gcc) $(flags) -I $(boost) kuramoto_functions.o main_kuramoto.cpp -o main_kuramoto_grid_0

run:
	time ./main_kuramoto_grid_0>Results/Testing_out.txt Sim_Settings/set_case9_sp_.txt

clean:
	rm -rf *.png *.jpg 