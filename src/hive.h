#ifndef HIVE_H
#define HIVE_H

#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <math.h>
#include "defines.h"
#include "parameters.h"
#include "bee.h"
#include "utils.h"
#include "flower.h"

// Cuda
#include <curand.h>
#include <curand_kernel.h>

class Hive 
{
	public:
		Hive(float x, float y, double* gene, float r, float g, float b, int qtyBees);
		~Hive();

		void reset(float x, float y, double* gene);
		void setGene(double* gene) { _gene = gene; }
		void setFlowers(Flower* flowers, int qtyFlowers);
		// void updateConsensus();
		int getQtyBees() const { return _qtyBees; }
		// int* getConsensus() const { return _consensus; }
		// float getFitness();
		float getNectar() { return _nectar; }
		double* getGene() const { return _gene; }
		float getColor(int color);
		std::string toString();

		void draw();
		void run(int steps);
	
		//Bee can acess
		float _nectar;
	private:
		// Gene
		double* _gene;
		// 0 -> _randomChance;// Chance search new flower
		// 1 -> _followChance;// Chance follow other bee
		// 2 -> _linearDecay; // Linear supporting decay (0-1)
		// 3 -> _danceForceExponent; // dance force = (flower goodness)^danceForceExponent (0-1) mapped to (0-10)
		
		// Hive info
		float _x;
		float _y;
		float _size;
		float _r,_g,_b;
		
		//bees
		const int _qtyBees;
		Bee* _bees;
		Bee* _beesCuda;
		enum State { REST, SEARCH_NEW_FLOWER, FIND_FLOWER, BACK_TO_HOME, DANCE };

		// Flowers
		int _qtyFlowers;
		Flower* _flowers;
		Flower* _flowersCuda;

		// Consensus
		float _fitness;
		int* _consensus;
		// Choice probability (used by bees to select who to follow)
		float* _choiceProb;
		float* _choiceProbCuda;


		// Cuda
		curandState* _cuState;  
};
#endif// HIVE_H
