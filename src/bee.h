#ifndef BEE_H
#define BEE_H

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <math.h>
#include <iostream>
#include "defines.h"
#include "flower.h"

// Cuda
#include <curand.h>
#include <curand_kernel.h>

class Bee 
{
	public:
		Bee();
		Bee(float x, float y, float theta, float size);
		~Bee();

		int getChoice() const { return _choice; }
		enum State { REST, SEARCH_NEW_FLOWER, FIND_FLOWER, BACK_TO_HOME, DANCE };
		State getState() const { return _state; }
		void setGene(double* gene);
		// __host__ __device__ void depositNectar(float* hiveNectar);

		void draw();
		__host__ __device__ void run(float random, float ratio, float hiveX, float hiveY, float* hiveNectar, Flower* Flowers, int qtyFloweres, float* choiceProb);
	private:
		// Gene
		double* _gene;
		double _randomChance;// Chance search new Flower
		double _followChance;// Chance follow other bee
		double _linearDecay; // Linear supporting decay (0-1)
		double _danceForceExponent; // Dance force exponent (0-1) mapped to (0-10)

		// Bee state
		State _state;
		float _x, _y;
		float _theta;
		float _size;
		float _velocity;
		float _age;
		float _load[2]; //amount of load and the max the bee can load
		// float _maxload;
		
		// The flower this bee is collecting
		int _choice; 
		float _danceForce;
		//float _choiceNectar;
		// float _danceForce;

};
#endif// BEE_H
