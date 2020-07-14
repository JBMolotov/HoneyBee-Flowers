#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "defines.h"
#include "parameters.h"
#include "hive.h"
#include "data.h"
#include "flower.h"

class Environment 
{
	public:
		Environment(Data* data);
		~Environment();

		void draw();
		void plotConsensus(); //honey amount on hive
		void plotGeneration();
		void run(int steps);
	private:
		std::vector<Hive*> _hives;
		std::vector<std::vector<float>> _generationFitness;
		std::vector<std::vector<float>> _repetitionFitness;
		Flower* _flowers;
		Data* _data;

		int _generation;
		int _step;
		int _repetition;

		int _stepsOffline; 
		int _stepsPerRepetition; 
		int _repetitionsPerGeneration; 
		int _qtyBees; 
		int _qtyHives;
		int _qtyFlowers;
};
#endif// ENVIRONMENT_H
