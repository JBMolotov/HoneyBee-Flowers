#ifndef FLOWER_H
#define FLOWER_H

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <stdlib.h>
#include "defines.h"

class Flower 
{
	public:
		Flower();
		Flower(float x, float y, float nectar);
		~Flower();

		__host__ __device__ float getNectar(float random) const;
		__host__ __device__ void getPosition(float *x, float *y, float *size);
		float getRealNectar() const { return _nectar; }

		void draw();
	private:
		float _x;
		float _y;
		float _size;
		float _nectar;
		//float _maxnectar;
};
#endif// NEXT_BOX_H
