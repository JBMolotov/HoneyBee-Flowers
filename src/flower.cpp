#include "flower.h"

Flower::Flower()
{

}

Flower::Flower(float x, float y, float nectar):
	_x(x), _y(y), _size(0.01f), _nectar(nectar)
{

}

Flower::~Flower()
{

}

__host__ __device__ float Flower::getNectar(float random) const
{
	float nectar = (random-0.5f)*1.0;	
	return nectar;
}

__host__ __device__ void Flower::getPosition(float *x, float *y, float *size)
{
	*x = _x;
	*y = _y;
	*size = _size;
}

void Flower::draw()
{
	float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;
	glColor3f(1 , 0, 1);
	glBegin(GL_POLYGON);
	{
		float sizeX = _size;
		float sizeY = _size*ratio;
		glVertex2d(_x+sizeX, _y);
		glVertex2d(_x , _y-sizeY);
		glVertex2d(_x-sizeX, _y);
		glVertex2d(_x , _y+sizeY);
	}
	glEnd();
}
