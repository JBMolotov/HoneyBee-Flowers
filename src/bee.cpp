#include "bee.h"

Bee::Bee()
{

}

Bee::Bee(float x, float y, float theta, float size):
	_x(x), _y(y), _theta(theta), _velocity(3.0f), _size(size), 
	_state(REST),  
	_choice(-1), _danceForce(0)
{
	_load[0] = 0;
}

Bee::~Bee()
{

}

void Bee::setGene(double* gene)
{
	_gene = gene;
	_randomChance = gene[0];
	_followChance = gene[1];
	_linearDecay = gene[2];
	_danceForceExponent = gene[3];
}

void Bee::draw()
{
	float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;
	// Draw Black body 
	glColor3f(0, 0, 0);
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

	// Draw Yellow point 
	switch(_state)
	{
		case REST:
			glColor3f(0.5, 0.5, 0);
			break;
		case SEARCH_NEW_FLOWER:
			glColor3f(1.0, 1.0, 0);
			break;
			case FIND_FLOWER:
			glColor3f(0.0, 0.0, 1.0);
			break;
		case BACK_TO_HOME:
		case DANCE:
			glColor3f(1.0, 0.0, 1.0);
			break;
	}

	glBegin(GL_POLYGON);
	{
		float sizeX = _size*0.5f;
		float sizeY = _size*0.5f*ratio;
		glVertex2d(_x+sizeX, _y);
		glVertex2d(_x , _y+sizeY);
		glVertex2d(_x-sizeX, _y);
		glVertex2d(_x , _y-sizeY);
	}
	glEnd();
}

// __host__ __device__ void Bee::depositNectar(float* hiveNectar)
// {
// 	float batata = 8.0;
// 	*hiveNectar = batata;
// }

__host__ __device__ void Bee::run(float random, float ratio, float hiveX, float hiveY, float* hiveNectar, Flower* flowers, int qtyFlowers, float* choiceProb)
{
	// *hiveNectar = 9;

	// Movement info
	const float angleToHive = atan2(_y-hiveY,_x-hiveX)*180.0/M_PI;
	const float distToHive = sqrt((_x-hiveX)*(_x-hiveX) + (_y-hiveY)*(_y-hiveY));
	const float inHome = 0.06f;
	const float danceRadius = 0.005f;

	const float randX = int(random*2342234)%1000/1000.f;
	
	const float randY = int(random*9321432)%1000/1000.f;
	const float maxRotation = 20;

	float x, y, size;
	float angleToFlower, distToFlower;
	switch(_state)
	{
		case REST:
			if(distToHive>inHome)
			{
				_theta = angleToHive-180;
				_x += _size*_velocity*cos(_theta/180.0*M_PI)*ratio;
				_y += _size*_velocity*sin(_theta/180.0*M_PI);
			}
			else
			{
				_theta += (random*maxRotation-maxRotation/2);
				_x += 0.03f*_size*_velocity*cos(_theta/180.0*M_PI)*ratio;
				_y += 0.03f*_size*_velocity*sin(_theta/180.0*M_PI);
			}

			if(random<_randomChance)
				_state = SEARCH_NEW_FLOWER;
			else if(random<(_randomChance+_followChance))// Some bee already chosen
			{
				float sum = 0;
				// Choose which bee to follow
				for(int i=0; i<qtyFlowers; i++)
				{
					sum += choiceProb[i];
					if(random<=sum)
					{
						_choice = i;
						break;
					}
				}
				if(_choice == -1)// Should never enter this if
					break;

				_state = FIND_FLOWER;
			}

			break;
		case SEARCH_NEW_FLOWER:
			if(_x>1 || _x<-1 || _y>1 || _y<-1)
				_theta = angleToHive-180;
			_theta += (random*maxRotation-maxRotation/2);
			_x += _size*_velocity*cos(_theta/180.0*M_PI);
			_y += _size*_velocity*sin(_theta/180.0*M_PI)*ratio;

			for(int i=0; i<qtyFlowers; i++)
			{
				flowers[i].getPosition(&x, &y, &size);
				distToFlower = sqrt((_x-x)*(_x-x) + (_y-y)*(_y-y));

				if(distToFlower<=size*2)
				{
					_state = BACK_TO_HOME;
					_choice = i;
					_danceForce = flowers[i].getNectar(random);
				}
			}

			break;
		case FIND_FLOWER:
			flowers[_choice].getPosition(&x, &y, &size);

			angleToFlower = atan2(_y-y,_x-x)*180.0/M_PI;
			distToFlower = sqrt((_x-x)*(_x-x) + (_y-y)*(_y-y));

			_theta = angleToFlower-180;
			_theta += (random*maxRotation-maxRotation/2);
			_x += _size*_velocity*cos(_theta/180.0*M_PI);
			_y += _size*_velocity*sin(_theta/180.0*M_PI)*ratio;

			if(distToFlower<size*2)
			{
				// _choiceNectar = flowers[_choice].getNectar(random);
				_load[0] += flowers[_choice].getNectar(random);
				_danceForce = pow(_choice, _danceForceExponent*10);
				_state = BACK_TO_HOME;
			}

			_load[0] = _load[0] > _load[1] ? _load[1] : _load[0]; 

			break;
		case BACK_TO_HOME:
			_theta = angleToHive-180;
			_x += _size*_velocity*cos(_theta/180.0*M_PI);
			_y += _size*_velocity*sin(_theta/180.0*M_PI)*ratio;

			if(distToHive<inHome)
			{
				_state = FIND_FLOWER;
				//_state = DANCE;
				
				// *hiveNectar = 0.1;
				//printf("%f ", );
				// nectar += _load[0];
				// _load[0] = 0;
			}
			
			break;
		case DANCE:
			if(distToHive>inHome)
				_theta = angleToHive-180;
			else
				_theta += (random*maxRotation-maxRotation/2);

			_x += 0.3f*_size*_velocity*cos(_theta/180.0*M_PI);
			_y += 0.3f*_size*_velocity*sin(_theta/180.0*M_PI)*ratio;
			_x += 2*randX*danceRadius-danceRadius;
			_y += (2*randY*danceRadius-danceRadius)*ratio;

			_danceForce -= _linearDecay;

			if(_danceForce<0)
			{
				_choice = -1;
				// _choiceNectar = 0;
				_danceForce = 0;
				_state = REST;
			}

			break;
	}
}