#include "hive.h"

__global__ void initCurand(curandState *state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void runCuda(Bee* bees, int qtyBees, Flower* flowers, int qtyFlowers, curandState *state, float ratio, float hiveX, float hiveY, float* hiveNectar, float* choiceProb)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < qtyBees) { bees[idx].run(curand_uniform(&state[idx]), ratio, hiveX, hiveY, hiveNectar, flowers, qtyFlowers, choiceProb); }
}

Hive::Hive(float x, float y, double* gene, float r, float g, float b, int qtyBees):
		_x(x), _y(y), _size(0.01f), _qtyBees(qtyBees), _gene(gene), _fitness(0), _r(r), _g(g), _b(b), _nectar(0)
{
	_bees = new Bee[_qtyBees];
	
	for(int i=0; i<_qtyBees; i++)
	{
		float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;

		float x = Utils::randomGauss(_x, 0.03);
		float y = Utils::randomGauss(_y, 0.03*ratio);
		float theta = rand()%360;
		float size = 0.005f;

		_bees[i] = Bee(x, y, theta, size);
		_bees[i].setGene(_gene);
	}

	// Cuda

	
	

	cudaMalloc(&_beesCuda, _qtyBees*sizeof(Bee));// Bees on GPU	
	cudaMalloc(&_cuState, _qtyBees*sizeof(curandState));// Rand
	initCurand<<<1+_qtyBees/256, 256>>>(_cuState, time(NULL));
	
}

Hive::~Hive()
{
	cudaFree(_beesCuda);
	//if(_bees != nullptr)// free() invalid pointer error...?
	//	delete _bees;
}

void Hive::reset(float x, float y, double* gene)
{
	_x = x;
	_y = y;
	_gene = gene;
	

	for(int i=0; i<_qtyBees; i++)
	{
		float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;

		float x = Utils::randomGauss(_x, 0.03);
		float y = Utils::randomGauss(_y, 0.03*ratio);
		float theta = rand()%360;
		float size = 0.005f;

		_bees[i] = Bee(x, y, theta, size);
		_bees[i].setGene(_gene);
	}
}

void Hive::setFlowers(Flower* flowers, int qtyFlowers) 
{ 
	// Nest box
	_flowers=flowers; 
	_qtyFlowers=qtyFlowers; 

	cudaMalloc(&_flowersCuda, _qtyFlowers*sizeof(Flower));// Flowers on GPU
	cudaMemcpy(_flowersCuda, _flowers, _qtyFlowers*sizeof(Flower), cudaMemcpyHostToDevice);

	// Consensus
	_consensus = new int[_qtyFlowers];
	_choiceProb = new float[_qtyFlowers];
	cudaMalloc(&_choiceProbCuda, _qtyFlowers*sizeof(float));// Chioce probabilities on GPU
}

/*void Hive::updateConsensus()
{
	int beesWithChoice = 0;

	for(int i=0; i<_qtyFlowers; i++)
	{
		_consensus[i] = 0;
	}

	for(int i=0; i<_qtyBees; i++)
	{
		int choice = _bees[i].getChoice();
		int state = _bees[i].getState();

		if(choice!=-1 && state==DANCE)
		{
			beesWithChoice++;
			_consensus[choice]++;
		}
	}

	// Check consensus reached and update fitness
	for(int i=0; i<_qtyFlowers; i++)
	{
		float consensus = float(_consensus[i])/_qtyBees;
		float g = _flowers[i].getRealNectar();// Nestbox goodness

		_fitness = g*consensus>_fitness? g*consensus : _fitness;

		//if(consensus>0.7)
		//{
		//	std::cout << "Consensus reached!" << consensus << std::endl;
		//	float x, y, size;
		//	_flowers[i].getPosition(&x, &y, &size);
		//	_x = x;
		//	_y = y;
		//}
	}
	//for(int i=0; i<_qtyFlowers; i++)
	//{
	//	printf("%d ", _consensus[i]);
	//}
	//printf("\n");

	// Build choice probabilities
	//float from = 0;
	//for(int i=0; i<_qtyFlowers; i++)
	//{
	//	float to = 0;
	//	if(beesWithChoice>0)
	//		to = _consensus[i]/_qtyBees;
	//	_choiceProb[i] = from + to;
	//	from = _choiceProb[i];
	//}
	
	for(int i=0; i<_qtyFlowers; i++)
	{
		_choiceProb[i] = float(_consensus[i])/_qtyBees;
	}

	cudaMemcpy(_choiceProbCuda, _choiceProb, _qtyFlowers*sizeof(float), cudaMemcpyHostToDevice);
}*/


// float* Hive::getFitness()
// {
// 	//float maxConsensus = 0;
// 	//float nestBoxQuality = 0;
// 	//for(int i=0; i<_qtyFlowers; i++)
// 	//{
// 	//	if(float(_consensus[i])/_qtyBees > maxConsensus)
// 	//	{
// 	//		maxConsensus = float(_consensus[i])/_qtyBees;
// 	//		nestBoxQuality = _flowers[i].getRealGoodness();
// 	//	}
// 	//}
// 	//return maxConsensus*nestBoxQuality;
// 	return _nectar;
// }

float Hive::getColor(int color)
{
	float c;
	switch (color)
	{
		case 1: c = _r; break;
		case 2: c = _g; break;
		case 3: c = _b; break;
	}
	return c;
}

std::string Hive::toString()
{
	std::stringstream sstr; 
	// sstr << "fitness " << 100*(*getFitness()) << " ";
	sstr << _gene[0] << " ";
	sstr << _gene[1] << " ";
	sstr << _gene[2] << " ";
	sstr << _gene[3] << " ";

	return sstr.str();
}

void Hive::draw()
{
	//auto start = std::chrono::high_resolution_clock::now();

	//---------- Draw hive ----------//
	float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;
	glColor3f(_r, _g, _b);
	glBegin(GL_POLYGON);
	{
		float sizeX = _size;
		float sizeY = _size*ratio;
		glVertex2d(_x+sizeX, _y+sizeY);
		glVertex2d(_x+sizeX, _y-sizeY);
		glVertex2d(_x-sizeX, _y-sizeY);
		glVertex2d(_x-sizeX, _y+sizeY);
	}
	glEnd();

	//---------- Draw bees ----------//
	for(int i=0; i<_qtyBees; i++)
	{
		_bees[i].draw();
	}

	//auto finish = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Draw: " << elapsed.count() << "s\n";
}

void Hive::run(int steps)
{
	bool useCuda = true;
	//auto start = std::chrono::high_resolution_clock::now();

	// updateConsensus();
	float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;
	int cycles = steps;

	if(!useCuda)
	{
		int i = 0;
		while(cycles--){
			i++;
			for(int i=0;i<_qtyBees; i++)
			{
				_bees[i].run(rand()%1000/1000.f, ratio, _x, _y, &_nectar, _flowers, _qtyFlowers, _choiceProb);
			}
		}
		// std::cout<<i<<std::endl;
	}
	else
	{
		cudaMemcpy(_beesCuda, _bees, _qtyBees*sizeof(Bee), cudaMemcpyHostToDevice);

		// float* nectar = (float*)malloc(sizeof(float));
		// *nectar = 5;
		float* nectar;
		cudaMalloc(&nectar, sizeof(float)); // Nectar on GPU
		// _nectar = 5;
		cudaMemcpy(nectar, &_nectar, sizeof(float), cudaMemcpyHostToDevice);

		while(cycles--){
			runCuda<<< 1+_qtyBees/256, 256>>>(_beesCuda, _qtyBees, _flowersCuda, _qtyFlowers, _cuState, ratio, _x, _y, nectar, _choiceProbCuda);
			cudaDeviceSynchronize();
		}

		cudaMemcpy(&_nectar, nectar, sizeof(float), cudaMemcpyDeviceToHost); 
  		cudaFree(nectar);

		cudaMemcpy(_bees, _beesCuda, _qtyBees*sizeof(Bee), cudaMemcpyDeviceToHost);
	}

	//auto finish = std::chrono::high_resolution_clock::now();

	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Run: " << elapsed.count() << "s\n";
}

