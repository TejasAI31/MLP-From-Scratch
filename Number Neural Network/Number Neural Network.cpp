#include <iostream>
#include <vector>
#include <fstream>
#include <raylib.h>
#include "Tile.h"

/*
for input/255:
hiddennodes=85
learning rate=0.65

for input=?0:1 :
hiddennodes=70
learning rate=0.35
*/
#define inputnodes 784
#define outputnodes 10
#define hiddenlayers 2
#define hiddennodes 70
#define frac 1
#define displayepochs 0

float learningrate = 0.35;

double inputs[inputnodes];
double hiddenvalues[hiddenlayers][hiddennodes];
double outputvalues[outputnodes];

double hiddenerror[hiddenlayers][hiddennodes];
double outputerror[outputnodes];

double inputweights[inputnodes][hiddennodes];
double hiddenweights[hiddenlayers-1][hiddennodes][hiddennodes];
double outputweights[hiddennodes][outputnodes];

double hiddenbiases[hiddenlayers][hiddennodes];
double outputbiases[outputnodes];

double costs[outputnodes];
double totalcost=0;

using namespace std;

vector<vector<unsigned char>> trainingset;
vector<unsigned char> labels;

Tile tiles[28][28];

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double x)
{
	return (double)(x * (1 - x));
}

void softmax()
{
	double denom = 0;
	for (int x = 0; x < outputnodes; x++)
	{
		denom += exp(outputvalues[x]);
	}

	for (int x = 0; x < outputnodes; x++)
	{
		outputvalues[x] = exp(outputvalues[x]) / (double)denom;
	}
}

void readTrainingData()
{
	ifstream file("../Dataset/train-images.idx3-ubyte",ios::binary);

	vector<unsigned char> header(16);
	file.read((char*)(header.data()), 16);

	for (int x = 0; x < 60000; x++)
	{
		vector<unsigned char> temp(28*28);

		file.read((char*)(temp.data()), 28 * 28);

		trainingset.push_back(temp);
	}
	file.close();
}

void readLabels()
{
	ifstream file("../Dataset/train-labels.idx1-ubyte");

	vector<unsigned char> header(8);
	file.read((char*)(header.data()), 8);

	for (int x = 0; x < 60000; x++)
	{
		vector<unsigned char> temp(1);

		file.read((char*)(temp.data()), 1);

		labels.push_back(temp[0]);
	}
	file.close();
}

void initialiseBiases()
{
	for (int x = 0; x < hiddenlayers; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			hiddenbiases[x][y] = frac*(double)(rand() / (double)RAND_MAX);
			if (rand() % 2 == 0)
				hiddenbiases[x][y] = -hiddenbiases[x][y];
		}
	}

	for (int x = 0; x < outputnodes; x++)
	{
		outputbiases[x] = frac*(double)(rand() / (double)RAND_MAX);
		if (rand() % 2 == 0)
			outputbiases[x] = -outputbiases[x];
	}
}

void initialiseWeights()
{
	for (int x = 0; x < inputnodes;x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			inputweights[x][y] = frac*(double)(rand() / (double)RAND_MAX);
			if (rand() % 2 == 0)
				inputweights[x][y] = -inputweights[x][y];
		}
	}

	for (int x = 0; x < hiddenlayers - 1; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			for (int z = 0; z < hiddennodes; z++)
			{
				hiddenweights[x][y][z] =  frac*(double)(rand() / (double)RAND_MAX);
				if (rand() % 2 == 0)
					hiddenweights[x][y][z] = -hiddenweights[x][y][z];
			}
		}
	}

	for (int x = 0; x < hiddennodes; x++)
	{
		for (int y = 0; y < outputnodes; y++)
		{
			outputweights[x][y] =  frac*(double)(rand() / (double)RAND_MAX);
			if (rand() % 2 == 0)
				outputweights[x][y] = -outputweights[x][y];
		}
	}
}

void forwardPass(int epoch)
{
	if (hiddenlayers == 0)
		return;

	for (int x = 0; x < inputnodes; x++)
	{
		//inputs[x] = (int)trainingset[epoch][x] / 255.0;
		inputs[x] = ((int)trainingset[epoch][x] == 0) ? 0 : 1;
	}

	for (int x = 0; x < hiddennodes; x++)
	{
		double z=0;
		for (int y = 0; y < inputnodes; y++)
		{
			z += inputweights[y][x] * inputs[y];
		}
		hiddenvalues[0][x] = sigmoid(z + hiddenbiases[0][x]);

	}


	for (int x = 1; x < hiddenlayers; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			double r = 0;
			for (int z = 0; z < hiddennodes; z++)
			{
				r += hiddenvalues[x - 1][z] * hiddenweights[x-1][z][y];
			}
			hiddenvalues[x][y] = sigmoid(r + hiddenbiases[x][y]);
		}
	}

	for (int x = 0; x < outputnodes; x++)
	{
		double r=0;
		for (int y = 0; y < hiddennodes; y++)
		{
			r += hiddenvalues[hiddenlayers - 1][y] * outputweights[y][x];
		}
		outputvalues[x] = sigmoid(r + outputbiases[x]);

	}

	//softmax();
}

void backPropogation(int epoch)
{
	totalcost = 0;
	int max = (int)labels[epoch];

	for (int x = 0; x < outputnodes; x++)
	{
		if (x != max)
		{
			costs[x] = -outputvalues[x];
			totalcost+= pow(outputvalues[x],2);
		}
		else
		{
			costs[x] = (1-outputvalues[x]);
			totalcost += pow(1 - outputvalues[x], 2);
		}
	}

	for (int x = 0; x < outputnodes; x++)
	{
		outputerror[x] = costs[x] * dsigmoid(outputvalues[x]);
	}

	for (int x = 0; x < hiddennodes; x++)
	{
		double error = 0;
		for (int y = 0; y < outputnodes; y++)
		{
			error += outputerror[y] * outputweights[x][y];
		}
		hiddenerror[hiddenlayers - 1][x] = dsigmoid(hiddenvalues[hiddenlayers-1][x]) * error;
	}

	for (int x = hiddenlayers - 2; x >= 0; x--)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			double error = 0;
			for (int z = 0; z < hiddennodes; z++)
			{
				error += hiddenerror[x + 1][z] * hiddenweights[x][y][z];
			}
			hiddenerror[x][y] = dsigmoid(hiddenvalues[x][y]) * error;
		}
	}


	//Changes Values of Weights and biases
	for (int x = 0; x < outputnodes; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			outputweights[y][x] += outputerror[x] * hiddenvalues[hiddenlayers - 1][y] * learningrate;
		}
		outputbiases[x] += outputerror[x] * learningrate;
	}

	for (int x = hiddenlayers - 1; x >= 1; x--)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			for (int z = 0; z < hiddennodes; z++)
			{
				hiddenweights[x - 1][z][y] += hiddenerror[x][y] * hiddenvalues[x - 1][z] * learningrate;
			}
			hiddenbiases[x][y] += hiddenerror[x][y] * learningrate;
		}
	}

	for (int x = 0; x < hiddennodes; x++)
	{
		for (int y = 0; y < inputnodes; y++)
		{
			inputweights[y][x] += hiddenerror[0][x] * inputs[y]*learningrate;
		}
		hiddenbiases[0][x] += hiddenerror[0][x] * learningrate;
	}
}

void calculate(int epoch)
{
	for (int x = 0; x < 28; x++)
	{
		for (int y = 0; y < 28; y++)
		{
			tiles[x][y].activated = 0;
		}
	}

	for (int x = 0; x < inputnodes; x++)
	{
		inputs[x] = ((int)trainingset[epoch][x] == 0) ? 0 : 1;
		if(inputs[x]==1)
			tiles[x % 28][x / 28].activated = 1;
	}

	for (int x = 0; x < hiddennodes; x++)
	{
		double z = 0;
		for (int y = 0; y < inputnodes; y++)
		{
			z += inputweights[y][x] * inputs[y];
		}
		hiddenvalues[0][x] = sigmoid(z + hiddenbiases[0][x]);

	}


	for (int x = 1; x < hiddenlayers; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			double r = 0;
			for (int z = 0; z < hiddennodes; z++)
			{
				r += hiddenvalues[x - 1][z] * hiddenweights[x - 1][z][y];
			}
			hiddenvalues[x][y] = sigmoid(r + hiddenbiases[x][y]);
		}
	}

	for (int x = 0; x < outputnodes; x++)
	{
		double r = 0;
		for (int y = 0; y < hiddennodes; y++)
		{
			r += hiddenvalues[hiddenlayers - 1][y] * outputweights[y][x];
		}
		outputvalues[x] = sigmoid(r + outputbiases[x]);

	}
}

void sendData(int casecounter)
{
	calculate(casecounter);
}

int main()
{
	srand(time(NULL));
	readTrainingData();
	readLabels();

	initialiseWeights();
	initialiseBiases();

	int correctcounter = 0;
	int wrongcounter = 0;
	
	cout << "TRAINING....";

	for (int x = 0; x <60000; x++)
	{
		forwardPass(x);

		int max = 0;
		for (int y = 0; y < outputnodes; y++)
		{
			if (outputvalues[y] > outputvalues[max])
				max = y;
		}


		/*if (x % 1000 == 0)
		{
			cout << (correctcounter / (double)(correctcounter + wrongcounter)) * 100 << endl;
			wrongcounter = 0;
			correctcounter = 0;
		}

		if (labels[x] == max)
		{
			correctcounter++;
		}
		else
		{
			wrongcounter++;
		}*/


		backPropogation(x);
	}
	
	

	InitWindow(1200, 784, "Neural Network");
	SetTargetFPS(144);

	int testcounter = 0;

	for (int x = 0; x < outputnodes; x++)
	{
		outputvalues[x] = 0;
	}


	for (int x = 0; x < 28; x++)
	{
		for (int y = 0; y < 28; y++)
		{
			tiles[x][y].activated = 0;
			tiles[x][y].rect.x = (float)x * 28;
			tiles[x][y].rect.y = (float)y * 28;
			tiles[x][y].rect.width = 28;
			tiles[x][y].rect.height = 28;
		}
	}

	Vector2 mousepos;

	while(!WindowShouldClose())
	{
		BeginDrawing();
		ClearBackground(RAYWHITE);

		mousepos = GetMousePosition();

		for (int x = 0; x < 28; x++)
		{
			for (int y = 0; y < 28; y++)
			{
				if(tiles[x][y].activated==0)
					DrawRectangleRec(tiles[x][y].rect, BLACK);
				else
					DrawRectangleRec(tiles[x][y].rect, WHITE);
			}
		}

		for (int x = 0; x < 28; x++)
		{
			DrawLine(x * 28, 0, x * 28, 800, WHITE);
			DrawLine(0, x * 28, 800, x * 28, WHITE);
		}

		if (testcounter % 144 == 0)
		{
			sendData(rand()%60000);
			testcounter = 0;
		}

		testcounter++;


		int max = 0;
		for (int x = 0; x < outputnodes; x++)
		{
			if (outputvalues[x] > outputvalues[max])
				max = x;
		}
		DrawText(TextFormat("%d", max), 950, 100, 80, RED);
		DrawText(TextFormat("0: %f",outputvalues[0]), 900, 250, 20, BLACK);
		DrawText(TextFormat("1: %f", outputvalues[1]), 900, 300, 20, BLACK);
		DrawText(TextFormat("2: %f", outputvalues[2]), 900, 350, 20, BLACK);
		DrawText(TextFormat("3: %f", outputvalues[3]), 900, 400, 20, BLACK);
		DrawText(TextFormat("4: %f", outputvalues[4]), 900, 450, 20, BLACK);
		DrawText(TextFormat("5: %f", outputvalues[5]), 900, 500, 20, BLACK);
		DrawText(TextFormat("6: %f", outputvalues[6]), 900, 550, 20, BLACK);
		DrawText(TextFormat("7: %f", outputvalues[7]), 900, 600, 20, BLACK);
		DrawText(TextFormat("8: %f", outputvalues[8]), 900, 650, 20, BLACK);
		DrawText(TextFormat("9: %f", outputvalues[9]), 900, 700, 20, BLACK);

		EndDrawing();
	}
}