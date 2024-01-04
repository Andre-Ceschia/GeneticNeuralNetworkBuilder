#include "Genetic.h"

#define CHILDREN_PER 2

Genetic::Genetic(std::string trainFile, std::string validationFile, int populationSize, int inShape, int epochs, float lr, int batchSize, int maxLayers, int maxNeurons, float dropoutP, float mutationProb, int generations)
	:chromosones(nullptr), best(nullptr), gen(trainFile, validationFile, batchSize), inShape(inShape), outShape(1), epochs(epochs), lr(lr), maxLayers(maxLayers), maxNeurons(maxNeurons), dropoutP(dropoutP), mutationProb(mutationProb), generations(generations), device(gen.GetDevice())
{

	int newParents, newPop;

	adjustPopulation(populationSize, CHILDREN_PER, &newParents, &newPop);

	this->parents = newParents;
	this->populationSize = newPop;

	chromosones = new Chromosone[newPop];

	srand((unsigned)time(NULL));

	for (int i = 0; i < newPop; i++) {
		chromosones[i] = RandomChromosone();
	}

}

Genetic::~Genetic() {
	if (best) {

		std::cout << std::endl;
		ChromsonePP(*best);

		delete[] best->layers;

		delete best;
	}

	if (chromosones) {
		for (int i = 0; i < populationSize; i++) {
			delete[] chromosones[i].layers;
		}

		delete[] chromosones;
	}


}

void Genetic::Train() {

	float* means = new float[generations];
	float* variances = new float[generations];



	for (int currGen = 0; currGen < generations; currGen++) {


		int currIndex = 0;
		int amountToTrain = populationSize;


		if (currGen != 0) {
			currIndex = parents;
			amountToTrain -= parents;
		}
		// initial thread init

		int finChromosones = 0;

		for (int i = currIndex; i < populationSize; i++) {

			TrainEvalChromosone(chromosones + i, gen, inShape, outShape);

			finChromosones++;

			ProgressBar((float)finChromosones / amountToTrain, currGen + 1);

		}

		std::cout << '\r';

		std::sort(chromosones, chromosones + populationSize, [](const Chromosone& a, const Chromosone& b) -> bool
		{
		
				return a.fitness < b.fitness;
		});

		float minLoss = chromosones[0].fitness;
		float maxLoss = chromosones[populationSize-1].fitness;

		float average = 0;

		for (int i = 0; i < populationSize; i++) {
			average += chromosones[i].fitness;
		}

		average /= populationSize;

		float variance = 0;

		for (int i = 0; i < populationSize; i++) {
			variance += pow(chromosones[i].fitness - average, 2);
		}

		variance = variance / (populationSize - 1);

		means[currGen] = average;
		variances[currGen] = variance;

		//print stats and everyting here

		float stats[9] = { currGen + 1, average, minLoss, maxLoss, sqrt(variance) };

		if (currGen == 0) {

			for (int i = 5; i < 9; i++) {
				stats[i] = -1000;
			}

		}
		else {
			float avgChange = (average - means[currGen - 1]) / means[currGen - 1];
			avgChange *= 100;

			float initialAvgChange = (average - means[0]) / means[0];
			initialAvgChange *= 100;

			float pValue = CalculatePValue(average, means[currGen - 1], variance, variances[currGen - 1], populationSize, populationSize);
			float initalPValue = CalculatePValue(average, means[0], variance, variances[0], populationSize, populationSize);

			stats[5] = avgChange;
			stats[6] = pValue;
			stats[7] = initialAvgChange;
			stats[8] = initalPValue;


		}

		PrintStats(currGen == 0, stats);

		// crossover

		Chromosone* newGen = new Chromosone[populationSize];


		for (int writeIndex = 0; writeIndex < parents; writeIndex++) {

			newGen[writeIndex] = chromosones[writeIndex];

		}

		int writeIndex = parents;

		for (int i = 0; i < parents; i++) {

			// this loop will not execute on the last iteration of the outer loop
			for (int j = i + 1; j < parents; j++) {

				Chromosone c1, c2;

				UniformCrossover(chromosones[i], chromosones[j], &c1, &c2, 0.5);

				newGen[writeIndex] = c1;
				writeIndex++;

				newGen[writeIndex] = c2;
				writeIndex++;

			}


		}

		// mutating

		// DONT MUTATE PARENTS
		for (int i = parents; i < populationSize; i++) {
			float prob = (float)rand() / RAND_MAX;

			if (prob < mutationProb) {
				Chromosone* curr = newGen + i;

				UniformMutation(curr, 0.5);

			}
		}

		// cleaing up memoery

		// cant delete the memory for the layer arrays in parents since they are puyt back into the population
		for (int i = parents; i < populationSize; i++) {

			delete[] chromosones[i].layers;

		}

		delete[] chromosones;

		chromosones = newGen;

		if (currGen != 0) {
			delete best;
			delete[] best->layers;
		}

		best = new Chromosone;

		best->fitness = newGen[0].fitness;
		best->numHiddenLayers = newGen[0].numHiddenLayers;

		best->layers = new int[maxLayers];

		std::memcpy(best->layers, newGen[0].layers, sizeof(int) * maxLayers);


	}

	delete[] means;
	delete[] variances;

	for (int i = 0; i < populationSize; i++) {
		delete[] chromosones[i].layers;
	}

	delete[] chromosones;

	chromosones = nullptr;

	std::cout << std::endl;


}

void Genetic::TrainEvalChromosone(Chromosone* ch, BatchGenerator gen, int inShape, int outShape) {

	Model* modelPtr = new Model(inShape, outShape, ch->layers, ch->numHiddenLayers);
	modelPtr->to(device);

	float loss = TrainValidateModel(modelPtr, gen, lr, epochs);

	delete modelPtr;


	ch->fitness = loss;

	//*fin = true;

	return;


}

void Genetic::UniformCrossover(Chromosone p1, Chromosone p2, Chromosone* c1Ptr, Chromosone* c2Ptr, float crossProb) {

	int iterations;

	if (p1.numHiddenLayers > p2.numHiddenLayers) {
		iterations = p1.numHiddenLayers;
	}
	else {
		iterations = p2.numHiddenLayers;

	}

	Chromosone* chArr = new Chromosone[CHILDREN_PER];

	for (int i = 0; i < 2; i++) {

		Chromosone currCh;

		int* initalLayers = new int[maxLayers];

		for (int j = 0; j < iterations; j++) {

			float prob = (float)rand() / RAND_MAX;

			if (prob < crossProb) {
				initalLayers[j] = p1.layers[j];
			}
			else {
				initalLayers[j] = p2.layers[j];
			}

		}

		for (int j = iterations; j < maxLayers; j++) {
			initalLayers[j] = 0;
		}

		int* newLayers = new int[maxLayers];

		int readIndex = 0;
		int writeIndex = 0;

		bool lastNeg = false;

		while (readIndex < maxLayers) {
			if (initalLayers[readIndex] == 0) {
				readIndex++;
			}
			else if (initalLayers[readIndex] < 0 && (lastNeg || writeIndex == 0)) {
				readIndex++;
			}
			else {
				newLayers[writeIndex] = initalLayers[readIndex];
				if (initalLayers[readIndex] < 0) {
					lastNeg = true;
				}
				else {
					lastNeg = false;
				}

				writeIndex++;
				readIndex++;
			}
		}

		for (int j = writeIndex; j < maxLayers; j++) {
			newLayers[j] = 0;
		}

		delete[] initalLayers;

		if (newLayers[writeIndex - 1] < 0) {
			newLayers[writeIndex - 1] = 0;
			writeIndex--;
		}


		currCh.numHiddenLayers = writeIndex;
		currCh.layers = newLayers;


		chArr[i] = currCh;


	}

	*c1Ptr = chArr[0];
	*c2Ptr = chArr[1];

	delete[] chArr;


}

// this mutation is just creating a random chromsone and crossing over with the chromosone to be mutated,
// i found this to have the best results compared to the mutation algorthims that just rearrange genes
void Genetic::UniformMutation(Chromosone* ch, float prob) {
	Chromosone randCh = RandomChromosone();

	Chromosone c1, c2;

	UniformCrossover(*ch, randCh, &c1, &c2, prob);

	delete[] c2.layers;

	ch->layers = c1.layers;
	ch->numHiddenLayers = c1.numHiddenLayers;


}

Chromosone Genetic::RandomChromosone() {

	int* layerArr = new int[maxLayers];


	int layers = 1 + (rand() % maxLayers);

	float epsilon = pow(10, -10);

	bool lastDrop = false;

	for (int i = 0; i < layers; i++) {
		if (i != 0 && i != (layers - 1) && dropoutP > epsilon) {

			float dropProb = ((float)(rand() % 101) / 100);

			if (dropProb <= dropoutP && !lastDrop) {

				layerArr[i] = -1 * (5 + (rand() % 80));
				lastDrop = true;

			}
			else {

				layerArr[i] = 1 + (rand() % maxNeurons);
				lastDrop = false;
			}


		}
		else {

			layerArr[i] = 1 + (rand() % maxNeurons);
		}
	
	}

	for (int i = layers; i < maxLayers; i++) {
		layerArr[i] = 0;
	}

	Chromosone ch;

	ch.numHiddenLayers = layers;
	ch.layers = layerArr;

	return ch;

}


// let k = children per crossover
// let u = # of parents that are crossing over
// let n = total population size
// then
// k * (uC2) + u = n (note uC2 is u choose 2)
// this equation says all of the children plus the parents must add up to be equal to the initail populatin size
//the two functions below this are the solution to the above equation
int Genetic::CalcParents(int childrenPer, int population) {
	float discriminant = pow(1 - ((float)childrenPer / 2), 2) + 2 * childrenPer * population;

	float parents = (((float)childrenPer / 2) - 1 + sqrt(discriminant)) / childrenPer;

	return (int)ceil(parents);

}

int Genetic::CalcPopulation(int childrenPer, int parents) {
	float newPop = (((float)childrenPer / 2) * parents * (parents - 1) + parents);

	return (int)newPop;

}

void Genetic::adjustPopulation(int initPopulation, int childrenPer, int* parentsPtr, int* newPopulationPtr) {
	int parents = CalcParents(childrenPer, initPopulation);
	int newPopulation = CalcPopulation(childrenPer, parents);

	*parentsPtr = parents;
	*newPopulationPtr = newPopulation;

}


// test statistic with null hypothesis mu1 = mu2
float Genetic::GetTestStatistic(float sample1Mean, float sample2Mean, float sample1Variance, float sample2Variance, int sample1Size, int sample2Size) {

	float expectedValue = sample1Mean - sample2Mean;

	float variance = (sample1Variance / sample1Size) + (sample2Variance / sample2Size);

	return expectedValue / sqrt(variance);


}

float Genetic::CalculatePValue(float sample1Mean, float sample2Mean, float sample1Variance, float sample2Variance, int sample1Size, int sample2Size) {
	int degreesOfFreedom = sample1Size + sample1Size - 2;

	float testStat = abs(GetTestStatistic(sample1Mean, sample2Mean, sample1Variance, sample2Variance, sample1Size, sample2Size));


	if (degreesOfFreedom < 30) {

		boost::math::students_t mydist(degreesOfFreedom);

		return 2 *  (1 - boost::math::cdf(mydist, testStat));
		
	}
	else {

		boost::math::normal mydist(0, 1);

		return 2 * (1 - boost::math::cdf(mydist, testStat));

	}

}

void Genetic::PrintBest() {
	if (best) {
		ChromsonePP(*best);
	}
	else {
		std::cout << "Best Chromosone does not exist" << std::endl;
	}
}


void Genetic::ChromsonePP(Chromosone ch) {

	std::cout << "Neuron Configuration: ";

	for (int i = 0; i < ch.numHiddenLayers; i++) {
		if (ch.layers[i] < 0) {
			std::cout << ch.layers[i] * -1 << "% ";
		}
		else {
			std::cout << ch.layers[i] << " ";
		}
	}

	std::cout << std::endl;

	std::cout << "Loss: " << ch.fitness << std::endl;


}

void Genetic::PrintStats(bool heading, float* data) {

	std::string line(178, '-');

	const int wordAmount = 9;

	std::string words[wordAmount] = {"Generation", "Average Loss", "Min Loss", "Max Loss", "Standard Deviation", "Average Change", "P-Value", "Average Change (Gen 1)", "P-Value (Gen 1)", };

	if (heading) {


		std::string spaces(3, ' ');

		std::cout << line << std::endl;
		std::cout << "|";

		for (int i = 0; i < wordAmount; i++) {
			std::cout << spaces << words[i] << spaces << "|";

		}
		std::cout << std::endl;
		std::cout << line << std::endl;

	}


	std::cout << "|";

	for (int i = 0; i < wordAmount; i++) {
		int space = words[i].length() + 6;

		std::string val;

		if (i == 0) {
			val = std::to_string((int)data[i]);

		} else if (i == 6 || i == 8) {

			if (data[i] < -999) {
				val = "N/A";
			}
			else {

				char buf[40];
				sprintf(buf, "%.5f", data[i]);

				val = buf;
			}

		} else if (i == 5 || i == 7) {

			if (data[i] < -999) {
				val = "N/A";
			}
			else {

				char buf[40];
				sprintf(buf, "%.2f%%", data[i]);

				val = buf;
			}

		} else {

			char buf[40];
			sprintf(buf, "%.2f", data[i]);

			val = buf;
		}


		int rem = space - val.length();

		int leftSpace, rightSpace;

		if (rem % 2 == 1) {

			leftSpace = (int)((float)rem / 2) + 1;
			rightSpace = (int)((float)rem / 2);

		}
		else {

			leftSpace = rem / 2;
			rightSpace = rem / 2;

		}

		std::string left(leftSpace, ' ');
		std::string right(rightSpace, ' ');

		std::cout << left << val << right << "|";

	}

	std::cout << std::endl;

	std::cout << line << std::endl;


}

void Genetic::ProgressBar(float progress, int currGen) {

	std::cout << '\r';

	const int BARS = 50;

	int prog = (int)std::round(progress * 100);

	int progBars = (int)std::round((float)prog/2);


	std::string out = "Generation: " + std::to_string(currGen) + "/" + std::to_string(generations) + " | "  +"[";

	out += std::string(progBars, '#');
	out += std::string(BARS - progBars, ' ');

	if (prog == 100 && progress > prog) {
		prog = 99;
	}

	out += "] " + std::to_string(prog) + "%";


	std::cout << out;



}
