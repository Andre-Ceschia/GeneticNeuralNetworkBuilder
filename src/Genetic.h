#pragma once

#include <cmath>
#include <torch/torch.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>

#include "BatchGen.h"
#include "NeuralNetwork.h"

// NEURONS ARE ON THE HEAP 
struct Chromosone {

	int numHiddenLayers;
	int* layers;
	float fitness;

};

class Genetic {
private:

	Chromosone* chromosones;
	Chromosone* best;
	BatchGenerator gen;

	int inShape;
	int outShape;
	int epochs;
	float lr;
	int maxLayers;
	int maxNeurons;
	int populationSize;
	int parents;

	// probablity that a layer is a dropout layer
	float dropoutP;

	float mutationProb;

	int generations;

	torch::Device device;

	void TrainEvalChromosone(Chromosone* chromosone, BatchGenerator gen, int inShape, int outShape);

	int CalcParents(int childrenPer, int population);

	int CalcPopulation(int childrenPer, int parents);

	void adjustPopulation(int initPopulation, int childrenPer, int* parents, int* newPopulation);

	void ProgressBar(float progress, int currGen);

	void PrintStats(bool heading, float* data);

	void UniformCrossover(Chromosone p1, Chromosone p2, Chromosone* c1Ptr, Chromosone* c2Ptr, float crossProb);

	void UniformMutation(Chromosone* ch, float prob);

	float GetTestStatistic(float sample1Mean, float sample2Mean, float sample1Variance, float sample2Variance, int sample1Size, int sample2Size);

	float CalculatePValue(float sample1Mean, float sample2Mean, float sample1Variance, float sample2Variance, int sample1Size, int sample2Size);

	void ChromsonePP(Chromosone ch);

	Chromosone RandomChromosone();

public:
	Genetic(std::string trainFile, std::string validationFile, int populationSize, int inShape, int epochs, float lr, int batchSize, int maxLayers, int maxNeurons, float dropoutP, float mutationProb, int generations);

	~Genetic();

	void PrintBest();

	void Train();


};


