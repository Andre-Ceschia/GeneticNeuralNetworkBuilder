#pragma once
#include <torch/torch.h>
#include <cmath>

#include "BatchGen.h"

struct Model : torch::nn::Module {

	torch::nn::Sequential mlist;
	int totalLayers;

	Model(int inShape, int outShape, int* hiddenLayers, int numHiddenLayers);

	torch::Tensor forward(torch::Tensor x);

};

float TrainValidateModel(Model* modelPtr, BatchGenerator gen, float lr, int epochs);

float Validate(Model* modelPtr, torch::Tensor features, torch::Tensor target);
