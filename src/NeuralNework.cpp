#include "NeuralNetwork.h"

Model::Model(int inShape, int outShape, int* hiddenLayers, int numHiddenLayers)
	:mlist(register_module("mlist", torch::nn::Sequential()))
{


	totalLayers = numHiddenLayers + 1;

	mlist->push_back(torch::nn::Linear(inShape, hiddenLayers[0]));

	int lastNeurons = hiddenLayers[0];
	bool lastDrop = false;

	for (int i = 0; i < numHiddenLayers - 1; i++) {

		if (hiddenLayers[i + 1] < 0) {
			float prob = ((float)hiddenLayers[i + 1] / 100) * -1;
			mlist->push_back(torch::nn::Dropout(prob));

			lastDrop = true;
		}
		else if (lastDrop) {

			torch::nn::Linear curr(lastNeurons, hiddenLayers[i + 1]);

			mlist->push_back(curr);
			lastDrop = false;
			lastNeurons = hiddenLayers[i + 1];

		}
		else {

			torch::nn::Linear curr(hiddenLayers[i], hiddenLayers[i + 1]);

			mlist->push_back(curr);
			lastNeurons = hiddenLayers[i + 1];
		}


	}

	mlist->push_back(torch::nn::Linear(hiddenLayers[numHiddenLayers - 1], outShape));



}

torch::Tensor Model::forward(torch::Tensor x) {

	torch::nn::SequentialImpl::Iterator ptr = mlist->begin();

	torch::Tensor prediction = torch::relu(ptr->forward(x));
	ptr++;

	while (ptr < mlist->end() - 1) {

		prediction = torch::relu(ptr->forward(prediction));
		ptr++;

	}

	return ptr->forward(prediction);



}

float TrainValidateModel(Model* modelPtr, BatchGenerator gen, float lr, int epochs) {
	const int ALPHA = 10;

	float minLoss = 0;

	torch::optim::AdamOptions options;
	options.lr(lr);

	torch::optim::Adam optimizer(modelPtr->parameters(), options);

	torch::nn::MSELoss lossFunc;


	int valBatches = gen.GetNumValBatches();


	for (int i  = 0; i < epochs; i++) {

		torch::Tensor features, target, valFeatures, valTarget;

		while (gen.GetTrainBatch(&features, &target)) {

			torch::Tensor prediction = modelPtr->forward(features);

			torch::Tensor loss = lossFunc->forward(prediction, target);


			optimizer.zero_grad();

			loss.backward();

			optimizer.step();

		}

		gen.resetTrain();

		// check here for early stopping
		// Early Stopping Method 
		// Generalization Loss > alpha


		float valLoss = 0;

		modelPtr->eval();

		while (gen.GetValBatch(&valFeatures, &valTarget)) {

			valLoss += Validate(modelPtr, valFeatures, valTarget);

		}

		gen.resetVal();

		valLoss /= valBatches;

		valLoss = sqrt(valLoss);


		modelPtr->train();

		if (i == 0 || valLoss < minLoss) {
			minLoss = valLoss;
		}

		float generalizedLoss = 100 * ((valLoss - minLoss) / minLoss);


		if (generalizedLoss > ALPHA) {
			break;
		}
	}

		return minLoss;

}



float Validate(Model* modelPtr, torch::Tensor features, torch::Tensor target) {

	torch::Tensor prediction = modelPtr->forward(features);

	torch::nn::MSELoss lossFunc;

	torch::Tensor loss = lossFunc(prediction, target);

	return loss.item<float>();

}
