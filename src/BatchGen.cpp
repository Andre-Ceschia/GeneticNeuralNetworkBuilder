#include <iostream>
#include "BatchGen.h"

BatchGenerator::BatchGenerator(std::string trainFilename, std::string valFilename, int batchSize)
	:trainBatchNum(0), valBatchNum(0), batchSize(batchSize), device(GetDevice())
{

	totalTrainBatches = FillBatchArr(trainFilename, batchSize, &trainFeatureArr, &trainTargetArr);
	totalValBatches = FillBatchArr(valFilename, batchSize, &valFeatureArr, &valTargetArr);


}

BatchGenerator::BatchGenerator(BatchGenerator& other)
	: trainBatchNum(0), valBatchNum(0), batchSize(other.batchSize), device(other.device), totalTrainBatches(other.totalTrainBatches), totalValBatches(other.totalValBatches)
{

	trainFeatureArr = new torch::Tensor[totalTrainBatches];
	trainTargetArr = new torch::Tensor[totalTrainBatches];

	valFeatureArr = new torch::Tensor[totalValBatches];
	valTargetArr = new torch::Tensor[totalValBatches];

	for (int i = 0; i < totalTrainBatches; i++) {
		trainFeatureArr[i] = other.trainFeatureArr[i].detach().clone();
		trainTargetArr[i] = other.trainTargetArr[i].detach().clone();
	}

	for (int i = 0; i < totalValBatches; i++) {
		valFeatureArr[i] = other.valFeatureArr[i].detach().clone();
		valTargetArr[i] = other.valTargetArr[i].detach().clone();
	}

}

int BatchGenerator::FillBatchArr(std::string filename, int batchSize, torch::Tensor** featuresPtr, torch::Tensor** targetPtr) {

	float* data;
	int rows, columns;

	bool ret = LoadCsv(filename, &data, &rows, &columns);

	if (!ret) {

		std::cout << "Error! Can not read " << filename << "!" << std::endl;;

		exit(1);

	}

	int totalBatches = rows / batchSize;

	if (totalBatches == 0) {
		totalBatches = 1;
		batchSize = rows;
	}

	torch::Tensor* currFeaturesArr = new torch::Tensor[totalBatches];
	torch::Tensor* currTargetArr = new torch::Tensor[totalBatches];


	for (int currBatch = 0; currBatch < totalBatches; currBatch++) {

		int adjustedBatchSize;

		if (currBatch == (totalBatches - 1)) {
			adjustedBatchSize = rows - (currBatch * batchSize);
		}
		else {
			adjustedBatchSize = batchSize;
		}


		torch::Tensor features = torch::empty({ adjustedBatchSize, (columns - 1) });
		torch::Tensor target = torch::empty({ adjustedBatchSize, 1 });


		int tensorRowInd = 0;

		for (int i = currBatch * batchSize; i < (currBatch * batchSize + adjustedBatchSize); i++) {

			for (int j = 0; j < columns - 1; j++) {

				features.index({ tensorRowInd, j }) = data[i * columns + j];
			}

			target.index({ tensorRowInd, 0 }) = data[i * columns + (columns - 1)];

			tensorRowInd++;
		}


		features = features.to(device);
		target = target.to(device);

		currFeaturesArr[currBatch] = features;
		currTargetArr[currBatch] = target;



	}

	delete[] data;

	*featuresPtr = currFeaturesArr;
	*targetPtr = currTargetArr;

	return totalBatches;


}

bool BatchGenerator::GetTrainBatch(torch::Tensor* featuresPtr, torch::Tensor* targetPtr) {

	if (trainBatchNum == totalTrainBatches) {
		return false;
	}

	*featuresPtr = trainFeatureArr[trainBatchNum];
	*targetPtr = trainTargetArr[trainBatchNum];

	trainBatchNum++;

	return true;



}

bool BatchGenerator::GetValBatch(torch::Tensor* featuresPtr, torch::Tensor* targetPtr) {

	if (valBatchNum == totalValBatches) {
		return false;
	}

	*featuresPtr = valFeatureArr[valBatchNum];
	*targetPtr = valTargetArr[valBatchNum];

	valBatchNum++;

	return true;

}


int BatchGenerator::GetBatchSize() {
	return batchSize;
}

void BatchGenerator::resetTrain() {
	trainBatchNum = 0;
}


void BatchGenerator::resetVal() {
	valBatchNum = 0;
}



int BatchGenerator::GetNumTrainBatches() {
	return totalTrainBatches;
}

int BatchGenerator::GetNumValBatches() {
	return totalValBatches;
}

torch::Device BatchGenerator::GetDevice() {
	if (torch::cuda::is_available()) {
		return torch::Device("cuda");
	}
	else {
		return torch::Device("cpu");
	}


}

int BatchGenerator::CountDelim(std::string str, char delim) {

	int count = 0;

	for (int i = 0; i < str.size(); i++) {
		if (str[i] == delim) {
			count++;
		}

	}

	return count;

}

// assuming there is a header line and it does not count it
bool BatchGenerator::GetCsvSize(std::string fname, int* linesPtr, int* fieldsPtr) {

	std::ifstream fptr(fname);

	if (!fptr) {
		return false;
	}

	std::string line;
	int lines = 0;

	if (getline(fptr, line)) {

		*fieldsPtr = CountDelim(line, ',') + 1;


	}
	else {
		*linesPtr = 0;
		*fieldsPtr = 0;

		return true;
	}

	while (getline(fptr, line)) {
		lines++;
	}

	*linesPtr = lines;

	fptr.close();

	return true;



}

bool BatchGenerator::LoadCsv(std::string fname, float** arrPtr, int* rowPtr, int* columnPtr) {

	int rows, columns;

	bool valid = GetCsvSize(fname, &rows, &columns);

	if (!valid) {
		return false;
	}

	float* arr = new float[rows * columns];

	std::ifstream fptr(fname);

	std::string line;

	// getting rid of header
	getline(fptr, line);

	for (int i = 0; i < rows; i++) {
		getline(fptr, line);

		int strIndex = 0;

		for (int j = 0; j < columns; j++) {

			std::string entry = "";

			while (line[strIndex] != ',' && line[strIndex] != '\0') {
				entry += line[strIndex];
				strIndex++;
			}
			// this puts the pointer to the start of the next string
			strIndex++;

			arr[i * columns + j] = std::stof(entry);

		}
	}

	fptr.close();

	*rowPtr = rows;
	*columnPtr = columns;
	*arrPtr = arr;


	return true;




}
