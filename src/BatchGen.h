#pragma once
#include <torch/torch.h>
#include <fstream>
#include <string>



class BatchGenerator {
	public:

		BatchGenerator(std::string trainFilename, std::string valFilename, int batchSize);
		BatchGenerator(BatchGenerator& other);

		bool GetTrainBatch(torch::Tensor* features, torch::Tensor* target);
		bool GetValBatch(torch::Tensor* features, torch::Tensor* target);

		void resetTrain();
		void resetVal();

		int GetNumTrainBatches();
		int GetNumValBatches();

		int GetBatchSize();

		torch::Device GetDevice();

	private: 


		int trainBatchNum, valBatchNum, batchSize, totalTrainBatches, totalValBatches;

		torch::Device device;

		torch::Tensor* trainFeatureArr, *trainTargetArr, *valFeatureArr, *valTargetArr;

		int FillBatchArr(std::string filename, int batchSize, torch::Tensor** featuresPtr, torch::Tensor** targetPtr);

		bool GetCsvSize(std::string fname, int* linesPtr, int* fieldsPtr);

		int CountDelim(std::string str, char delim);

		bool LoadCsv(std::string fname, float** arrPtr, int* rowPtr, int* columnPtr);



};
