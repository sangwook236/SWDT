#define _PUBLIC 1
#include <hcrf/hCRF.h>
#include <string>
#include <fstream>
#include <iostream>

//#define _OPENMP 1
#ifdef _OPENMP
#include <omp.h>
#endif


namespace {
namespace local {

enum HCRF_MODE { MODE_TRAIN = 1, MODE_TEST = 2, MODE_VALIDATE = 4, MODE_DEBUG = 8 };
enum HCRF_TOOLBOX { TOOLBOX_CRF = 1, TOOLBOX_HCRF = 2, TOOLBOX_LDCRF = 4, TOOLBOX_GHCRF = 8, TOOLBOX_SDCRF = 16, TOOLBOX_LVPERCEPTRON = 32 };
enum HCRF_TEST_DATASET { TEST_ON_TEST_SET = 1, TEST_ON_TRAINING_SET = 2, TEST_ON_VALIDATION_SET = 4 };

void sample()
{
	const int mode = MODE_TRAIN; // int mode = 0;
	const int toolboxType = TOOLBOX_LDCRF;
	Toolbox *toolbox = NULL;

	const int opt = OPTIMIZER_BFGS;
	const int InitMode = INIT_RANDOM;
	const double initWeightRangeMin = -1.0;
	const double initWeightRangeMax = 1.0;

	const bool doesContinueTraining = false;
	const int testDataset = TEST_ON_TRAINING_SET;

	const int maxIterationCount = 300; // int max = -1;

	const int nbHiddenStates = 3;
	const int windowSize = 0;
	const int debugLevel = 1;

	double regFactorL2 = 0.0;  // L2 regularization factor
	double regFactorL1 = 0.0;  // L1 regularization factor

#ifdef UNIX
	const std::string data_home = "./crf_data/";
#else
	const std::string data_home = ".\\crf_data\\";
#endif
	const std::string filenameDataTrain(data_home + "dataTrain.csv");
	const std::string filenameDataTrainSparse;
	const std::string filenameLabelsTrain(data_home + "labelsTrain.csv");
	const std::string filenameSeqLabelsTrain(data_home + "seqLabelsTrain.csv");

	const std::string filenameDataTest(data_home + "dataTest.csv");
	const std::string filenameDataTestSparse;
	const std::string filenameLabelsTest(data_home + "labelsTest.csv");
	const std::string filenameSeqLabelsTest(data_home + "seqLabelsTest.csv");

	const std::string filenameDataValidate(data_home + "dataValidate.csv");
	const std::string filenameDataValidateSparse;
	const std::string filenameLabelsValidate(data_home + "labelsValidate.csv");
	const std::string filenameSeqLabelsValidate(data_home + "seqLabelsValidate.csv");

	const std::string filenameModel(data_home + "model.txt");
	const std::string filenameFeatures(data_home + "features.txt");
	const std::string filenameOutput(data_home + "results.txt");
	const std::string filenameStats(data_home + "stats.txt");

/*
	// read command-line arguments 
	for (int k = 1; k < argc; ++k)
	{
		if (argv[k][0] != '-') break;
		else if (argv[k][1] == 't') 
		{
			mode |= MODE_TRAIN;
			if(argv[k][2] == 'c') 
				doesContinueTraining = true;
		}
		else if (argv[k][1] == 'T') 
		{
			mode |= MODE_TEST;
			if (argv[k][2] == 'T') 
				testDataset |= TEST_ON_TRAINING_SET;
			if (argv[k][2] == 'V') 
				testDataset |= TEST_ON_VALIDATION_SET;
		}
		else if (argv[k][1] == 'v') 
		{
			mode |= MODE_VALIDATE;
		}
		else if (argv[k][1] == 'd') 
		{
			if (argv[k][2] == 's') 
			{
				filenameDataTrainSparse = argv[++k];
				filenameDataTrain = "";
			}
			else
				filenameDataTrain = argv[++k];
		}
		else if (argv[k][1] == 'l') 
		{
			filenameLabelsTrain = argv[++k];
		}
		else if (argv[k][1] == 'D') 
		{
			if (argv[k][2] == 'S') 
			{
				filenameDataTestSparse = argv[++k];
				filenameDataTest = "";
			}
			else
				filenameDataTest = argv[++k];
		}
		else if (argv[k][1] == 'L') 
		{
			filenameLabelsTest = argv[++k];
		}
		else if (argv[k][1] == 'm') 
		{
			filenameModel = argv[++k];
		}
		else if (argv[k][1] == 'f') 
		{
			filenameFeatures = argv[++k];
		}
		else if (argv[k][1] == 'r') 
		{
			filenameOutput = argv[++k];
		}
		else if (argv[k][1] == 'c') 
		{
			filenameStats = argv[++k];
		}
		else if (argv[k][1] == 'I') 
		{
			if (!strcmp(argv[k+1], "random"))
				InitMode = INIT_RANDOM;
			else if (!strcmp(argv[k+1], "gaussian"))
				InitMode = INIT_RANDOM_GAUSSIAN;
			else if (!strcmp(argv[k+1], "zero"))
				InitMode = INIT_ZERO;
			++k;
		}
		else if (argv[k][1] == 'o') 
		{
			if (!strcmp(argv[k+1], "cg"))
				opt = OPTIMIZER_CG;
			else if (!strcmp(argv[k+1], "bfgs"))
				opt = OPTIMIZER_BFGS;
			else if (!strcmp(argv[k+1], "asa"))
				opt = OPTIMIZER_ASA;
			else if (!strcmp(argv[k+1], "owlqn"))
				opt = OPTIMIZER_OWLQN;
			else if (!strcmp(argv[k+1], "lbfgs"))
				opt = OPTIMIZER_LBFGS;
			++k;
		}
		else if (argv[k][1] == 'a') 
		{
			if (!strcmp(argv[k+1], "crf"))
				toolboxType = TOOLBOX_CRF;
			else if (!strcmp(argv[k+1], "hcrf"))
				toolboxType = TOOLBOX_HCRF;
			else if (!strcmp(argv[k+1], "ldcrf") || !strcmp(argv[k+1], "fhcrf"))
				toolboxType = TOOLBOX_LDCRF;
			else if (!strcmp(argv[k+1], "ghcrf"))
				toolboxType = TOOLBOX_GHCRF;
			else if (!strcmp(argv[k+1], "sdcrf"))
				toolboxType = TOOLBOX_SDCRF;
			k++;
		}
		else if (argv[k][1] == 'p')
		{
			debugLevel = atoi(argv[++k]);
		}
		else if (argv[k][1] == 'i')
		{
			maxIterationCount = atoi(argv[++k]);
		}
		else if (argv[k][1] == 'h')
		{
			nbHiddenStates = atoi(argv[++k]);
		}
		else if (argv[k][1] == 'w')
		{
			windowSize = atoi(argv[++k]);
		}
		else if (argv[k][1] == 's')
		{
			if (argv[k][2] == '1')
			{
				regFactorL1 = atof(argv[++k]);  // L1 regularization factor
			}
			else
			{
				regFactorL2 = atof(argv[++k]);  // L2 regularization factor
			}
		}
		else if (argv[k][1] == 'R')
		{
			initWeightRangeMin = atof(argv[++k]);
			initWeightRangeMax = atof(argv[++k]);
		}
		else if (argv[k][1] == 'P')
		{
#ifdef _OPENMP
			omp_set_num_threads(atoi(argv[++k]));
#else
			std::cerr << "No OpenMP support";
#endif
		}
		else usage(argv);
    }

	if (mode == 0)
		usage(argv);
*/

	if (mode & MODE_TRAIN || mode & MODE_TEST || mode & MODE_VALIDATE)
	{
		if (TOOLBOX_HCRF == toolboxType)
			toolbox = new ToolboxHCRF(nbHiddenStates, opt, windowSize);
		else if (TOOLBOX_LDCRF == toolboxType)
			toolbox = new ToolboxLDCRF(nbHiddenStates, opt, windowSize);
		else if (TOOLBOX_GHCRF == toolboxType)
			toolbox = new ToolboxGHCRF(nbHiddenStates, opt, windowSize);
#ifndef _PUBLIC
		else if (TOOLBOX_SDCRF == toolboxType)
			toolbox = new ToolboxSharedLDCRF(nbHiddenStates, opt, windowSize);
#endif
		else
			toolbox = new ToolboxCRF(opt, windowSize);

		toolbox->setDebugLevel(debugLevel);
	}

	if (MODE_TRAIN & mode)
	{
		std::cout << "reading training set..." << std::endl;
		const char *fileData = filenameDataTrain.empty() ? 0 : filenameDataTrain.c_str();
		const char *fileDataSparse = filenameDataTrainSparse.empty() ? 0 : filenameDataTrainSparse.c_str();

		DataSet data;
		if (toolboxType == TOOLBOX_HCRF || toolboxType == TOOLBOX_GHCRF )
			data.load(fileData, NULL, (char *)filenameSeqLabelsTrain.c_str(), NULL, NULL, fileDataSparse);
		else
			data.load(fileData, (char *)filenameLabelsTrain.c_str(), NULL, NULL, NULL, fileDataSparse);

		if (maxIterationCount >= 0)
			toolbox->setMaxNbIteration(maxIterationCount);
		if (regFactorL2 >= 0)
			toolbox->setRegularizationL2(regFactorL2);
		if (regFactorL1 >= 0)
			toolbox->setRegularizationL1(regFactorL1);
		toolbox->setRangeWeights(initWeightRangeMin, initWeightRangeMax);
		toolbox->setWeightInitType(InitMode);

		// Modified by Hugues Salamin 07-16-09.
		// To compare CRF and LDCRF with one hidden state. Looking at value of gradient and function.
		// Uncomment if you want same starting point.
		// toolbox->setWeightInitType(INIT_ZERO);

		std::cout << "starting training ..." << std::endl;
		if (doesContinueTraining)
		{
			toolbox->load((char *)filenameModel.c_str(), (char *)filenameFeatures.c_str());
			toolbox->train(data, false);
		}
		else
		{
			toolbox->train(data, true);
		}

		toolbox->save((char *)filenameModel.c_str(), (char *)filenameFeatures.c_str());
	}

	// TODO: Implement the validate function in Toolbox
	if (MODE_VALIDATE & mode)
	{
		std::cout << "reading training set..." << std::endl;
		DataSet dataTrain;
		if (toolboxType == TOOLBOX_HCRF || toolboxType == TOOLBOX_GHCRF )
			dataTrain.load((char *)filenameDataTrain.c_str(), NULL, (char *)filenameSeqLabelsTrain.c_str());
		else
			dataTrain.load((char *)filenameDataTrain.c_str(), (char *)filenameLabelsTrain.c_str());

		std::cout << "reading validation set..." << std::endl;
		DataSet dataValidate;
		if (toolboxType == TOOLBOX_HCRF || toolboxType == TOOLBOX_GHCRF )
			dataValidate.load((char *)filenameDataValidate.c_str(), NULL, (char *)filenameSeqLabelsValidate.c_str());
		else
			dataValidate.load((char *)filenameDataValidate.c_str(), (char *)filenameLabelsValidate.c_str());

		if (maxIterationCount >= 0)
			toolbox->setMaxNbIteration(maxIterationCount);

		std::cout << "starting validation ..." << std::endl;
		toolbox->validate(dataTrain, dataValidate, regFactorL2, (char *)filenameStats.c_str());
	}

	if (MODE_TEST & mode)
	{
		std::cout << "reading testing set..." << std::endl;
		DataSet data;
		if (toolboxType == TOOLBOX_HCRF || toolboxType == TOOLBOX_GHCRF)
			data.load((char *)filenameDataTest.c_str(), NULL, (char *)filenameSeqLabelsTest.c_str());
		else
			data.load((char *)filenameDataTest.c_str(), (char *)filenameLabelsTest.c_str());

		std::ofstream fileStats1 ((char *)filenameStats.c_str());
		if (fileStats1.is_open())
		{
			fileStats1 << std::endl << std::endl << "TESTING DATA SET" << std::endl << std::endl;
			fileStats1.close();
		}

		std::cout << "starting testing ..." << std::endl;
		toolbox->load((char *)filenameModel.c_str(), (char *)filenameFeatures.c_str());
		toolbox->test(data, (char *)filenameOutput.c_str(), (char *)filenameStats.c_str());

		if (testDataset & TEST_ON_TRAINING_SET)
		{
			std::ofstream fileStats((char *)filenameStats.c_str(), std::ios::out | std::ios::app);
			if (fileStats.is_open())
			{
				fileStats << std::endl << std::endl << "TRAINING DATA SET" << std::endl << std::endl;
				fileStats.close();
			}
/*
			std::ofstream fileOutput((char *)filenameOutput.c_str(), std::ios::out | std::ios::app);
			if (fileOutput.is_open())
			{
				fileOutput << std::endl << std::endl << "TRAINING DATA SET" << std::endl << std::endl;
				fileOutput.close();
			}
*/
			std::cout << "reading training set..." << std::endl;
			DataSet dataTrain((char *)filenameDataTrain.c_str(), (char *)filenameLabelsTrain.c_str(), (char *)filenameSeqLabelsTrain.c_str());

			std::cout << "starting testing ..." << std::endl;
			toolbox->test(dataTrain, NULL, (char *)filenameStats.c_str());
		}

		if (testDataset & TEST_ON_VALIDATION_SET)
		{
			std::ofstream fileStats((char *)filenameStats.c_str(), std::ios::out | std::ios::app);
			if (fileStats.is_open())
			{
				fileStats << std::endl << std::endl << "VALIDATION DATA SET" << std::endl << std::endl;
				fileStats.close();
			}
/*
			std::ofstream fileOutput((char *)filenameOutput.c_str(), std::ios::out | std::ios::app);
			if (fileOutput.is_open())
			{
				fileOutput << std::endl << std::endl << "TRAINING DATA SET" << std::endl << std::endl;
				fileOutput.close();
			}
*/
			std::cout << "reading validation set..." << std::endl;
			DataSet dataValidate((char *)filenameDataValidate.c_str(), (char *)filenameLabelsValidate.c_str(), (char *)filenameSeqLabelsValidate.c_str());

			std::cout << "starting testing ..." << std::endl;
			toolbox->test(dataValidate, NULL, (char *)filenameStats.c_str());
		}
	}

	if (toolbox)
		delete toolbox;
}

}  // namespace local
}  // unnamed namespace

void hcrf()
{
	local::sample();
}
