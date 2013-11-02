#define _PUBLIC 1
#include <hCRF/hCRF.h>
#include <boost/smart_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>
#include <list>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <ctime>

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
	const int mode = MODE_TRAIN | MODE_TEST; //int mode = 0;
	const int toolboxType = TOOLBOX_LDCRF;

	const int opt = OPTIMIZER_BFGS;  // OPTIMIZER_CG, OPTIMIZER_BFGS, OPTIMIZER_ASA, OPTIMIZER_OWLQN, OPTIMIZER_LBFGS.
	const int initMode = INIT_RANDOM;  // INIT_RANDOM, INIT_RANDOM_GAUSSIAN, INIT_ZERO.
	const double initWeightRangeMin = -1.0;
	const double initWeightRangeMax = 1.0;

	const bool doesContinueTraining = false;
	const int testDataset = TEST_ON_TRAINING_SET;

	const int maxIterationCount = 300; //int max = -1;

	const int nbHiddenStates = 3;
	const int windowSize = 0;
	const int debugLevel = 1;

	double regFactorL2 = 0.0;  // L2 regularization factor.
	double regFactorL1 = 0.0;  // L1 regularization factor.

	const std::string dataset_base_directory = "./data/probabilistic_graphical_model/hcrf/";

	const std::string filenameDataTrain(dataset_base_directory + "dataTrain.csv");
	const std::string filenameDataTrainSparse;
	const std::string filenameLabelsTrain(dataset_base_directory + "labelsTrain.csv");  // for CRF & LDCRF.
	const std::string filenameSeqLabelsTrain(dataset_base_directory + "seqLabelsTrain.csv");  // for HCRF & GHCRF.

	const std::string filenameDataTest(dataset_base_directory + "dataTest.csv");
	const std::string filenameDataTestSparse;
	const std::string filenameLabelsTest(dataset_base_directory + "labelsTest.csv");  // for CRF & LDCRF.
	const std::string filenameSeqLabelsTest(dataset_base_directory + "seqLabelsTest.csv");  // for HCRF & GHCRF.

	const std::string filenameDataValidate(dataset_base_directory + "dataValidate.csv");
	const std::string filenameDataValidateSparse;
	const std::string filenameLabelsValidate(dataset_base_directory + "labelsValidate.csv");  // for CRF & LDCRF.
	const std::string filenameSeqLabelsValidate(dataset_base_directory + "seqLabelsValidate.csv");  // for HCRF & GHCRF.

	const std::string filenameModel(dataset_base_directory + "model.txt");
	const std::string filenameFeatures(dataset_base_directory + "features.txt");
	const std::string filenameOutput(dataset_base_directory + "results.txt");
	const std::string filenameStats(dataset_base_directory + "stats.txt");

/*
	// read command-line arguments.
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
				initMode = INIT_RANDOM;
			else if (!strcmp(argv[k+1], "gaussian"))
				initMode = INIT_RANDOM_GAUSSIAN;
			else if (!strcmp(argv[k+1], "zero"))
				initMode = INIT_ZERO;
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
				regFactorL1 = atof(argv[++k]);  // L1 regularization factor.
			}
			else
			{
				regFactorL2 = atof(argv[++k]);  // L2 regularization factor.
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

	Toolbox *toolbox = NULL;
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
		std::cout << "reading training set (for training)..." << std::endl;
		const char *fileData = filenameDataTrain.empty() ? NULL : filenameDataTrain.c_str();
		const char *fileDataSparse = filenameDataTrainSparse.empty() ? NULL : filenameDataTrainSparse.c_str();

		DataSet data;
		if (TOOLBOX_HCRF == toolboxType || TOOLBOX_GHCRF == toolboxType)
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
		toolbox->setWeightInitType(initMode);

		// Modified by Hugues Salamin 07-16-09.
		// To compare CRF and LDCRF with one hidden state. Looking at value of gradient and function.
		// Uncomment if you want same starting point.
		//toolbox->setWeightInitType(INIT_ZERO);

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

	// TODO: Implement the validate function in Toolbox.
	if (MODE_VALIDATE & mode)
	{
		std::cout << "reading training set (for validation)..." << std::endl;
		DataSet dataTrain;
		if (TOOLBOX_HCRF == toolboxType || TOOLBOX_GHCRF == toolboxType)
			dataTrain.load((char *)filenameDataTrain.c_str(), NULL, (char *)filenameSeqLabelsTrain.c_str());
		else
			dataTrain.load((char *)filenameDataTrain.c_str(), (char *)filenameLabelsTrain.c_str());

		std::cout << "reading validation set (for validation)..." << std::endl;
		DataSet dataValidate;
		if (TOOLBOX_HCRF == toolboxType || TOOLBOX_GHCRF == toolboxType)
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
		std::cout << "reading testing set (for testing)..." << std::endl;
		DataSet data;
		if (TOOLBOX_HCRF == toolboxType || TOOLBOX_GHCRF == toolboxType)
			data.load((char *)filenameDataTest.c_str(), NULL, (char *)filenameSeqLabelsTest.c_str());
		else
			data.load((char *)filenameDataTest.c_str(), (char *)filenameLabelsTest.c_str());

		std::ofstream fileStats1((char *)filenameStats.c_str());
		if (fileStats1.is_open())
		{
			fileStats1 << std::endl << std::endl << "TESTING DATA SET" << std::endl << std::endl;
			fileStats1.close();
		}

		std::cout << "starting testing ..." << std::endl;
		toolbox->load((char *)filenameModel.c_str(), (char *)filenameFeatures.c_str());
		toolbox->test(data, (char *)filenameOutput.c_str(), (char *)filenameStats.c_str());

		if (TEST_ON_TRAINING_SET & testDataset)
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
			std::cout << "reading training set (for testing)..." << std::endl;
			DataSet dataTrain((char *)filenameDataTrain.c_str(), (char *)filenameLabelsTrain.c_str(), (char *)filenameSeqLabelsTrain.c_str());

			std::cout << "starting testing ..." << std::endl;
			toolbox->test(dataTrain, NULL, (char *)filenameStats.c_str());
			//toolbox->test(dataTrain, (char *)filenameOutput.c_str(), (char *)filenameStats.c_str());
		}

		if (TEST_ON_VALIDATION_SET & testDataset)
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
			std::cout << "reading validation set (for testing)..." << std::endl;
			DataSet dataValidate((char *)filenameDataValidate.c_str(), (char *)filenameLabelsValidate.c_str(), (char *)filenameSeqLabelsValidate.c_str());

			std::cout << "starting testing ..." << std::endl;
			toolbox->test(dataValidate, NULL, (char *)filenameStats.c_str());
			//toolbox->test(dataValidate, (char *)filenameDataTrain.c_str(), (char *)filenameStats.c_str());
		}
	}

	if (toolbox)
		delete toolbox;
}

void gesture_recognition_using_THoG(const std::string &datasetDirectory, const std::size_t numTrainDataFiles, const bool isSegmenting, const int nbHiddenStates, const int windowSize)
{
	const int mode = MODE_TRAIN | MODE_TEST; //int mode = 0;
	const int toolboxType = TOOLBOX_LDCRF;

	const int opt = OPTIMIZER_LBFGS;  // OPTIMIZER_CG, OPTIMIZER_BFGS, OPTIMIZER_ASA, OPTIMIZER_OWLQN, OPTIMIZER_LBFGS.
	const int initMode = INIT_RANDOM;  // INIT_RANDOM, INIT_RANDOM_GAUSSIAN, INIT_ZERO.
	const double initWeightRangeMin = -1.0;
	const double initWeightRangeMax = 1.0;

	const bool doesContinueTraining = false;
	const int testDataset = TEST_ON_TEST_SET | TEST_ON_TRAINING_SET;

	const int maxIterationCount = 300; //int max = -1;

	//const int nbHiddenStates = 3;
	//const int windowSize = 10;
	const int debugLevel = 1;

	double regFactorL2 = 0.0;  // L2 regularization factor.
	double regFactorL1 = 0.0;  // L1 regularization factor.

	const std::string featureFilePrefix("M_");
	//const std::string featureFilePrefix("K_");

	const std::string featureDataExtName(".HoG");
	//const std::string featureDataExtName(".THoG");

	// for sequence labeling or for segmenting.
	const std::string labelExtName(isSegmenting ? ".seg_lbl" : ".lbl");  // for CRF & LDCRF.
	const std::string sequenceLabelExtName(isSegmenting ? ".seg_seqlbl" : ".seqlbl");  // for HCRF & GHCRF.

	std::list<std::pair<const std::string, const std::string> > trainDataFilePairList;  // pairs of data file & sparse data file.
	std::list<std::pair<const std::string, const std::string> > testDataFilePairList;  // pairs of data file & sparse data file.
	std::list<std::pair<const std::string, const std::string> > validateDataFilePairList;  // pairs of data file & sparse data file.
	const std::size_t numTotalDataFiles = 47;
	{
		for (std::size_t i = 1; i <= numTrainDataFiles; ++i)
		{
			std::ostringstream stream;
			stream << featureFilePrefix << i;
			trainDataFilePairList.push_back(std::make_pair(stream.str(), std::string()));
		}
		for (std::size_t i = numTrainDataFiles + 1; i <= numTotalDataFiles; ++i)
		{
			std::ostringstream stream;
			stream << featureFilePrefix << i;
			testDataFilePairList.push_back(std::make_pair(stream.str(), std::string()));
		}
	}

	std::string resultantDirectoryName;
	{
		time_t rawtime;
		std::time(&rawtime);
		struct tm *timeinfo = std::localtime(&rawtime);

		char buffer[80];
		strftime(buffer, 80, "%Y%m%dT%H%M%S", timeinfo);

		resultantDirectoryName = datasetDirectory + std::string("HCRF_result") + std::string(isSegmenting ? "_segmentation_" : "_labeling_") + std::string(buffer) + std::string("/");

		boost::filesystem::path dir(resultantDirectoryName);
		if (!boost::filesystem::create_directory(dir))
		{
			std::cerr << "cannot create a HCRF resultant directory: " << resultantDirectoryName << std::endl;
			return;
		}
	}

	const std::string filenameModel(resultantDirectoryName + "model.txt");
	const std::string filenameFeatures(resultantDirectoryName + "features.txt");
	const std::string filenameOutput(resultantDirectoryName + "results.txt");
	const std::string filenameStats(resultantDirectoryName + "stats.txt");

	const std::string filenameErrorReport(resultantDirectoryName + "error_report.txt");
	std::ofstream ofsErrorReport(filenameErrorReport, std::ios::trunc | std::ios::out);

	//
	boost::scoped_ptr<Toolbox> toolbox;
	if (mode & MODE_TRAIN || mode & MODE_TEST || mode & MODE_VALIDATE)
	{
		if (TOOLBOX_HCRF == toolboxType)
			toolbox.reset(new ToolboxHCRF(nbHiddenStates, opt, windowSize));
		else if (TOOLBOX_LDCRF == toolboxType)
			toolbox.reset(new ToolboxLDCRF(nbHiddenStates, opt, windowSize));
		else if (TOOLBOX_GHCRF == toolboxType)
			toolbox.reset(new ToolboxGHCRF(nbHiddenStates, opt, windowSize));
#ifndef _PUBLIC
		else if (TOOLBOX_SDCRF == toolboxType)
			toolbox.reset(new ToolboxSharedLDCRF(nbHiddenStates, opt, windowSize));
#endif
		else
			toolbox.reset(new ToolboxCRF(opt, windowSize));

		toolbox->setDebugLevel(debugLevel);
	}

	if (MODE_TRAIN & mode)
	{
		boost::timer::auto_cpu_timer timer;

		std::size_t feature_file_id = 1;
		for (std::list<std::pair<const std::string, const std::string> >::const_iterator cit = trainDataFilePairList.begin(); cit != trainDataFilePairList.end(); ++cit, ++feature_file_id)
		{
			const std::string filenameDataTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + featureDataExtName));
			const std::string filenameDataTrainSparse(cit->second.empty() ? std::string() : (datasetDirectory + cit->second + featureDataExtName));
			const std::string filenameLabelsTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + labelExtName));
			const std::string filenameSeqLabelsTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + sequenceLabelExtName));

			const char *fileData = filenameDataTrain.empty() ? NULL : filenameDataTrain.c_str();
			const char *fileDataSparse = filenameDataTrainSparse.empty() ? NULL : filenameDataTrainSparse.c_str();

			std::cout << "reading training set (for training)...: " << filenameDataTrain << std::endl;
			DataSet data;
			if (TOOLBOX_HCRF == toolboxType || TOOLBOX_GHCRF == toolboxType)
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
#if 1
			toolbox->setWeightInitType(initMode);
#else
			// Modified by Hugues Salamin 07-16-09.
			// To compare CRF and LDCRF with one hidden state. Looking at value of gradient and function.
			toolbox->setWeightInitType(INIT_ZERO);
#endif

			std::cout << "starting training (training feature file ID: " << feature_file_id << ") ..." << std::endl;
			try
			{
				if (doesContinueTraining || trainDataFilePairList.begin() != cit)
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
			catch (const std::exception &e)
			{
				if (ofsErrorReport.is_open())
				{
					ofsErrorReport << "error: " << e.what() << ", dataset dir: " << datasetDirectory << ", video file id: " << cit->first << std::endl;
				}
			}
		}
	}

	// TODO: Implement the validate function in Toolbox.
	if (MODE_VALIDATE & mode)
	{
		boost::timer::auto_cpu_timer timer;

		DataSet dataTrain;

		for (std::list<std::pair<const std::string, const std::string> >::const_iterator cit = trainDataFilePairList.begin(); cit != trainDataFilePairList.end(); ++cit)
		{
			const std::string filenameDataTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + featureDataExtName));
			const std::string filenameDataTrainSparse(cit->second.empty() ? std::string() : (datasetDirectory + cit->second + featureDataExtName));
			const std::string filenameLabelsTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + labelExtName));
			const std::string filenameSeqLabelsTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + sequenceLabelExtName));

			const char *fileData = filenameDataTrain.empty() ? NULL : filenameDataTrain.c_str();
			const char *fileDataSparse = filenameDataTrainSparse.empty() ? NULL : filenameDataTrainSparse.c_str();

			std::cout << "reading training set (for validation)...: " << filenameDataTrain << std::endl;
			if (TOOLBOX_HCRF == toolboxType || TOOLBOX_GHCRF == toolboxType)
				dataTrain.load((char *)filenameDataTrain.c_str(), NULL, (char *)filenameSeqLabelsTrain.c_str());
			else
				dataTrain.load((char *)filenameDataTrain.c_str(), (char *)filenameLabelsTrain.c_str());
		}

		std::size_t feature_file_id = 1;
		for (std::list<std::pair<const std::string, const std::string> >::const_iterator cit = validateDataFilePairList.begin(); cit != validateDataFilePairList.end(); ++cit, ++feature_file_id)
		{
			const std::string filenameDataValidate(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + featureDataExtName));
			const std::string filenameLabelsValidate(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + labelExtName));
			const std::string filenameSeqLabelsValidate(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + sequenceLabelExtName));

			std::cout << "reading validation set (for validation)...: " << filenameDataValidate << std::endl;
			DataSet dataValidate;
			if (TOOLBOX_HCRF == toolboxType || TOOLBOX_GHCRF == toolboxType)
				dataValidate.load((char *)filenameDataValidate.c_str(), NULL, (char *)filenameSeqLabelsValidate.c_str());
			else
				dataValidate.load((char *)filenameDataValidate.c_str(), (char *)filenameLabelsValidate.c_str());

			if (maxIterationCount >= 0)
				toolbox->setMaxNbIteration(maxIterationCount);

			std::cout << "starting validation (validation feature file ID: " << feature_file_id << ") ..." << std::endl;
			try
			{
				toolbox->validate(dataTrain, dataValidate, regFactorL2, (char *)filenameStats.c_str());
			}
			catch (const std::exception &e)
			{
				if (ofsErrorReport.is_open())
				{
					ofsErrorReport << "error: " << e.what() << ", dataset dir: " << datasetDirectory << ", video file id: " << cit->first << std::endl;
				}
			}
		}
	}

	if (MODE_TEST & mode)
	{
		boost::timer::auto_cpu_timer timer;

		std::cout << "loading model & feature files for testing..." << std::endl;
		toolbox->load((char *)filenameModel.c_str(), (char *)filenameFeatures.c_str());

		//if (TEST_ON_TEST_SET & testDataset)
		{
			std::ofstream fileStats1((char *)filenameStats.c_str());
			if (fileStats1.is_open())
			{
				fileStats1 << std::endl << std::endl << "TESTING DATA SET" << std::endl << std::endl;
				fileStats1.close();
			}
		}

		if (TEST_ON_TEST_SET & testDataset)
		{
			std::size_t feature_file_id = 1;
			for (std::list<std::pair<const std::string, const std::string> >::const_iterator cit = testDataFilePairList.begin(); cit != testDataFilePairList.end(); ++cit, ++feature_file_id)
			{
				const std::string filenameDataTest(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + featureDataExtName));
				const std::string filenameLabelsTest(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + labelExtName));
				const std::string filenameSeqLabelsTest(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + sequenceLabelExtName));

				std::cout << "reading testing set (for testing)...: " << filenameDataTest << std::endl;
				DataSet data;
				if (TOOLBOX_HCRF == toolboxType || TOOLBOX_GHCRF == toolboxType)
					data.load((char *)filenameDataTest.c_str(), NULL, (char *)filenameSeqLabelsTest.c_str());
				else
					data.load((char *)filenameDataTest.c_str(), (char *)filenameLabelsTest.c_str());

				std::cout << "starting testing (testing feature file ID: " << feature_file_id << ") ..." << std::endl;
				try
				{
					toolbox->test(data, (char *)filenameOutput.c_str(), (char *)filenameStats.c_str());
				}
				catch (const std::exception &e)
				{
					if (ofsErrorReport.is_open())
					{
						ofsErrorReport << "error: " << e.what() << ", dataset dir: " << datasetDirectory << ", video file id: " << cit->first << std::endl;
					}
				}
			}
		}

		if (TEST_ON_TRAINING_SET & testDataset)
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

			std::size_t feature_file_id = 1;
			for (std::list<std::pair<const std::string, const std::string> >::const_iterator cit = trainDataFilePairList.begin(); cit != trainDataFilePairList.end(); ++cit, ++feature_file_id)
			{
				const std::string filenameDataTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + featureDataExtName));
				const std::string filenameDataTrainSparse(cit->second.empty() ? std::string() : (datasetDirectory + cit->second + featureDataExtName));
				const std::string filenameLabelsTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + labelExtName));
				const std::string filenameSeqLabelsTrain(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + sequenceLabelExtName));

				std::cout << "reading training set (for testing)...: " << filenameDataTrain << std::endl;
				DataSet dataTrain((char *)filenameDataTrain.c_str(), (char *)filenameLabelsTrain.c_str(), (char *)filenameSeqLabelsTrain.c_str());

				std::cout << "starting testing (training feature file ID: " << feature_file_id << ") ..." << std::endl;
				try
				{
					toolbox->test(dataTrain, NULL, (char *)filenameStats.c_str());
					//toolbox->test(dataTrain, (char *)filenameOutput.c_str(), (char *)filenameStats.c_str());
				}
				catch (const std::exception &e)
				{
					if (ofsErrorReport.is_open())
					{
						ofsErrorReport << "error: " << e.what() << ", dataset dir: " << datasetDirectory << ", video file id: " << cit->first << std::endl;
					}
				}
			}
		}

		if (TEST_ON_VALIDATION_SET & testDataset)
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
				fileOutput << std::endl << std::endl << "VALIDATION DATA SET" << std::endl << std::endl;
				fileOutput.close();
			}
*/

			std::size_t feature_file_id = 1;
			for (std::list<std::pair<const std::string, const std::string> >::const_iterator cit = validateDataFilePairList.begin(); cit != validateDataFilePairList.end(); ++cit, ++feature_file_id)
			{
				const std::string filenameDataValidate(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + featureDataExtName));
				const std::string filenameLabelsValidate(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + labelExtName));
				const std::string filenameSeqLabelsValidate(cit->first.empty() ? std::string() : (datasetDirectory + cit->first + sequenceLabelExtName));

				std::cout << "reading validation set (for testing)..." << std::endl;
				DataSet dataValidate((char *)filenameDataValidate.c_str(), (char *)filenameLabelsValidate.c_str(), (char *)filenameSeqLabelsValidate.c_str());

				std::cout << "starting testing (validation feature file ID: " << feature_file_id << ") ..." << std::endl;
				try
				{
					toolbox->test(dataValidate, NULL, (char *)filenameStats.c_str());
					//toolbox->test(dataValidate, (char *)filenameDataTrain.c_str(), (char *)filenameStats.c_str());
				}
				catch (const std::exception &e)
				{
					if (ofsErrorReport.is_open())
					{
						ofsErrorReport << "error: " << e.what() << ", dataset dir: " << datasetDirectory << ", video file id: " << cit->first << std::endl;
					}
				}
			}
		}
	}
}

void gesture_recognition_using_THoG()
{
#if 1
	// at desire.kaist.ac.kr.

	// using ChaLearn Gesture Challenge dataset.
	const std::string dataset_base_directory("E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/");
	// using AIM gesture dataset.
	//const std::string dataset_base_directory("F:/AIM_gesture_dataset/");
#elif 0
	// at eden.kaist.ac.kr.

	// using ChaLearn Gesture Challenge dataset.
	const std::string dataset_base_directory("E:/sangwook/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/");
	// using AIM gesture dataset.
	const std::string dataset_base_directory("E:/sangwook/AIM_gesture_dataset/");
#endif

	// the number of training dataset (devel01 ~ devel20).
	//	10, 10, 8, 10, 8, 10, 9, 11, 9, 9, 8, 11, 12, 8, 8, 13, 8, 10, 9, 9

	std::list<std::pair<std::string, std::size_t> > dataset_directory_list;
#if 1
	{
		// for ChaLearn Gesture Challenge dataset.

		//dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel01_thog2_10deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel02_thog2_10deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel03_thog2_10deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel04_thog2_10deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel05_thog2_10deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel06_thog2_10deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel07_thog2_10deg_hcrf/"), 9));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel08_thog2_10deg_hcrf/"), 11));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel09_thog2_10deg_hcrf/"), 9));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel10_thog2_10deg_hcrf/"), 9));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel11_thog2_10deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel12_thog2_10deg_hcrf/"), 11));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel13_thog2_10deg_hcrf/"), 12));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel14_thog2_10deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel15_thog2_10deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel16_thog2_10deg_hcrf/"), 13));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel17_thog2_10deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel18_thog2_10deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel19_thog2_10deg_hcrf/"), 9));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel20_thog2_10deg_hcrf/"), 9));

		//dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel01_thog2_1deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel02_thog2_1deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel03_thog2_1deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel04_thog2_1deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel05_thog2_1deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel06_thog2_1deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel07_thog2_1deg_hcrf/"), 9));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel08_thog2_1deg_hcrf/"), 11));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel09_thog2_1deg_hcrf/"), 9));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel10_thog2_1deg_hcrf/"), 9));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel11_thog2_1deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel12_thog2_1deg_hcrf/"), 11));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel13_thog2_1deg_hcrf/"), 12));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel14_thog2_1deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel15_thog2_1deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel16_thog2_1deg_hcrf/"), 13));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel17_thog2_1deg_hcrf/"), 8));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel18_thog2_1deg_hcrf/"), 10));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel19_thog2_1deg_hcrf/"), 9));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("devel20_thog2_1deg_hcrf/"), 9));
	}
#else
	{
		// for AIM gesture dataset.

		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("s01_sangwook.lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_thog2_1deg_hcrf/"), ?));
		dataset_directory_list.push_back(std::make_pair(dataset_base_directory + std::string("s01_sangwook.lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_thog2_10deg_hcrf/"), ?));
	}
#endif

	std::list<int> numHiddenStatesList;
	std::list<int> windowSizeList;
	{
		for (int i = 3; i <= 5; ++i)
			numHiddenStatesList.push_back(i);
		for (int i = 0; i <= 20; ++i)
			windowSizeList.push_back(i);
	}

	for (std::list<std::pair<std::string, std::size_t> >::const_iterator cit = dataset_directory_list.begin(); cit != dataset_directory_list.end(); ++cit)
	{
		{
			const bool isSegmenting = false;

			for (std::list<int>::const_iterator citNumHiddenStates = numHiddenStatesList.begin(); citNumHiddenStates != numHiddenStatesList.end(); ++citNumHiddenStates)
				for (std::list<int>::const_iterator citWindowSize = windowSizeList.begin(); citWindowSize != windowSizeList.end(); ++citWindowSize)
					gesture_recognition_using_THoG(cit->first, cit->second, isSegmenting, *citNumHiddenStates, *citWindowSize);
		}

		{
			const bool isSegmenting = true;

			for (std::list<int>::const_iterator citNumHiddenStates = numHiddenStatesList.begin(); citNumHiddenStates != numHiddenStatesList.end(); ++citNumHiddenStates)
				for (std::list<int>::const_iterator citWindowSize = windowSizeList.begin(); citWindowSize != windowSizeList.end(); ++citWindowSize)
					gesture_recognition_using_THoG(cit->first, cit->second, isSegmenting, *citNumHiddenStates, *citWindowSize);
		}
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_hcrf {

}  // namespace my_hcrf

// C++ & Matlab.
int hcrf_main(int argc, char *argv[])
{
	//local::sample();

	// gesture recognition using THoG.
	local::gesture_recognition_using_THoG();

	return 0;
}
