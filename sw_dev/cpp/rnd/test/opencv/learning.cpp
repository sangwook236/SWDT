// Test.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include <opencv/ml.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <ctime>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

const size_t trainingSampleNum = 8;  // # of rows
const size_t trainingFeatureNum = 8;  // # of cols

void train_by_svm()
{
	const int whichSvmType = 0;

	CvSVMParams svmParams;
	switch (whichSvmType)
	{
	case 0:
		// C_SVC, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR
		svmParams.svm_type = CvSVM::NU_SVC;
		// LINEAR, POLY, RBF, SIGMOID
		svmParams.kernel_type = CvSVM::RBF;
		svmParams.degree = 3.0;
		svmParams.gamma = 64.0;
		svmParams.coef0 = 1.0;
		svmParams.C = 8.0;
		svmParams.nu = 0.5;
		//svmParams.p = 0.0;
		//svmParams.class_weights = 0L;
		break;
	case 1:
		// C_SVC, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR
		svmParams.svm_type = CvSVM::NU_SVR;
		// LINEAR, POLY, RBF, SIGMOID
		svmParams.kernel_type = CvSVM::RBF;
		svmParams.degree = 3.0;
		svmParams.gamma = 100.0;
		svmParams.coef0 = 1.0;
		svmParams.C = 0.01;
		svmParams.nu = 0.5;
		//svmParams.p = 0.0;
		//svmParams.class_weights = 0L;
		break;
	default:
		// do nothing
		break;
	}

	// a combination of CV_TERMCRIT_ITER and CV_TERMCRIT_EPS
	svmParams.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	svmParams.term_crit.max_iter = 500;
	svmParams.term_crit.epsilon = 0.0001;

	//
	CvMat *trainInput = cvCreateMat(trainingSampleNum, trainingFeatureNum, CV_32FC1);
	CvMat *trainOutput = cvCreateMat(trainingSampleNum, 1, CV_32FC1);
	for (int i = 0; i < trainingSampleNum; ++i)
	{
		for (int j = 0; j < trainingFeatureNum; ++j)
		{
			//cvmSet(trainInput, i, j, i == j ? 0.9 : -0.9);
			const int val = rand() % 10000;
			const int sgn = rand() % 2 ? 1 : -1;
			cvmSet(trainInput, i, j, (double)sgn * (double)val * 0.01);
		}
		cvmSet(trainOutput, i, 0, i);
	}

	std::cout << "<svm> training input:" << std::endl;
	for (int i = 0; i < trainingSampleNum; ++i)
	{
		for (int j = 0; j < trainingFeatureNum; ++j)
			std::cout << '\t' << cvmGet(trainInput, i, j);
		std::cout << std::endl;
	}
	std::cout << "<svm> training output:" << std::endl;
	for (int i = 0; i < trainingSampleNum; ++i)
		std::cout << '\t' << cvmGet(trainOutput, i, 0);
	std::cout << std::endl;

	//
	CvSVM svm;
	const bool trainRet = svm.train(trainInput, trainOutput, 0L, 0L, svmParams);
	std::cout << "<svm> training return result: " << (trainRet ? "true" : "false") << std::endl;

	//
	srand((unsigned int)time(NULL));

	CvMat *predictInput = cvCreateMat(1, trainingFeatureNum, CV_32FC1);
	std::vector<float> predictOutput(trainingSampleNum);
	std::vector<int> predictInputIdx(trainingSampleNum);
	for (int i = 0; i < trainingSampleNum; ++i)
	{
		predictInputIdx[i] = rand() % trainingSampleNum;
		cvGetRow(trainInput, predictInput, predictInputIdx[i]);
		predictOutput[i] = svm.predict(predictInput);
	}

	cvReleaseMat(&predictInput);
	cvReleaseMat(&trainInput);
	cvReleaseMat(&trainOutput);

	//
	std::cout << "<svm> prediction input index:" << std::endl;
	for (int i = 0; i < trainingSampleNum; ++i)
		std::cout << '\t' << predictInputIdx[i];
	std::cout << std::endl << "<svm> prediction output:" << std::endl;
	for (int i = 0; i < trainingSampleNum; ++i)
		std::cout << '\t' << predictOutput[i];
	switch (whichSvmType)
	{
	case 0:
		std::cout << std::endl << "<svm> prediction result:" << std::endl;
		for (int i = 0; i < trainingSampleNum; ++i)
			std::cout << '\t' << (predictOutput[i] == predictInputIdx[i] ? 'o' : 'x');
		std::cout << std::endl;
		break;
	}
}

void train_by_ann()
{
	const size_t inputLayerNeuronNum = trainingFeatureNum;
	const size_t hiddenLayerNeuronNum = 3;
	const size_t ouputLayerNeuronNum = 9;

	int layerSizesData[] = { inputLayerNeuronNum, hiddenLayerNeuronNum, ouputLayerNeuronNum };
	//3 layers, hidden layer has num_eigen neurons
	CvMat layerSizes;
	cvInitMatHeader(&layerSizes, 1, 3, CV_32SC1, layerSizesData);

	// possible activation functions: IDENTITY, SIGMOID_SYM, GAUSSIAN
	CvANN_MLP neuralNetwork(&layerSizes, CvANN_MLP::SIGMOID_SYM, 1.0, 1.0);

	// Create neuron network classification using training examples 
	CvANN_MLP_TrainParams annParams;
	annParams.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	annParams.term_crit.max_iter = 5000;
	annParams.term_crit.epsilon = 0.0001;

	// BACKPROP, RPROP
	annParams.train_method = CvANN_MLP_TrainParams::BACKPROP;
	// backpropagation parameters
	annParams.bp_dw_scale = 0.25;
	annParams.bp_moment_scale = 0.0;
	// rprop parameters
	//annParams.rp_dw0 = 0.0;
	//annParams.rp_dw_plus = 0.0;
	//annParams.rp_dw_minus = 0.0;
	//annParams.rp_dw_min = 0.0;
	//annParams.rp_dw_max = 0.0;

	//
	srand((unsigned int)time(NULL));

	CvMat *trainInput = cvCreateMat(trainingSampleNum, inputLayerNeuronNum, CV_32FC1);
	CvMat *trainOutput = cvCreateMat(trainingSampleNum, ouputLayerNeuronNum, CV_32FC1);
	for (int i = 0; i < trainingSampleNum; ++i)
	{
		for (int j = 0; j < inputLayerNeuronNum; ++j)
		{
			//cvmSet(trainInput, i, j, i == j ? 0.9 : -0.9);
			const int val = rand() % 10000;
			const int sgn = rand() % 2 ? 1 : -1;
			cvmSet(trainInput, i, j, (double)sgn * (double)val * 0.01);
		}
		for (int j = 0; j < ouputLayerNeuronNum; ++j)
			cvmSet(trainOutput, i, j, i == j ? 1.0 : -1.0);
	}

	std::cout << "<ann> training input:" << std::endl;
	for (int i = 0; i < trainingSampleNum; ++i)
	{
		for (int j = 0; j < inputLayerNeuronNum; ++j)
			std::cout << '\t' << cvmGet(trainInput, i, j);
		std::cout << std::endl;
	}
	std::cout << "<ann> training(desired) output:" << std::endl;
	for (int i = 0; i < trainingSampleNum; ++i)
	{
		for (int j = 0; j < ouputLayerNeuronNum; ++j)
			std::cout << '\t' << cvmGet(trainOutput, i, j);
		std::cout << std::endl;
	}

	// available training flags: UPDATE_WEIGHTS, NO_INPUT_SCALE, NO_OUTPUT_SCALE
	const bool trainRet = neuralNetwork.train(trainInput, trainOutput, 0L, 0L, annParams, 0);
	std::cout << "<ann> training result result: " << (trainRet ? "true" : "false") << std::endl;

	//
	CvMat *predictOutput = cvCreateMat(trainingSampleNum, ouputLayerNeuronNum, CV_32FC1);
	const float predictRet = neuralNetwork.predict(trainInput, predictOutput);
	std::cout << "<ann> prediction return result: " << (predictRet ? "true" : "false") << std::endl;

	//
	std::cout << "<ann> prediction output:" << std::endl;
	for (int i = 0; i < trainingSampleNum; ++i)
	{
		for (int j = 0; j < ouputLayerNeuronNum; ++j)
			std::cout << '\t' << std::setprecision(4) << cvmGet(predictOutput, i, j);
		std::cout << std::endl;
	}
	std::cout << "<ann> prediction result:" << std::endl;
	for (int i = 0; i < trainingSampleNum; ++i)
	{
		for (int j = 0; j < ouputLayerNeuronNum; ++j)
		{
			const char ch = fabs((cvmGet(predictOutput, i, j) >= 0 ? 1.0 : -1.0) - cvmGet(trainOutput, i, j)) < 1.0e-5 ? 'o' : 'x';
			std::cout << '\t' << std::setprecision(4) << ch;
		}
		std::cout << std::endl;
	}

	cvReleaseMat(&predictOutput);
	cvReleaseMat(&trainInput);
	cvReleaseMat(&trainOutput);
}
