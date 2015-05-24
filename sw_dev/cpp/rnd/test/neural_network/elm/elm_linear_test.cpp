//#include "stdafx.h"
#include "../elm_linear_lib/elml.hpp"
#include <Eigen/Core>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <list>
#include <vector>
#include <cmath>
#include <cassert>


namespace {
namespace local {

// [ref] DataNormalization::normalizeDataByRange() in ${SWL_CPP_HOME}/src/math/DataNormalization.cpp.
bool normalizeDataByRange(Eigen::MatrixXd &D, const double minBound, const double maxBound, const double tol = 1.0e-15)
// goal : min & max of each row = [minBound, maxBound].
// row : the dimension of data.
// col : the number of data.
{
	const std::size_t rows = D.rows();
	const std::size_t cols = D.cols();

	const Eigen::VectorXd minVec(D.rowwise().minCoeff());
	const Eigen::VectorXd maxVec(D.rowwise().maxCoeff());

#if 0
	// FIXME [modify] >> have to consider the case that the max. and min. values are equal.
	const Eigen::VectorXd factor((maxVec - minVec) / (maxBound - minBound));
	D = ((D.colwise() - minVec).colwise() / factor).array() + minBound;  // error : THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES.
#elif 0
	// FIXME [modify] >> have to consider the case that the max. and min. values are equal.
	const Eigen::VectorXd factor((maxVec - minVec) / (maxBound - minBound));
	D.colwise() -= minVec;
	D.colwise() /= factor;  // error : THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES.
	D.array() += minBound;
#elif 1
	for (std::size_t i = 0; i < rows; ++i)
	{
		if (std::abs(maxVec(i) - minVec(i)) < tol)
		{
			// TODO [check] >>
			//D.row(i).array() = (maxBound - minBound) * 0.5;
			D.row(i).setConstant((maxBound - minBound) * 0.5);
		}
		else
		{
			const double factor = (maxBound - minBound) / (maxVec(i) - minVec(i));
			D.row(i) = ((D.row(i).array() - minVec(i)) * factor).array() + minBound;
		}
	}
#else
	for (std::size_t i = 0; i < rows; ++i)
	{
		if (std::abs(maxVec(i) - minVec(i)) < tol)
		{
			// TODO [check] >>
			//D.row(i).array() = (maxBound - minBound) * 0.5;
			D.row(i).setConstant((maxBound - minBound) * 0.5);
		}
		else
		{
			const double factor = (maxBound - minBound) / (maxVec(i) - minVec(i));
			for (std::size_t j = 0; j < cols; ++j)
				D(i, j) = (D(i, j) - minVec(i)) * factor + minBound;
		}
	}
#endif

	return true;
}

bool load_classification_data(const std::string &filename, const std::size_t indexOffset, double *&attributes, double *&labels, std::size_t &numAttributes, std::size_t &numSample)
{
	std::ifstream stream(filename);
	if (!stream.is_open())
	{
		std::cerr << "file not found : " << filename << std::endl;
		return false;
	}

	std::string line;
	std::list<double> lbls, attrs;
	bool first = true;
	while (stream)
	{
		if (!std::getline(stream, line)) continue;

		std::istringstream sstream(line);
#if 1
		std::list<std::string> tokens;
		std::copy(std::istream_iterator<std::string>(sstream), std::istream_iterator<std::string>(), std::back_inserter(tokens));
#else
		const std::list<std::string> tokens(std::istream_iterator<std::string>(sstream), std::istream_iterator<std::string>());
#endif

		for (std::list<std::string>::const_iterator cit = tokens.begin(); cit != tokens.end(); ++cit)
		{
			if (tokens.begin() == cit)
				lbls.push_back(std::stof(*cit) + indexOffset);
			else
				attrs.push_back(std::stof(*cit));
		}

		if (first)
		{
			numAttributes = attrs.size();
			first = false;
		}
	}

	numSample = attrs.size() / numAttributes;
	if (attrs.size() != numSample * numAttributes)
	{
		std::cerr << "the size of sample is inconsistent" << std::endl;
		return false;
	}
	if (lbls.size() != numSample)
	{
		std::cerr << "the sizes of attributes and labels are not equal" << std::endl;
		return false;
	}

	attributes = new double [numSample * numAttributes];
	std::copy(attrs.begin(), attrs.end(), attributes);
	labels = new double [numSample];
	std::copy(lbls.begin(), lbls.end(), labels);

	return true;
}

bool load_regression_data(const std::string &filename, double *&attributes, double *&ys, std::size_t &numAttributes, std::size_t &numSample)
{
	std::ifstream stream(filename);
	if (!stream.is_open())
	{
		std::cerr << "file not found : " << filename << std::endl;
		return false;
	}

	std::string line;
	std::list<double> funcVals, attrs;
	bool first = true;
	while (stream)
	{
		if (!std::getline(stream, line)) continue;

		std::istringstream sstream(line);
#if 1
		std::list<std::string> tokens;
		std::copy(std::istream_iterator<std::string>(sstream), std::istream_iterator<std::string>(), std::back_inserter(tokens));
#else
		const std::list<std::string> tokens(std::istream_iterator<std::string>(sstream), std::istream_iterator<std::string>());
#endif
		for (std::list<std::string>::const_iterator cit = tokens.begin(); cit != tokens.end(); ++cit)
		{
			if (tokens.begin() == cit)
				funcVals.push_back(std::stof(*cit));
			else
				attrs.push_back(std::stof(*cit));
		}

		if (first)
		{
			numAttributes = attrs.size();
			first = false;
		}
	}

	numSample = attrs.size() / numAttributes;
	if (attrs.size() != numSample * numAttributes)
	{
		std::cerr << "the size of sample is inconsistent" << std::endl;
		return false;
	}
	if (funcVals.size() != numSample)
	{
		std::cerr << "the sizes of attributes and labels are not equal" << std::endl;
		return false;
	}

	attributes = new double [numSample * numAttributes];
	std::copy(attrs.begin(), attrs.end(), attributes);
	ys = new double [numSample];
	std::copy(funcVals.begin(), funcVals.end(), ys);

	return true;
}

void elm_linear_classification_test()
{
	// classification.
	const std::string train_filename("./data/neural_network/elm/diabetes_train.dat");
	const std::string test_filename("./data/neural_network/elm/diabetes_test.dat");

	const std::size_t indexOffset = 1;  // one-based index.

	//
	std::size_t dimTrain = 0;  // dimension.
	std::size_t numTrainSample = 0;  // sample size.
	std::size_t dimTest = 0;  // dimension.
	std::size_t numTestSample = 0;  // sample size.

	double *trainX = NULL;
	double *trainY = NULL;
	double *testX = NULL;
	double *testY = NULL;

	// load data.
	// labels use one-based index.
	if (!local::load_classification_data(train_filename, indexOffset, trainX, trainY, dimTrain, numTrainSample) ||
		!local::load_classification_data(test_filename, indexOffset, testX, testY, dimTest, numTestSample))
	{
		std::cerr << "classification data loading error" << std::endl;
		return;
	}

	assert(dimTrain == dimTest);

	Eigen::MatrixXd mTrainX(Eigen::Map<Eigen::MatrixXd>(trainX, (int)dimTrain, (int)numTrainSample));  // column-major matrix, dimTrain * numTrainSample.
	Eigen::VectorXd vTrainY(Eigen::Map<Eigen::VectorXd>(trainY, (int)numTrainSample));  // column-major vector, numTrainSample * 1.
	Eigen::MatrixXd mTestX(Eigen::Map<Eigen::MatrixXd>(testX, (int)dimTest, (int)numTestSample));  // column-major matrix, dimTest * numTestSample.
	Eigen::VectorXd vTestY(Eigen::Map<Eigen::VectorXd>(testY, (int)numTestSample));  // column-major vector, numTestSample * 1.

	// data normalization.
	{
#if 0
		// TODO [check] >> what range has to be used?
		normalizeDataByRange(mTrainX, 0.0, 1.0);
		// TODO [check] >> what range has to be used?
		normalizeDataByRange(mTestX, 0.0, 1.0);
#else
		Eigen::MatrixXd mX(mTrainX.rows(), mTrainX.cols() + mTestX.cols());
		mX << mTrainX, mTestX;
		// TODO [check] >> what range has to be used?
		//normalizeDataByRange(mX, 0.0, 1.0);
		normalizeDataByRange(mX, -1.0, 1.0);

		mTrainX = mX.topLeftCorner(mTrainX.rows(), mTrainX.cols());
		mTestX = mX.topRightCorner(mTestX.rows(), mTestX.cols());
#endif
	}

	// clean-up.
	delete [] trainX; trainX = NULL;
	delete [] trainY; trainY = NULL;
	delete [] testX; testX = NULL;
	delete [] testY; testY = NULL;

#if 0
	{
		std::cout << mTrainX.size() << " : " << mTrainX.rows() << ", " << mTrainX.cols() << std::endl;
		std::cout << vTrainY.size() << " : " << vTrainY.rows() << ", " << vTrainY.cols() << std::endl;
		std::cout << mTestX.size() << " : " << mTestX.rows() << ", " << mTestX.cols() << std::endl;
		std::cout << vTestY.size() << " : " << vTestY.rows() << ", " << vTestY.cols() << std::endl;
	}
#endif

	// train.
	Eigen::MatrixXd inW;  // weights in the input nodes.
	Eigen::MatrixXd bias;  // bias in the output nodes.
    Eigen::MatrixXd outW;  // weights in the output nodes.

	const std::size_t numHiddens = 20;  // number of hidden neurons.
	const double C = 1.00;  // a tradeoff between the distance of the separating margin and the training error.
    const int resultTrain = elmTrain(
        mTrainX.data(), (int)dimTrain, (int)numTrainSample, vTrainY.data(), (int)numHiddens, C,  // inputs.
        inW, bias, outW,  // outputs.
		//	FIXME [check] >> this need to be checked.
		true  // is_classification.
    );

	// predict.
	Eigen::MatrixXd mScores;  // output scores.
	const int resultPredict = elmPredict(
		mTestX.data(), (int)dimTest, (int)numTestSample,  // inputs.
		mScores,  // outputs.
		inW, bias, outW  // inputs.
	);

	assert(mScores.cols() == numTestSample);

	// performance evaluation.
	// [ref] https://github.com/usptact/ELM-C--/blob/master/mat/breast_test.m.
	double recognitionRate = 0.0;
	//std::cout << mScores.rows() << ", " << mScores.cols() << std::endl;
	std::vector<Eigen::MatrixXf::Index> testYHat(numTestSample, -1);
	{
		Eigen::VectorXd maxVal(numTestSample);
		for (std::size_t i = 0; i < numTestSample; ++i)
		//for (std::size_t i = 0; i < 10; ++i)
		{
			maxVal(i) = mScores.col(i).maxCoeff(&testYHat[i]);
			testYHat[i] += indexOffset;
			if (vTestY(i) == testYHat[i]) ++recognitionRate;
		}
		recognitionRate /= (double)numTestSample;
	}

	// display.
	std::cout << "recognition rate = " << recognitionRate << std::endl;

	// FIXME [check] >> this need to be checked.
	//std::cout << mScores.rows() << ", " << mScores.cols() << std::endl;
	//for (std::size_t i = 0; i < mScores.cols(); ++i)
	for (std::size_t i = 0; i < 10; ++i)
		std::cout << "predicted = " << testYHat[i] << ", true = " << vTestY(i) << std::endl;
}

void elm_linear_regression_test()
{
	// regression.
	const std::string train_filename("./data/neural_network/elm/sinc_train.dat");
	const std::string test_filename("./data/neural_network/elm/sinc_test.dat");

	//
	std::size_t dimTrain = 0;  // dimension.
	std::size_t numTrainSample = 0;  // sample size.
	std::size_t dimTest = 0;  // dimension.
	std::size_t numTestSample = 0;  // sample size.

	double *trainX = NULL;
	double *trainY = NULL;
	double *testX = NULL;
	double *testY = NULL;

	// load data.
	if (!local::load_regression_data(train_filename, trainX, trainY, dimTrain, numTrainSample) ||
		!local::load_regression_data(test_filename, testX, testY, dimTest, numTestSample))
	{
		std::cerr << "regression data loading error" << std::endl;
		return;
	}

	assert(dimTrain == dimTest);

	Eigen::MatrixXd mTrainX(Eigen::Map<Eigen::MatrixXd>(trainX, (int)dimTrain, (int)numTrainSample));  // column-major matrix, dimTrain * numTrainSample.
	Eigen::VectorXd vTrainY(Eigen::Map<Eigen::VectorXd>(trainY, (int)numTrainSample));  // column-major vector, numTrainSample * 1.
	Eigen::MatrixXd mTestX(Eigen::Map<Eigen::MatrixXd>(testX, (int)dimTest, (int)numTestSample));  // column-major matrix, dimTest * numTestSample.
	Eigen::VectorXd vTestY(Eigen::Map<Eigen::VectorXd>(testY, (int)numTestSample));  // column-major vector, numTestSample * 1.

	// data normalization.
	{
#if 0
		// TODO [check] >> what range has to be used?
		normalizeDataByRange(mTrainX, 0.0, 1.0);
		// TODO [check] >> what range has to be used?
		normalizeDataByRange(mTestX, 0.0, 1.0);
#else
		Eigen::MatrixXd mX(mTrainX.rows(), mTrainX.cols() + mTestX.cols());
		mX << mTrainX, mTestX;
		// TODO [check] >> what range has to be used?
		//normalizeDataByRange(mX, 0.0, 1.0);
		normalizeDataByRange(mX, -1.0, 1.0);

		mTrainX = mX.topLeftCorner(mTrainX.rows(), mTrainX.cols());
		mTestX = mX.topRightCorner(mTestX.rows(), mTestX.cols());
#endif
	}

	// clean-up.
	delete [] trainX; trainX = NULL;
	delete [] trainY; trainY = NULL;
	delete [] testX; testX = NULL;
	delete [] testY; testY = NULL;

#if 0
	{
		std::cout << mTrainX.size() << " : " << mTrainX.rows() << ", " << mTrainX.cols() << std::endl;
		std::cout << vTrainY.size() << " : " << vTrainY.rows() << ", " << vTrainY.cols() << std::endl;
		std::cout << mTestX.size() << " : " << mTestX.rows() << ", " << mTestX.cols() << std::endl;
		std::cout << vTestY.size() << " : " << vTestY.rows() << ", " << vTestY.cols() << std::endl;
	}
#endif

	// train.
	Eigen::MatrixXd inW;  // weights in the input nodes.
	Eigen::MatrixXd bias;  // bias in the output nodes.
    Eigen::MatrixXd outW;  // weights in the output nodes.

	const std::size_t numHiddens = 20;  // number of hidden neurons.
	const double C = 1.00;  // a tradeoff between the distance of the separating margin and the training error.
    const int resultTrain = elmTrain(
        mTrainX.data(), (int)dimTrain, (int)numTrainSample, vTrainY.data(), (int)numHiddens, C,  // inputs.
        inW, bias, outW,  // outputs.
		//	FIXME [check] >> this need to be checked.
		false  // is_classification.
    );

	// predict.
	Eigen::MatrixXd mScores;  // output scores.
	const int resultPredict = elmPredict(
		mTestX.data(), (int)dimTest, (int)numTestSample,  // inputs.
		mScores,  // outputs.
		inW, bias, outW  // inputs.
	);

	// performance evaluation.
	const double rmse = std::sqrt((mScores - vTestY).squaredNorm() / (double)numTestSample);  // root-mean-square error (RMSE).

	// display.
	std::cout << "RMSE = " << rmse << std::endl;

	// FIXME [check] >> this need to be checked.
	//std::cout << mScores.rows() << ", " << mScores.cols() << std::endl;
	//for (std::size_t i = 0; i < mScores.cols(); ++i)
	for (std::size_t i = 0; i < 10; ++i)
		std::cout << "predicted = " << mScores(i) << ", true = " << vTestY(i) << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_elm {

void elm_linear_test()
{
	local::elm_linear_classification_test();
	local::elm_linear_regression_test();
}

}  // namespace my_elm
