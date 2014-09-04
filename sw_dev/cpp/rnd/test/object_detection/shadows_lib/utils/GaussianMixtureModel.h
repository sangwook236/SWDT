// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef GAUSSIANMIXTUREMODEL_H_
#define GAUSSIANMIXTUREMODEL_H_

#include <cxcore.h>

/**
 * GaussianMixtureModel class.
 */
class GaussianMixtureModel {

	public:
		enum SortMode {
			SORT_BY_WEIGHT, SORT_BY_WEIGHT_OVER_SD
		};

		std::vector<cv::Mat> means;
		std::vector<cv::Mat> variances;
		std::vector<cv::Mat> weights;
		std::vector<cv::Mat> counts;
		std::vector<cv::Mat> logisticVals;
		std::vector<cv::Mat> indices;

		GaussianMixtureModel();
		virtual ~GaussianMixtureModel();

		static float evaluateGaussian(int dimensions, const uchar* value, const uchar* mean, const ushort* variance);

		bool empty() const;
		void clear();
		void init(int gaussians, int rows, int cols, int dimensions, const std::vector<double>& initVars,
				const std::vector<double>& minVars, float stdThreshold, bool winnerTakesAll, float learningRate,
				SortMode sortMode, bool fitLogistic);
		void update(const cv::Mat& example, const cv::Mat& exampleWeights = cv::Mat());
		void evaluate(const cv::Mat& example, cv::Mat& result, double weightThresh, const cv::Mat& mask =
				cv::Mat()) const;
		void classify(const cv::Mat& example, cv::Mat& result, double weightThresh, const cv::Mat& mask =
				cv::Mat()) const;
		void estimateMean(cv::Mat& mean, double weightThreshold) const;

	private:
		float stdThreshold;
		bool winnerTakesAll;
		std::vector<double> initVars;
		std::vector<double> minVars;
		bool confidenceRated;
		float learningRate;
		SortMode sortMode;
		bool fitLogistic;

		float* xVals;
		float* aVals;
		float* bVals;

		static void setTo(cv::Mat& mat, const std::vector<double>& val);
		static void getLogisticVals(int gaussians, const float* xVals, const float* yVals, const float* aVals,
				const float* bVals, float* hVals);

		bool checkSizeAndType(const cv::Mat& example, const cv::Mat& exampleWeights = cv::Mat());
};

#endif /* GAUSSIANMIXTUREMODEL_H_ */
