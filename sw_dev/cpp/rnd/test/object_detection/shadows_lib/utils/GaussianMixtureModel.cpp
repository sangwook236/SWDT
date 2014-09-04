// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include <iostream>
#include <limits>
#include "GaussianMixtureModel.h"

GaussianMixtureModel::GaussianMixtureModel() {
	xVals = NULL;
	aVals = new float[3];
	aVals[0] = 0.5;
	aVals[1] = 1;
	aVals[2] = 2;
	bVals = NULL;
}

GaussianMixtureModel::~GaussianMixtureModel() {
	delete[] xVals;
	delete[] aVals;
	delete[] bVals;
}

float GaussianMixtureModel::evaluateGaussian(int dimensions, const uchar* value, const uchar* mean,
		const ushort* variance) {
	float nom = 0;
	float den = 1;
	for (int i = 0; i < dimensions; ++i) {
		float diff = (float) value[i] - mean[i];
		nom += (diff * diff) / variance[i];
		den *= 2 * CV_PI * variance[i];
	}

	nom = std::exp(-nom / 2);
	den = std::sqrt(den);

	return nom / den;
}

bool GaussianMixtureModel::empty() const {
	return means.empty();
}

void GaussianMixtureModel::clear() {
	delete[] xVals;
	xVals = NULL;
	delete[] bVals;
	bVals = NULL;

	means.clear();
	variances.clear();
	weights.clear();
	counts.clear();
	logisticVals.clear();
	indices.clear();
}

void GaussianMixtureModel::init(int gaussians, int rows, int cols, int dimensions, const std::vector<double>& initVars,
		const std::vector<double>& minVars, float stdThreshold, bool winnerTakesAll, float learningRate,
		SortMode sortMode, bool fitLogistic) {
	if (gaussians > 0 && rows > 0 && cols > 0 && dimensions > 0) {
		this->initVars = initVars;
		this->minVars = minVars;
		this->stdThreshold = stdThreshold;
		this->winnerTakesAll = winnerTakesAll;
		this->learningRate = learningRate;
		this->sortMode = sortMode;
		this->fitLogistic = fitLogistic;

		means.resize(gaussians);
		variances.resize(gaussians);
		weights.resize(gaussians);
		counts.resize(gaussians);
		logisticVals.resize(gaussians);
		indices.resize(gaussians);

		std::vector<double> initMeans(dimensions, std::numeric_limits<double>::infinity());
		for (int k = 0; k < gaussians; ++k) {
			means[k].create(rows, cols, CV_MAKETYPE(CV_8U, dimensions));
			setTo(means[k], initMeans);
			variances[k].create(rows, cols, CV_MAKETYPE(CV_16U, dimensions));
			setTo(variances[k], initVars);
			weights[k].create(rows, cols, CV_32F);
			weights[k].setTo(cv::Scalar(0));
			counts[k].create(rows, cols, CV_32F);
			counts[k].setTo(cv::Scalar(0));
			logisticVals[k].create(rows, cols, CV_32F);
			indices[k].create(rows, cols, CV_32S);
			indices[k].setTo(k);
		}

		delete[] xVals;
		xVals = new float[gaussians];
		delete[] bVals;
		bVals = new float[gaussians - 1];for
(		int k = 0; k < gaussians; ++k) {
			xVals[k] = (float) k / (gaussians - 1);
			xVals[k] *= 20;
			xVals[k] -= 10;

			if (k > 0) {
				bVals[k - 1] = (xVals[k - 1] + xVals[k]) / 2;
			}
		}
	}
	else {
		std::cerr << "Invalid initialisation" << std::endl;
	}
}

void GaussianMixtureModel::update(const cv::Mat& example, const cv::Mat& exampleWeights) {
	if (checkSizeAndType(example, exampleWeights)) {
		int gaussians = means.size();
		int dimensions = example.channels();

		//--S [] 2014/08/28: Sang-Wook Lee
		/*
		uchar* meanPtrs[gaussians];
		ushort* variancePtrs[gaussians];
		float* weightPtrs[gaussians];
		float* countPtrs[gaussians];
		float* logisticValPtrs[gaussians];
		int* indexPtrs[gaussians];

		float p[gaussians];

		float yVals[gaussians];
		float hVals[gaussians];
		*/
		std::vector<uchar*> meanPtrs(gaussians, NULL);
		std::vector<ushort*> variancePtrs(gaussians, NULL);
		std::vector<float*> weightPtrs(gaussians, NULL);
		std::vector<float*> countPtrs(gaussians, NULL);
		std::vector<float*> logisticValPtrs(gaussians, NULL);
		std::vector<int*> indexPtrs(gaussians, NULL);

		std::vector<float> p(gaussians, 0.0f);

		std::vector<float> yVals(gaussians, 0.0f);
		std::vector<float> hVals(gaussians, 0.0f);
		//--E [] 2014/08/28: Sang-Wook Lee

		for (int y = 0; y < example.rows; ++y) {
			const uchar* examplePtr = example.ptr(y);
			const float* exampleWeightsPtr = (!exampleWeights.empty() ? exampleWeights.ptr<float>(y) : NULL);
			for (int k = 0; k < gaussians; ++k) {
				meanPtrs[k] = means[k].ptr(y);
				variancePtrs[k] = variances[k].ptr<ushort>(y);
				weightPtrs[k] = weights[k].ptr<float>(y);
				countPtrs[k] = counts[k].ptr<float>(y);
				logisticValPtrs[k] = logisticVals[k].ptr<float>(y);
				indexPtrs[k] = indices[k].ptr<int>(y);
			}

			for (int x = 0; x < example.cols; ++x) {
				float weightedLearningRate = (exampleWeightsPtr ? exampleWeightsPtr[x] * learningRate : learningRate);
				if (weightedLearningRate > 0) {
					int rowPtrShift = x * dimensions;

					// find probability of current value and each Gaussian
					float pSum = 0;
					for (int k = 0; k < gaussians; ++k) {
						p[k] = 0;
					}
					for (int k = 0; k < gaussians; ++k) {
						int index = indexPtrs[k][x];

						bool allLessThan = true;
						for (int d = 0; d < dimensions; ++d) {
							float diff = std::abs(
									(float) examplePtr[rowPtrShift + d] - meanPtrs[index][rowPtrShift + d]);
							float thresh = stdThreshold * std::sqrt((float)variancePtrs[index][rowPtrShift + d]);

							if (diff > thresh) {
								allLessThan = false;
								break;
							}
						}

						if (allLessThan) {
							if (winnerTakesAll) {
								p[index] = 1;
								pSum = 1;
								break;
							}
							else {
								p[index] = weightPtrs[index][x]
										* evaluateGaussian(dimensions, &examplePtr[rowPtrShift],
												&meanPtrs[index][rowPtrShift], &variancePtrs[index][rowPtrShift]);
								pSum += p[index];
							}
						}
					}

					if (pSum > 0) {
						// update weights
						for (int k = 0; k < gaussians; ++k) {
							p[k] /= pSum;
							weightPtrs[k][x] = (1 - weightedLearningRate) * weightPtrs[k][x]
									+ weightedLearningRate * p[k];
						}

						// update Gaussians
						for (int k = 0; k < gaussians; ++k) {
							if (p[k] > 0) {
								countPtrs[k][x] += p[k];
								float updateRate = p[k]
										* ((1 - weightedLearningRate) / countPtrs[k][x] + weightedLearningRate);

								for (int d = 0; d < dimensions; ++d) {
									uchar prevMean = meanPtrs[k][rowPtrShift + d];
									meanPtrs[k][rowPtrShift + d] = (1 - updateRate) * prevMean
											+ updateRate * examplePtr[rowPtrShift + d];
									int diff = (int) examplePtr[rowPtrShift + d] - prevMean;
									variancePtrs[k][rowPtrShift + d] = (1 - updateRate)
											* variancePtrs[k][rowPtrShift + d] + updateRate * diff * diff;
									if (variancePtrs[k][rowPtrShift + d] < minVars[d]) {
										variancePtrs[k][rowPtrShift + d] = minVars[d];
									}
								}
							}
						}
					}
					else {
						// update weights
						for (int k = 0; k < gaussians; ++k) {
							weightPtrs[k][x] = (1 - weightedLearningRate) * weightPtrs[k][x];
						}

						// create new Gaussian
						int lastIndex = indexPtrs[gaussians - 1][x];
						for (int d = 0; d < dimensions; ++d) {
							meanPtrs[lastIndex][rowPtrShift + d] = examplePtr[rowPtrShift + d];
							variancePtrs[lastIndex][rowPtrShift + d] = initVars[d];
						}
						weightPtrs[lastIndex][x] = weightedLearningRate;
						countPtrs[lastIndex][x] = 1;
					}

					// normalise weights
					float totalWeight = 0;
					for (int k = 0; k < gaussians; ++k) {
						totalWeight += weightPtrs[k][x];
					}
					for (int k = 0; k < gaussians; ++k) {
						weightPtrs[k][x] /= totalWeight;
					}

					// sort Gaussians
					int rightIndex = indexPtrs[gaussians - 1][x];
					float currFitness = weightPtrs[rightIndex][x];
					float varSum = 0;
					if (sortMode == SORT_BY_WEIGHT_OVER_SD) {
						for (int d = 0; d < dimensions; ++d) {
							varSum += variancePtrs[rightIndex][rowPtrShift + d];
						}

						currFitness /= varSum;
					}
					for (int k = gaussians - 2; k >= 0; --k) {
						int leftIndex = indexPtrs[k][x];
						float prevFitness = weightPtrs[leftIndex][x];
						varSum = 0;
						if (sortMode == SORT_BY_WEIGHT_OVER_SD) {
							for (int d = 0; d < dimensions; ++d) {
								varSum += variancePtrs[leftIndex][rowPtrShift + d];
							}

							prevFitness /= varSum;
						}

						if (currFitness > prevFitness) {
							indexPtrs[k + 1][x] = leftIndex;
							indexPtrs[k][x] = rightIndex;
						}
						else {
							rightIndex = leftIndex;
							currFitness = prevFitness;
						}
					}

					// fit sort values to logistic function
					if (fitLogistic) {
						float yMin = std::numeric_limits<float>::infinity();
						float yMax = 0;
						for (int k = 0; k < gaussians; ++k) {
							int index = indexPtrs[(gaussians - 1) - k][x];
							yVals[k] = weightPtrs[index][x];
							varSum = 0;
							if (sortMode == SORT_BY_WEIGHT_OVER_SD) {
								for (int d = 0; d < dimensions; ++d) {
									varSum += variancePtrs[index][rowPtrShift + d];
								}

								yVals[k] /= varSum;
							}

							if (yVals[k] < yMin) {
								yMin = yVals[k];
							}
							if (yVals[k] > yMax) {
								yMax = yVals[k];
							}
						}
						yMax -= yMin;

						for (int k = 0; k < gaussians; ++k) {
							yVals[k] = (yVals[k] - yMin) / yMax;
						}

						//--S [] 2014/08/28: Sang-Wook Lee
						//getLogisticVals(gaussians, xVals, yVals, aVals, bVals, hVals);
						getLogisticVals(gaussians, xVals, &yVals[0], aVals, bVals, &hVals[0]);
						//--E [] 2014/08/28: Sang-Wook Lee
						for (int k = 0; k < gaussians; ++k) {
							logisticValPtrs[indexPtrs[k][x]][x] = hVals[(gaussians - 1) - k];
						}
					}
				}
			}
		}
	}
	else {
		std::cerr << "Invalid example or model not initialised" << std::endl;
	}
}

void GaussianMixtureModel::evaluate(const cv::Mat& example, cv::Mat& result, double weightThresh,
		const cv::Mat& mask) const {
	if (!means.empty()) {
		result.create(means[0].size(), CV_32F);
		result.setTo(cv::Scalar(0));

		int gaussians = means.size();
		int dimensions = example.channels();

		//--S [] 2014/08/28: Sang-Wook Lee
		/*
		const uchar* meanPtrs[gaussians];
		const ushort* variancePtrs[gaussians];
		const float* weightPtrs[gaussians];
		const int* indexPtrs[gaussians];
		*/
		std::vector<const uchar*> meanPtrs(gaussians, NULL);
		std::vector<const ushort*> variancePtrs(gaussians, NULL);
		std::vector<const float*> weightPtrs(gaussians, NULL);
		std::vector<const int*> indexPtrs(gaussians, NULL);
		//--E [] 2014/08/28: Sang-Wook Lee

		for (int y = 0; y < example.rows; ++y) {
			const uchar* examplePtr = example.ptr(y);
			const uchar* maskPtr = (!mask.empty() ? mask.ptr(y) : NULL);
			float* resultPtr = result.ptr<float>(y);
			for (int k = 0; k < gaussians; ++k) {
				meanPtrs[k] = means[k].ptr(y);
				variancePtrs[k] = variances[k].ptr<ushort>(y);
				weightPtrs[k] = weights[k].ptr<float>(y);
				indexPtrs[k] = indices[k].ptr<int>(y);
			}

			for (int x = 0; x < example.cols; ++x) {
				if (!maskPtr || maskPtr[x] > 0) {
					float totalWeight = 0;
					for (int k = 0; k < gaussians && totalWeight < weightThresh; ++k) {
						int index = indexPtrs[k][x];

						bool allLessThan = true;
						for (int d = 0; d < dimensions; ++d) {
							float diff = std::abs(
									(float) examplePtr[x * dimensions + d] - meanPtrs[index][x * dimensions + d]);
							float thresh = stdThreshold * std::sqrt((float)variancePtrs[index][x * dimensions + d]);

							if (diff > thresh) {
								allLessThan = false;
								break;
							}
						}

						if (allLessThan) {
							double p = evaluateGaussian(dimensions, &examplePtr[x * dimensions],
									&meanPtrs[index][x * dimensions], &variancePtrs[index][x * dimensions]);
							double maxP = evaluateGaussian(dimensions, &meanPtrs[index][x * dimensions],
									&meanPtrs[index][x * dimensions], &variancePtrs[index][x * dimensions]);

							//--S [] 2014/08/28: Sang-Wook Lee
							//resultPtr[x] = log2(p / maxP + 1);
							resultPtr[x] = log(p / maxP + 1) / log(2.);
							//--E [] 2014/08/28: Sang-Wook Lee
							break;
						}

						totalWeight += weightPtrs[index][x];
					}
				}
			}
		}
	}
	else {
		result.release();
	}
}

void GaussianMixtureModel::classify(const cv::Mat& example, cv::Mat& result, double weightThresh,
		const cv::Mat& mask) const {
	if (!means.empty()) {
		result.create(means[0].size(), CV_8U);
		result.setTo(cv::Scalar(0));

		int gaussians = means.size();
		int dimensions = example.channels();

		//--S [] 2014/08/28: Sang-Wook Lee
		/*
		const uchar* meanPtrs[gaussians];
		const ushort* variancePtrs[gaussians];
		const float* weightPtrs[gaussians];
		const int* indexPtrs[gaussians];
		*/
		std::vector<const uchar*> meanPtrs(gaussians, NULL);
		std::vector<const ushort*> variancePtrs(gaussians, NULL);
		std::vector<const float*> weightPtrs(gaussians, NULL);
		std::vector<const int*> indexPtrs(gaussians, NULL);
		//--E [] 2014/08/28: Sang-Wook Lee

		for (int y = 0; y < example.rows; ++y) {
			const uchar* examplePtr = example.ptr(y);
			const uchar* maskPtr = (!mask.empty() ? mask.ptr(y) : NULL);
			uchar* resultPtr = result.ptr(y);
			for (int k = 0; k < gaussians; ++k) {
				meanPtrs[k] = means[k].ptr(y);
				variancePtrs[k] = variances[k].ptr<ushort>(y);
				weightPtrs[k] = weights[k].ptr<float>(y);
				indexPtrs[k] = indices[k].ptr<int>(y);
			}

			for (int x = 0; x < example.cols; ++x) {
				if (!maskPtr || maskPtr[x] > 0) {
					float totalWeight = 0;
					for (int k = 0; k < gaussians && totalWeight < weightThresh; ++k) {
						int index = indexPtrs[k][x];

						bool allLessThan = true;
						for (int d = 0; d < dimensions; ++d) {
							float diff = std::abs(
									(float) examplePtr[x * dimensions + d] - meanPtrs[index][x * dimensions + d]);
							float thresh = stdThreshold * std::sqrt((float)variancePtrs[index][x * dimensions + d]);

							if (diff > thresh) {
								allLessThan = false;
								break;
							}
						}

						if (allLessThan) {
							resultPtr[x] = 255;
							break;
						}

						totalWeight += weightPtrs[index][x];
					}
				}
			}
		}
	}
	else {
		result.release();
	}
}

void GaussianMixtureModel::estimateMean(cv::Mat& mean, double weightThreshold) const {
	if (!means.empty()) {
		mean.create(means[0].size(), means[0].type());

		int gaussians = means.size();
		int dimensions = means[0].channels();
		//--S [] 2014/08/28: Sang-Wook Lee
		/*
		const uchar* meanPtrs[gaussians];
		const float* weightPtrs[gaussians];
		const int* indexPtrs[gaussians];
		*/
		std::vector<const uchar*> meanPtrs(gaussians, NULL);
		std::vector<const float*> weightPtrs(gaussians, NULL);
		std::vector<const int*> indexPtrs(gaussians, NULL);
		//--E [] 2014/08/28: Sang-Wook Lee

		for (int y = 0; y < means[0].rows; ++y) {
			uchar* meanPtr = mean.ptr(y);

			for (int k = 0; k < gaussians; ++k) {
				meanPtrs[k] = means[k].ptr(y);
				weightPtrs[k] = weights[k].ptr<float>(y);
				indexPtrs[k] = indices[k].ptr<int>(y);
			}

			for (int x = 0; x < means[0].cols; ++x) {
				float weightSum = 0;
				//--S [] 2014/08/28: Sang-Wook Lee
				//float vals[dimensions];
				std::vector<float> vals(dimensions, 0.0f);
				//--E [] 2014/08/28: Sang-Wook Lee
				for (int d = 0; d < dimensions; ++d) {
					vals[d] = 0;
				}

				for (int k = 0; k < gaussians && weightSum <= weightThreshold; ++k) {
					int pos = indexPtrs[k][x];

					for (int d = 0; d < dimensions; ++d) {
						vals[d] += weightPtrs[pos][x] * meanPtrs[pos][x * dimensions + d];
					}
					weightSum += weightPtrs[pos][x];
				}

				for (int d = 0; d < dimensions; ++d) {
					meanPtr[x * dimensions + d] = vals[d] / weightSum;
				}
			}
		}
	}
	else {
		mean.release();
	}
}

void GaussianMixtureModel::setTo(cv::Mat& mat, const std::vector<double>& val) {
	if (mat.depth() == CV_8U) {
		int channels = mat.channels();
		int rows = mat.rows;
		int cols = mat.cols;
		if (mat.isContinuous()) {
			cols *= rows;
			rows = 1;
		}

		for (int y = 0; y < rows; y++) {
			uchar* ptr = mat.ptr(y);

			for (int x = 0; x < cols; x++) {
				for (int c = 0; c < channels && c < (int) val.size(); ++c) {
					ptr[x * channels + c] = val[c];
				}
			}
		}
	}
	else if (mat.depth() == CV_16U) {
		int channels = mat.channels();
		int rows = mat.rows;
		int cols = mat.cols;
		if (mat.isContinuous()) {
			cols *= rows;
			rows = 1;
		}

		for (int y = 0; y < rows; y++) {
			ushort* ptr = mat.ptr<ushort>(y);

			for (int x = 0; x < cols; x++) {
				for (int c = 0; c < channels && c < (int) val.size(); ++c) {
					ptr[x * channels + c] = val[c];
				}
			}
		}
	}
	else {
		std::cerr << "Unsupported mat depth" << std::endl;
	}
}

void GaussianMixtureModel::getLogisticVals(int gaussians, const float* xVals, const float* yVals, const float* aVals,
		const float* bVals, float* hVals) {
	float initA = 1;
	float bestB = 0;
	//--S [] 2014/08/28: Sang-Wook Lee
	//float h[gaussians];
	std::vector<float> h(gaussians, 0.0f);
	//--S [] 2014/08/28: Sang-Wook Lee
	float minError = std::numeric_limits<float>::infinity();

	float yy = 0;
	for (int k = 0; k < gaussians; ++k) {
		yy += yVals[k] * yVals[k];
	}

	for (int b = 0; b < gaussians - 1; ++b) { // TODO bSize
		float hMin = std::numeric_limits<float>::infinity();
		float hMax = 0;
		for (int k = 0; k < gaussians; ++k) {
			h[k] = 1.0 / (1 + std::exp(-initA * xVals[k] + bVals[b]));

			if (h[k] < hMin) {
				hMin = h[k];
			}
			if (h[k] > hMax) {
				hMax = h[k];
			}
		}
		hMax -= hMin;

		float hy = 0;
		float hh = 0;
		for (int k = 0; k < gaussians; ++k) {
			h[k] = (h[k] - hMin) / hMax;
			hy += h[k] * yVals[k];
			hh += h[k] * h[k];
		}

		float error = yy - (hy * hy) / hh;
		if (error < minError) {
			minError = error;
			bestB = bVals[b];

			for (int k = 0; k < gaussians; ++k) {
				hVals[k] = h[k];
			}
		}
	}

	for (int a = 0; a < 3; ++a) { // TODO aSize
		if (aVals[a] != initA) {
			float hMin = std::numeric_limits<float>::infinity();
			float hMax = 0;
			for (int k = 0; k < gaussians; ++k) {
				h[k] = 1.0 / (1 + std::exp(-aVals[a] * xVals[k] + bestB));

				if (h[k] < hMin) {
					hMin = h[k];
				}
				if (h[k] > hMax) {
					hMax = h[k];
				}
			}
			hMax -= hMin;

			float hy = 0;
			float hh = 0;
			for (int k = 0; k < gaussians; ++k) {
				h[k] = (h[k] - hMin) / hMax;
				hy += h[k] * yVals[k];
				hh += h[k] * h[k];
			}

			float error = yy - (hy * hy) / hh;
			if (error < minError) {
				minError = error;

				for (int k = 0; k < gaussians; ++k) {
					hVals[k] = h[k];
				}
			}
		}
	}
}

bool GaussianMixtureModel::checkSizeAndType(const cv::Mat& example, const cv::Mat& exampleWeights) {
	if (example.empty() || example.depth() != CV_8U) {
		return false;
	}

	if (!exampleWeights.empty()) {
		if (exampleWeights.size() != example.size() || exampleWeights.type() != CV_32F) {
			return false;
		}
	}

	if (means.empty() || example.size() != means[0].size() || example.type() != means[0].type()) {
		return false;
	}

	return true;
}
