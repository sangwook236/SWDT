#include "Ransac.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>


#if defined(max)
#undef max
#endif


namespace {
namespace local {

struct CompareByScore
{
public:
	CompareByScore(const std::vector<double> &scores)
	: scores_(scores)
	{}

	bool operator()(const int lhs, const int rhs) const
	{
		return scores_[lhs] > scores_[rhs];
	}

private:
	const std::vector<double> &scores_;
};

}  // namespace local
}  // unnamed namespace


/*virtual*/ Ransac::~Ransac()
{}

size_t Ransac::runRANSAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double threshold)
{
	if (totalSampleCount_ < minimalSampleSetSize_)
		return -1;

	if (isProsacSampling) sortSamples();

	size_t maxIteration = maxIterationCount;

	size_t inlierCount = 0;
	inlierFlags_.resize(totalSampleCount_, false);
	std::vector<bool> currInlierFlags(totalSampleCount_, false);

	std::vector<size_t> indices(minimalSampleSetSize_, -1);

	// TODO [check] >>
	size_t prosacSampleCount = 10;
	iteration_ = 0;
	while (iteration_ < maxIteration && inlierCount < minInlierCount)
	{
		// Draw a sample.
		if (isProsacSampling)
		{
			drawRandomSample(prosacSampleCount, minimalSampleSetSize_, true, indices);

			// This incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleCount_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleCount_, minimalSampleSetSize_, false, indices);

		// Estimate a model.
		if (estimateModel(indices) && verifyModel())
		{
			// Evaluate a model.
			const size_t currInlierCount = lookForInliers(currInlierFlags, threshold);

			if (currInlierCount > inlierCount)
			{
				const double inlierRatio = double(currInlierCount) / totalSampleCount_;
				const size_t newMaxIteration = (size_t)std::floor(std::log(alarmRatio) / std::log(1.0 - std::pow(inlierRatio, (double)minimalSampleSetSize_)));
				if (newMaxIteration < maxIteration) maxIteration = newMaxIteration;

				inlierCount = currInlierCount;
				inlierFlags_.swap(currInlierFlags);
			}
		}

		++iteration_;
	}

	// Re-estimate with all inliers and loop until the number of inliers does not increase anymore.
	size_t oldInlierCount = inlierCount;
	do
	{
		if (!estimateModelFromInliers()) return -1;

		oldInlierCount = inlierCount;
		inlierCount = lookForInliers(inlierFlags_, threshold);
	} while (inlierCount > oldInlierCount);

	inlierCount = lookForInliers(inlierFlags_, threshold);

	return inlierCount;
}

size_t Ransac::runMLESAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double inlierSquaredStandardDeviation, const double outlierUniformProbability, const size_t maxEMIterationCount)
{
	if (totalSampleCount_ < minimalSampleSetSize_)
		return -1;

	if (isProsacSampling) sortSamples();

	size_t maxIteration = maxIterationCount;

	size_t inlierCount = 0;
	inlierFlags_.resize(totalSampleCount_, false);
	std::vector<double> inlierProbs(totalSampleCount_, 0.0);
	double minNegativeLogLikelihood = std::numeric_limits<double>::max();

	std::vector<size_t> indices(minimalSampleSetSize_, -1);

	// TODO [check] >>
	size_t prosacSampleCount = 10;
	iteration_ = 0;
	while (iteration_ < maxIteration && inlierCount < minInlierCount)
	{
		// Draw a sample.
		if (isProsacSampling)
		{
			drawRandomSample(prosacSampleCount, minimalSampleSetSize_, true, indices);

			// This incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleCount_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleCount_, minimalSampleSetSize_, false, indices);

		// Estimate a model.
		if (estimateModel(indices) && verifyModel())
		{
			// Compute inliers' probabilities.
			computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

			// EM algorithm.
			const double tol = 1.0e-5;

			double gamma = 0.5, prevGamma;
			for (size_t i = 0; i < maxEMIterationCount; ++i)
			{
				const double outlierProb = (1.0 - gamma) * outlierUniformProbability;
				double sumInlierProb = 0.0;
				for (size_t k = 0; k < totalSampleCount_; ++k)
				{
					const double inlierProb = gamma * inlierProbs[k];
					sumInlierProb += inlierProb / (inlierProb + outlierProb);
				}

				prevGamma = gamma;
				gamma = sumInlierProb / totalSampleCount_;

				if (std::fabs(gamma - prevGamma) < tol) break;
			}

			// Evaluate a model.
			const double outlierProb = (1.0 - gamma) * outlierUniformProbability;
			double negativeLogLikelihood = 0.0;
			for (size_t k = 0; k < totalSampleCount_; ++k)
				negativeLogLikelihood -= std::log(gamma * inlierProbs[k] + outlierProb);  // Negative log likelihood.

			//
			if (negativeLogLikelihood < minNegativeLogLikelihood)
			{
				const size_t newMaxIteration = (size_t)std::floor(std::log(alarmRatio) / std::log(1.0 - std::pow(gamma, (double)minimalSampleSetSize_)));
				if (newMaxIteration < maxIteration) maxIteration = newMaxIteration;

				inlierCount = lookForInliers(inlierFlags_, inlierProbs, outlierUniformProbability);

				minNegativeLogLikelihood = negativeLogLikelihood;
			}
		}

		++iteration_;
	}

	// Re-estimate with all inliers and loop until the number of inliers does not increase anymore.
	size_t oldInlierCount = 0;
	do
	{
		if (!estimateModelFromInliers()) return -1;

		// Compute inliers' probabilities.
		computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

		oldInlierCount = inlierCount;
		inlierCount = lookForInliers(inlierFlags_, inlierProbs, outlierUniformProbability);
	} while (inlierCount > oldInlierCount);

	// Compute inliers' probabilities.
	computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

	inlierCount = lookForInliers(inlierFlags_, inlierProbs, outlierUniformProbability);

	return inlierCount;
}

void Ransac::drawRandomSample(const size_t maxCount, const size_t count, const bool isProsacSampling, std::vector<size_t> &indices) const
{
	for (size_t i = 0; i < count; )
	{
		const size_t idx = std::rand() % maxCount;
		std::vector<size_t>::iterator it = std::find(indices.begin(), indices.end(), idx);

		if (indices.end() == it)
			indices[i++] = idx;
	}

	if (isProsacSampling)
		for (std::vector<size_t>::iterator it = indices.begin(); it != indices.end(); ++it)
			*it = sortedIndices_[*it];
}

void Ransac::sortSamples()
{
	sortedIndices_.reserve(totalSampleCount_);
	for (size_t i = 0; i < totalSampleCount_; ++i)
		sortedIndices_.push_back(i);

	if (scores_ && !scores_->empty())
		std::sort(sortedIndices_.begin(), sortedIndices_.end(), local::CompareByScore(*scores_));
}
