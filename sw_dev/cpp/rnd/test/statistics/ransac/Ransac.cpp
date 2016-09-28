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
	const size_t availableSampleSetSize = usedSampleSize_ > 0 ? std::max(usedSampleSize_, minimalSampleSize_) : minimalSampleSize_;
	if (totalSampleSize_ < availableSampleSetSize)
		return -1;

	if (isProsacSampling) sortSamples();

	size_t maxIteration = maxIterationCount;

	size_t inlierCount = 0;
	inlierFlags_.resize(totalSampleSize_, false);
	std::vector<bool> currInlierFlags(totalSampleSize_, false);

	std::vector<size_t> indices(availableSampleSetSize, -1);

	// TODO [check] >>
	size_t prosacSampleCount = 10;
	iteration_ = 0;
	while (iteration_ < maxIteration && inlierCount < minInlierCount)
	{
		// Draw a sample.
		if (isProsacSampling)
		{
			drawRandomSample(prosacSampleCount, availableSampleSetSize, true, indices);

			// This incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleSize_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleSize_, availableSampleSetSize, false, indices);

		// Estimate a model.
		if (estimateModel(indices) && verifyModel())
		{
			// Evaluate a model.
			const size_t currInlierCount = lookForInliers(currInlierFlags, threshold);

			if (currInlierCount > inlierCount)
			{
				const double inlierRatio = double(currInlierCount) / totalSampleSize_;
				const size_t newMaxIteration = (size_t)std::floor(std::log(alarmRatio) / std::log(1.0 - std::pow(inlierRatio, (double)availableSampleSetSize)));
				if (newMaxIteration < maxIteration) maxIteration = newMaxIteration;

				inlierCount = currInlierCount;
				inlierFlags_.swap(currInlierFlags);
			}
		}

		++iteration_;
	}

	// Re-estimate with all inliers and loop until the number of inliers does not increase anymore.
	if (inlierCount >= minimalSampleSize_)
	{
		size_t oldInlierCount = inlierCount;
		do
		{
			if (!estimateModelFromInliers()) return -1;

			oldInlierCount = inlierCount;
			inlierCount = lookForInliers(inlierFlags_, threshold);
		} while (inlierCount > oldInlierCount);

		inlierCount = lookForInliers(inlierFlags_, threshold);
	}

	return inlierCount;
}

size_t Ransac::runMLESAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double inlierSquaredStandardDeviation, const double inlierThresholdProbability, const size_t maxEMIterationCount)
{
	const size_t availableSampleSetSize = usedSampleSize_ > 0 ? std::max(usedSampleSize_, minimalSampleSize_) : minimalSampleSize_;
	if (totalSampleSize_ < availableSampleSetSize)
		return -1;

	if (isProsacSampling) sortSamples();

	size_t maxIteration = maxIterationCount;

	size_t inlierCount = 0;
	inlierFlags_.resize(totalSampleSize_, false);
	std::vector<double> inlierProbs(totalSampleSize_, 0.0);
	double minNegativeLogLikelihood = std::numeric_limits<double>::max();

	std::vector<size_t> indices(availableSampleSetSize, -1);

	// TODO [check] >>
	size_t prosacSampleCount = 10;
	iteration_ = 0;
	const double eps = 1.0e-10;
	while (iteration_ < maxIteration && inlierCount < minInlierCount)
	{
		// Draw a sample.
		if (isProsacSampling)
		{
			drawRandomSample(prosacSampleCount, availableSampleSetSize, true, indices);

			// This incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleSize_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleSize_, availableSampleSetSize, false, indices);

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
				const double outlierProb = (1.0 - gamma) * inlierThresholdProbability;
				double sumInlierProb = 0.0;
				for (size_t k = 0; k < totalSampleSize_; ++k)
				{
					const double inlierProb = gamma * inlierProbs[k];
					sumInlierProb += inlierProb / (inlierProb + outlierProb);
				}

				prevGamma = gamma;
				gamma = sumInlierProb / totalSampleSize_;

				if (std::abs(gamma - prevGamma) < tol) break;
			}

			// Evaluate a model.
			const double outlierProb = (1.0 - gamma) * inlierThresholdProbability;
			double negativeLogLikelihood = 0.0;
			for (size_t k = 0; k < totalSampleSize_; ++k)
				negativeLogLikelihood -= std::log(gamma * inlierProbs[k] + outlierProb);  // Negative log likelihood.

			if (negativeLogLikelihood < minNegativeLogLikelihood)
			{
				const double denom = std::log(1.0 - std::pow(gamma, (double)availableSampleSetSize));
				if (std::abs(denom) > eps)
				{
					const size_t newMaxIteration = (size_t)std::floor(std::log(alarmRatio) / denom);
					if (newMaxIteration < maxIteration) maxIteration = newMaxIteration;
				}

				inlierCount = lookForInliers(inlierFlags_, inlierProbs, inlierThresholdProbability);

				minNegativeLogLikelihood = negativeLogLikelihood;
			}
		}

		++iteration_;
	}

	// Re-estimate with all inliers and loop until the number of inliers does not increase anymore.
	if (inlierCount >= minimalSampleSize_)
	{
		size_t oldInlierCount = 0;
		do
		{
			if (!estimateModelFromInliers()) return inlierCount;

			// Compute inliers' probabilities.
			computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

			oldInlierCount = inlierCount;
			inlierCount = lookForInliers(inlierFlags_, inlierProbs, inlierThresholdProbability);
		} while (inlierCount > oldInlierCount);

		// Compute inliers' probabilities.
		computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

		inlierCount = lookForInliers(inlierFlags_, inlierProbs, inlierThresholdProbability);
	}

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
	sortedIndices_.reserve(totalSampleSize_);
	for (size_t i = 0; i < totalSampleSize_; ++i)
		sortedIndices_.push_back(i);

	if (scores_ && !scores_)
		std::sort(sortedIndices_.begin(), sortedIndices_.end(), local::CompareByScore(*scores_));
}
