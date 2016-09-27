#pragma once

#if !defined(__GDT_CPP_RND_STATISTICS__RANSAC__H_)
#define __GDT_CPP_RND_STATISTICS__RANSAC__H_ 1


#include <vector>


//--------------------------------------------------------------------------
// Random Sample Consensus (RANSAC).

class Ransac
{
protected:
	Ransac(const size_t sampleCount, const size_t minimalSampleSetSize)
	: totalSampleCount_(sampleCount), minimalSampleSetSize_(minimalSampleSetSize), scores_(NULL), sortedIndices_(), inlierFlags_(), iteration_(0)
	{}
	Ransac(const size_t sampleCount, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: totalSampleCount_(sampleCount), minimalSampleSetSize_(minimalSampleSetSize), scores_(&scores), sortedIndices_(), inlierFlags_(), iteration_(0)
	{}
public:
	virtual ~Ransac();

public:
	virtual size_t runRANSAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double threshold);
	virtual size_t runMLESAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double inlierSquaredStandardDeviation, const double outlierUniformProbability, const size_t maxEMIterationCount);

	const std::vector<bool> & getInliers() const { return inlierFlags_; }
	size_t getIterationCount() const { return iteration_; }

protected:
	void drawRandomSample(const size_t maxCount, const size_t count, const bool isProsacSampling, std::vector<size_t> &indices) const;
	void sortSamples();

private:
	virtual bool estimateModel(const std::vector<size_t> &indices) = 0;
	virtual bool verifyModel() const = 0;
	virtual bool estimateModelFromInliers() = 0;

	// For RANSAC.
	virtual size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const = 0;
	// For MLESAC.
	virtual void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const = 0;
	virtual size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const = 0;

protected:
	const size_t totalSampleCount_;
	const size_t minimalSampleSetSize_;

	const std::vector<double> *scores_;
	std::vector<size_t> sortedIndices_;

	std::vector<bool> inlierFlags_;
	size_t iteration_;
};


#endif  // __GDT_CPP_RND_STATISTICS__RANSAC__H_
