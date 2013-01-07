//#include "stdafx.h"
#include <map>
#include <list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>

#if defined(max)
#undef max
#endif


namespace {
namespace local {

const double PI = 4.0 * std::atan(1.0);

class RansacEstimator
{
protected:
	RansacEstimator(const size_t sampleCount, const size_t minimalSampleSetSize)
	: totalSampleCount_(sampleCount), minimalSampleSetSize_(minimalSampleSetSize), scores_(NULL), sortedIndices_(), inlierFlags_(), iteration_(0)
	{
	}
	RansacEstimator(const size_t sampleCount, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: totalSampleCount_(sampleCount), minimalSampleSetSize_(minimalSampleSetSize), scores_(&scores), sortedIndices_(), inlierFlags_(), iteration_(0)
	{
	}
public:
	virtual ~RansacEstimator()
	{
	}

public:
	virtual size_t runRANSAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double threshold);
	virtual size_t runMLESAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double inlierSquaredStandardDeviation, const double outlierUniformProbability, const size_t maxEMIterationCount);

	const std::vector<bool> & getInliers() const  {  return inlierFlags_;  }
	size_t getIterationCount() const  {  return iteration_;  }

protected:
	void drawRandomSample(const size_t maxCount, const size_t count, std::vector<size_t> &indices) const
	{
		for (size_t i = 0; i < count; )
		{
			const size_t idx = std::rand() % maxCount;
			std::vector<size_t>::iterator it = std::find(indices.begin(), indices.end(), idx);

			if (indices.end() == it)
				indices[i++] = idx;
		}
	}

	void drawProsacSample(const size_t maxCount, const size_t count, std::vector<size_t> &indices) const
	{
		for (size_t i = 0; i < count; )
		{
			const size_t idx = std::rand() % maxCount;
			std::vector<size_t>::iterator it = std::find(indices.begin(), indices.end(), idx);

			if (indices.end() == it)
				indices[i++] = idx;
		}

		for (std::vector<size_t>::iterator it = indices.begin(); it != indices.end(); ++it)
			*it = sortedIndices_[*it];
	}

	void sortSamples()
	{
		sortedIndices_.reserve(totalSampleCount_);
		for (size_t i = 0; i < totalSampleCount_; ++i)
			sortedIndices_.push_back(i);

		if (scores_ && !scores_->empty())
			std::sort(sortedIndices_.begin(), sortedIndices_.end(), CompareByScore(*scores_));
	}

private:
	virtual bool estimateModel(const std::vector<size_t> &indices) = 0;
	virtual bool verifyModel() const = 0;
	virtual bool estimateModelFromInliers() = 0;

	// for RANSAC
	virtual size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const = 0;
	// for MLESAC
	virtual void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const = 0;
	virtual size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const = 0;

private:
	struct CompareByScore
	{
	public:
		CompareByScore(const std::vector<double> &scores) : scores_(scores) {}

		bool operator()(const int lhs, const int rhs) const
		{  return scores_[lhs] > scores_[rhs];  }

	private:
		const std::vector<double> &scores_;
	};

protected:
	const size_t totalSampleCount_;
	const size_t minimalSampleSetSize_;

	const std::vector<double> *scores_;
	std::vector<size_t> sortedIndices_;

	std::vector<bool> inlierFlags_;
	size_t iteration_;
};

size_t RansacEstimator::runRANSAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double threshold)
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
	while (maxIteration > iteration_ && inlierCount < minInlierCount)
	{
		// draw a sample
		if (isProsacSampling)
		{
			drawProsacSample(prosacSampleCount, minimalSampleSetSize_, indices);

			// this incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleCount_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleCount_, minimalSampleSetSize_, indices);

		// estimate a model
		if (estimateModel(indices) && verifyModel())
		{
			// evaluate a model
			const size_t currInlierCount = lookForInliers(currInlierFlags, threshold);

			if (currInlierCount > inlierCount)
			{
				const double inlierRatio = double(currInlierCount) / totalSampleCount_;
				const size_t newMaxIteration = (size_t)std::floor(std::log(alarmRatio) / std::log(1.0 - std::pow(inlierRatio, (double)minimalSampleSetSize_)));
				if (newMaxIteration < maxIteration) maxIteration = newMaxIteration;

				inlierCount = currInlierCount;
				//for (size_t i = 0; i < totalSampleCount_; ++i) inlierFlags_[i] = currInlierFlags[i];
				inlierFlags_.swap(currInlierFlags);
			}
		}

		++iteration_;
	}

	// re-estimate with all inliers and loop until the number of inliers is not increased anymore
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

size_t RansacEstimator::runMLESAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double inlierSquaredStandardDeviation, const double outlierUniformProbability, const size_t maxEMIterationCount)
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
	while (maxIteration > iteration_ && inlierCount < minInlierCount)
	{
		// draw a sample
		if (isProsacSampling)
		{
			drawProsacSample(prosacSampleCount, minimalSampleSetSize_, indices);

			// this incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleCount_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleCount_, minimalSampleSetSize_, indices);

		// estimate a model
		if (estimateModel(indices) && verifyModel())
		{
			// compute inliers' probabilities
			computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

			// EM algorithm
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

			// evaluate a model
			const double outlierProb = (1.0 - gamma) * outlierUniformProbability;
			double negativeLogLikelihood = 0.0;
			for (size_t k = 0; k < totalSampleCount_; ++k)
				negativeLogLikelihood -= std::log(gamma * inlierProbs[k] + outlierProb);  // negative log likelihood

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

	// re-estimate with all inliers and loop until the number of inliers is not increased anymore
	size_t oldInlierCount = 0;
	do
	{
		if (!estimateModelFromInliers()) return -1;

		// compute inliers' probabilities
		computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

		oldInlierCount = inlierCount;
		inlierCount = lookForInliers(inlierFlags_, inlierProbs, outlierUniformProbability);
	} while (inlierCount > oldInlierCount);

	// compute inliers' probabilities
	computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

	inlierCount = lookForInliers(inlierFlags_, inlierProbs, outlierUniformProbability);

	return inlierCount;
}

struct Point3
{
	Point3(const double _x, const double _y, const double _z)
	: x(_x), y(_y), z(_z)
	{}
	Point3(const Point3 &rhs)
	: x(rhs.x), y(rhs.y), z(rhs.z)
	{}

	double x, y, z;
};

class Plane3RansacEstimator: public RansacEstimator
{
public:
	typedef RansacEstimator base_type;

public:
	Plane3RansacEstimator(const std::vector<Point3> &samples, const size_t minimalSampleSetSize)
	: base_type(samples.size(), minimalSampleSetSize), samples_(samples)
	{
	}
	Plane3RansacEstimator(const std::vector<Point3> &samples, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: base_type(samples.size(), minimalSampleSetSize, scores), samples_(samples)
	{
	}

public:
	double getA() const  {  return a_;  }
	double getB() const  {  return b_;  }
	double getC() const  {  return c_;  }
	double getD() const  {  return d_;  }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices);
	/*virtual*/ bool verifyModel() const;
	/*virtual*/ bool estimateModelFromInliers();

	// for RANSAC
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const;
	// for MLESAC
	/*virtual*/ void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const;
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const;

	bool calculateNormal(const double vx1, const double vy1, const double vz1, const double vx2, const double vy2, const double vz2, double &nx, double &ny, double &nz) const
	{
		nx = vy1 * vz2 - vz1 * vy2;
		ny = vz1 * vx2 - vx1 * vz2;
		nz = vx1 * vy2 - vy1 * vx2;

		const double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
		const double eps = 1.0e-20;
		if (norm < eps) return false;

		nx /= norm;
		ny /= norm;
		nz /= norm;
		return true;
	}

private:
	const std::vector<Point3> &samples_;

	// plane equation: a * x + b * y + c * z + d = 0
	double a_, b_, c_, d_;
};

bool Plane3RansacEstimator::estimateModel(const std::vector<size_t> &indices)
{
	if (indices.size() < minimalSampleSetSize_) return false;

	const Point3 &pt1 = samples_[indices[0]];
	const Point3 &pt2 = samples_[indices[1]];
	const Point3 &pt3 = samples_[indices[2]];

	if (calculateNormal(pt2.x - pt1.x, pt2.y - pt1.y, pt2.z - pt1.z, pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z, a_, b_, c_))
	{
		d_ = -(a_ * pt1.x + b_ * pt1.y + c_ * pt1.z);
		return true;
	}
	else return false;
}

bool Plane3RansacEstimator::verifyModel() const
{
	return true;
}

bool Plane3RansacEstimator::estimateModelFromInliers()
{
	// TODO [improve] >> (e.g.) can find the least squares solution from inliers
	return true;
}

size_t Plane3RansacEstimator::lookForInliers(std::vector<bool> &inliers, const double threshold) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = std::fabs(a_ * it->x + b_ * it->y + c_ * it->z + d_) < threshold;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

void Plane3RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double factor = 1.0 / std::sqrt(2.0 * PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// compute errors
		const double err = a_ * it->x + b_ * it->y + c_ * it->z + d_;

		// compute inliers' probabilities
		inlierProbs[k] = factor * std::exp(-0.5 * err * err / inlierSquaredStandardDeviation);
	}
}

size_t Plane3RansacEstimator::lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = inlierProbs[k] >= outlierUniformProbability;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

}  // namespace local
}  // unnamed namespace

namespace ransac {

void plane_estimation()
{
	const size_t N_plane = 30;
	const size_t N_noise = 100;

	// generate random points
	std::vector<local::Point3> samples;
	{
		const double PLANE_EQ[4] = { 1, -1, 1, -2 };  // { 0.5774, -0.5774, 0.5774, -1.1547 }

		for (size_t i = 0; i < N_plane; ++i)
		{
			const double x = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3]
			const double y = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3]
			const double z = -(PLANE_EQ[3] + PLANE_EQ[0] * x + PLANE_EQ[1] * y) / PLANE_EQ[2];
			samples.push_back(local::Point3(x, y, z));
		}

		for (size_t i = 0; i < N_noise; ++i)
		{
			const double x = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5]
			const double y = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5]
			const double z = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5]
			samples.push_back(local::Point3(x, y, z));
		}
	}

	const size_t minimalSampleSetSize = 3;
	local::Plane3RansacEstimator ransac(samples, minimalSampleSetSize);

	const size_t maxIterationCount = 500;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = false;

	std::cout << "********* RANSAC" << std::endl;
	{
		const double threshold = 0.2;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "the number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "the number of inliers: " << inlierCount << std::endl;
		std::cout << "indices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}

	std::cout << "********* MLESAC" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.15;
		const double outlierUniformProbability = 0.1;
		const size_t maxEMIterationCount = 10;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, outlierUniformProbability, maxEMIterationCount);

		std::cout << "the number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "the number of inliers: " << inlierCount << std::endl;
		std::cout << "indices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}
}

}  // namespace ransac
