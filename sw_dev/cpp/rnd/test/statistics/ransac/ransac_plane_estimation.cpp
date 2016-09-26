//#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <list>
#include <vector>
#include <limits>
#include <cmath>

#if defined(max)
#undef max
#endif


namespace {
namespace local {

const double PI = 4.0 * std::atan(1.0);

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
	virtual ~Ransac()
	{}

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

	// For RANSAC.
	virtual size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const = 0;
	// For MLESAC.
	virtual void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const = 0;
	virtual size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const = 0;

private:
	struct CompareByScore
	{
	public:
		CompareByScore(const std::vector<double> &scores)
		: scores_(scores)
		{}

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
	while (maxIteration > iteration_ && inlierCount < minInlierCount)
	{
		// Draw a sample.
		if (isProsacSampling)
		{
			drawProsacSample(prosacSampleCount, minimalSampleSetSize_, indices);

			// This incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleCount_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleCount_, minimalSampleSetSize_, indices);

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
				//for (size_t i = 0; i < totalSampleCount_; ++i) inlierFlags_[i] = currInlierFlags[i];
				inlierFlags_.swap(currInlierFlags);
			}
		}

		++iteration_;
	}

	// Re-estimate with all inliers and loop until the number of inliers is not increased anymore.
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
	while (maxIteration > iteration_ && inlierCount < minInlierCount)
	{
		// Draw a sample.
		if (isProsacSampling)
		{
			drawProsacSample(prosacSampleCount, minimalSampleSetSize_, indices);

			// This incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleCount_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleCount_, minimalSampleSetSize_, indices);

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

	// Re-estimate with all inliers and loop until the number of inliers is not increased anymore.
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

struct Point2
{
	Point2(const double _x, const double _y)
	: x(_x), y(_y)
	{}
	Point2(const Point2 &rhs)
	: x(rhs.x), y(rhs.y)
	{}

	double x, y;
};

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

class Line2RansacEstimator : public Ransac
{
public:
	typedef Ransac base_type;

public:
	Line2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSetSize)
	: base_type(samples.size(), minimalSampleSetSize), samples_(samples)
	{}
	Line2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: base_type(samples.size(), minimalSampleSetSize, scores), samples_(samples)
	{}

public:
	double getA() const { return a_; }
	double getB() const { return b_; }
	double getC() const { return c_; }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices);
	/*virtual*/ bool verifyModel() const;
	/*virtual*/ bool estimateModelFromInliers();

	// For RANSAC.
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const;
	// For MLESAC.
	/*virtual*/ void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const;
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const;

private:
	const std::vector<Point2> &samples_;

	// Line equation: a * x + b * y + c = 0.
	double a_, b_, c_;
};

bool Line2RansacEstimator::estimateModel(const std::vector<size_t> &indices)
{
	if (indices.size() < minimalSampleSetSize_) return false;

	const Point2 &pt1 = samples_[indices[0]];
	const Point2 &pt2 = samples_[indices[1]];

	a_ = pt2.y - pt1.y;
	b_ = pt1.x - pt2.x;
	c_ = -a_ * pt1.x - b_ * pt1.y;

	return true;
}

bool Line2RansacEstimator::verifyModel() const
{
	return true;
}

bool Line2RansacEstimator::estimateModelFromInliers()
{
	// TODO [improve] >> For example, can find the least squares solution from inliers.
	return true;
}

size_t Line2RansacEstimator::lookForInliers(std::vector<bool> &inliers, const double threshold) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = std::fabs(a_ * it->x + b_ * it->y + c_) < threshold;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

void Line2RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double factor = 1.0 / std::sqrt(2.0 * PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute errors.
		const double err = a_ * it->x + b_ * it->y + c_;

		// Compute inliers' probabilities.
		inlierProbs[k] = factor * std::exp(-0.5 * err * err / inlierSquaredStandardDeviation);
	}
}

size_t Line2RansacEstimator::lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = inlierProbs[k] >= outlierUniformProbability;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

class Circle2RansacEstimator : public Ransac
{
public:
	typedef Ransac base_type;

public:
	Circle2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSetSize)
	: base_type(samples.size(), minimalSampleSetSize), samples_(samples)
	{}
	Circle2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: base_type(samples.size(), minimalSampleSetSize, scores), samples_(samples)
	{}

public:
	double getA() const { return a_; }
	double getB() const { return b_; }
	double getC() const { return c_; }
	double getD() const { return d_; }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices);
	/*virtual*/ bool verifyModel() const;
	/*virtual*/ bool estimateModelFromInliers();

	// For RANSAC.
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const;
	// For MLESAC.
	/*virtual*/ void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const;
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const;

private:
	const std::vector<Point2> &samples_;

	// Circle equation: a * x^2 + a * y^2 + b * x + c * y + d = 0.
	double a_, b_, c_, d_;
};

bool Circle2RansacEstimator::estimateModel(const std::vector<size_t> &indices)
{
	if (indices.size() < minimalSampleSetSize_) return false;

	const Point2 &pt1 = samples_[indices[0]];
	const Point2 &pt2 = samples_[indices[1]];
	const Point2 &pt3 = samples_[indices[2]];

	const double x1 = pt1.x, y1 = pt1.y, x1_2 = x1 * x1, y1_2 = y1 * y1;
	const double x2 = pt2.x, y2 = pt2.y, x2_2 = x2 * x2, y2_2 = y2 * y2;
	const double x3 = pt3.x, y3 = pt3.y, x3_2 = x3 * x3, y3_2 = y3 * y3;

	a_ = x1*(y3 - y2) - x2*y3 + x3*y2 + (x2 - x3)*y1;
	b_ = (y1*(y3_2 - y2_2 + x3_2 - x2_2) + y2*(-y3_2 - x3_2) + y2_2*y3 + x2_2*y3 + y1_2*(y2 - y3) + x1_2*(y2 - y3));
	c_ = -(x1*(y3_2 - y2_2 + x3_2 - x2_2) + x2*(-y3_2 - x3_2) + x3*y2_2 + (x2 - x3)*y1_2 + x2_2*x3 + x1_2*(x2 - x3));
	d_ = -(y1*(x2*(y3_2 + x3_2) - x3*y2_2 - x2_2*x3) + x1*(y2*(-y3_2 - x3_2) + y2_2*y3 + x2_2*y3) + y1_2*(x3*y2 - x2*y3) + x1_2*(x3*y2 - x2*y3));

	return true;
}

bool Circle2RansacEstimator::verifyModel() const
{
	return true;
}

bool Circle2RansacEstimator::estimateModelFromInliers()
{
	// TODO [improve] >> For example, can find the least squares solution from inliers.
	return true;
}

size_t Circle2RansacEstimator::lookForInliers(std::vector<bool> &inliers, const double threshold) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = std::fabs(a_ * it->x * it->x + a_ * it->y * it->y + b_ * it->x + c_ * it->y + d_) < threshold;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

void Circle2RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double factor = 1.0 / std::sqrt(2.0 * PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute errors.
		const double err = a_ * it->x * it->x + a_ * it->y * it->y + b_ * it->x + c_ * it->y + d_;

		// Compute inliers' probabilities.
		inlierProbs[k] = factor * std::exp(-0.5 * err * err / inlierSquaredStandardDeviation);
	}
}

size_t Circle2RansacEstimator::lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = inlierProbs[k] >= outlierUniformProbability;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

class Quadratic2RansacEstimator : public Ransac
{
public:
	typedef Ransac base_type;

public:
	Quadratic2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSetSize)
	: base_type(samples.size(), minimalSampleSetSize), samples_(samples)
	{}
	Quadratic2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: base_type(samples.size(), minimalSampleSetSize, scores), samples_(samples)
	{}

public:
	double getA() const { return a_; }
	double getB() const { return b_; }
	double getC() const { return c_; }
	double getD() const { return d_; }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices);
	/*virtual*/ bool verifyModel() const;
	/*virtual*/ bool estimateModelFromInliers();

	// For RANSAC.
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const;
	// For MLESAC.
	/*virtual*/ void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const;
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const;

private:
	const std::vector<Point2> &samples_;

	// Quadratic curve equation: a * x^2 + b * x + c * y + d = 0.
	double a_, b_, c_, d_;
};

bool Quadratic2RansacEstimator::estimateModel(const std::vector<size_t> &indices)
{
	if (indices.size() < minimalSampleSetSize_) return false;

	const Point2 &pt1 = samples_[indices[0]];
	const Point2 &pt2 = samples_[indices[1]];
	const Point2 &pt3 = samples_[indices[2]];

	const double x1 = pt1.x, y1 = pt1.y, x1_2 = x1 * x1;
	const double x2 = pt2.x, y2 = pt2.y, x2_2 = x2 * x2;
	const double x3 = pt3.x, y3 = pt3.y, x3_2 = x3 * x3;

	a_ = x1*(y3 - y2) - x2*y3 + x3*y2 + (x2 - x3)*y1;
	b_ = x1_2*(y3 - y2) - x2_2*y3 + x3_2*y2 + (x2_2 - x3_2)*y1;
	c_ = x1*(x3_2 - x2_2) - x2*x3_2 + x2_2*x3 + x1_2*(x2 - x3);
	d_ = x1*(x3_2*y2 - x2_2*y3) + x1_2*(x2*y3 - x3*y2) + (x2_2*x3 - x2*x3_2)*y1;

	return true;
}

bool Quadratic2RansacEstimator::verifyModel() const
{
	return true;
}

bool Quadratic2RansacEstimator::estimateModelFromInliers()
{
	// TODO [improve] >> For example, can find the least squares solution from inliers.
	return true;
}

size_t Quadratic2RansacEstimator::lookForInliers(std::vector<bool> &inliers, const double threshold) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = std::fabs(a_ * it->x * it->x + b_ * it->x + c_ * it->y + d_) < threshold;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

void Quadratic2RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double factor = 1.0 / std::sqrt(2.0 * PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute errors.
		const double err = a_ * it->x * it->x + b_ * it->x + c_ * it->y + d_;

		// Compute inliers' probabilities.
		inlierProbs[k] = factor * std::exp(-0.5 * err * err / inlierSquaredStandardDeviation);
	}
}

size_t Quadratic2RansacEstimator::lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inliers[k] = inlierProbs[k] >= outlierUniformProbability;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

class Plane3RansacEstimator: public Ransac
{
public:
	typedef Ransac base_type;

public:
	Plane3RansacEstimator(const std::vector<Point3> &samples, const size_t minimalSampleSetSize)
	: base_type(samples.size(), minimalSampleSetSize), samples_(samples)
	{}
	Plane3RansacEstimator(const std::vector<Point3> &samples, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: base_type(samples.size(), minimalSampleSetSize, scores), samples_(samples)
	{}

public:
	double getA() const  {  return a_;  }
	double getB() const  {  return b_;  }
	double getC() const  {  return c_;  }
	double getD() const  {  return d_;  }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices);
	/*virtual*/ bool verifyModel() const;
	/*virtual*/ bool estimateModelFromInliers();

	// For RANSAC.
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const;
	// For MLESAC.
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

	// Plane equation: a * x + b * y + c * z + d = 0.
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
	// TODO [improve] >> For example, can find the least squares solution from inliers.
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
		// Compute errors.
		const double err = a_ * it->x + b_ * it->y + c_ * it->z + d_;

		// Compute inliers' probabilities.
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

namespace my_ransac {

void line2_estimation()
{
	const double LINE_EQN[3] = { 2, 3, -1 };  // 2 * x + 3 * y - 1 = 0.
	const size_t NUM_LINE = 100;
	const size_t NUM_NOISE = 500;
	const double eps = 1.0e-10;

	// Generate random points.
	std::vector<local::Point2> samples;
	samples.reserve(NUM_LINE + NUM_NOISE);
	{
		for (size_t i = 0; i < NUM_LINE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3].
			const double y = -(LINE_EQN[2] + LINE_EQN[0] * x) / LINE_EQN[1];
			samples.push_back(local::Point2(x, y));
		}

		for (size_t i = 0; i < NUM_NOISE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			const double y = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			samples.push_back(local::Point2(x, y));
		}
	}

	const size_t minimalSampleSetSize = 2;
	local::Line2RansacEstimator ransac(samples, minimalSampleSetSize);

	const size_t maxIterationCount = 500;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = false;

	std::cout << "********* RANSAC of Line2" << std::endl;
	{
		const double threshold = 0.2;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		if (std::abs(ransac.getA()) > eps)
			std::cout << "\tEstimated line model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " = 0" << std::endl;
		else
			std::cout << "\tEstimated line model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " = 0" << std::endl;
		std::cout << "\tTrue line model:      " << "x + " << (LINE_EQN[1] / LINE_EQN[0]) << " * y + " << (LINE_EQN[2] / LINE_EQN[0]) << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		std::cout << "\tIndices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}

	std::cout << "********* MLESAC of Line2" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.15;
		const double outlierUniformProbability = 0.1;
		const size_t maxEMIterationCount = 10;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, outlierUniformProbability, maxEMIterationCount);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		if (std::abs(ransac.getA()) > eps)
			std::cout << "\tEstimated line model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " = 0" << std::endl;
		else
			std::cout << "\tEstimated line model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " = 0" << std::endl;
		std::cout << "\tTrue line model:      " << "x + " << (LINE_EQN[1] / LINE_EQN[0]) << " * y + " << (LINE_EQN[2] / LINE_EQN[0]) << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		std::cout << "\tIndices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}
}

void circle2_estimation()
{
	const double CIRCLE_EQN[4] = { 1, -2, 4, -4 };  // (x - 1)^2 + (y + 2)^2 = 3^2 <=> x^2 + y^2 - 2 * x + 4 * y - 4 = 0.
	const size_t NUM_CIRCLE = 100;
	const size_t NUM_NOISE = 500;
	const double eps = 1.0e-10;

	// Generate random points.
	std::vector<local::Point2> samples;
	samples.reserve(NUM_CIRCLE + NUM_NOISE);
	{
		const double b = CIRCLE_EQN[1] / CIRCLE_EQN[0], c = CIRCLE_EQN[2] / CIRCLE_EQN[0], d = CIRCLE_EQN[3] / CIRCLE_EQN[0];

		for (size_t i = 0; i < NUM_CIRCLE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3].
			const double y = (std::rand() % 2) ? (std::sqrt(c*c/4.0 - x*x - b*x - d) - c/2.0) : (-std::sqrt(c*c/4.0 - x*x - b*x - d) - c/2.0);
			samples.push_back(local::Point2(x, y));
		}

		for (size_t i = 0; i < NUM_NOISE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			const double y = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			samples.push_back(local::Point2(x, y));
		}
	}

	const size_t minimalSampleSetSize = 3;
	local::Circle2RansacEstimator ransac(samples, minimalSampleSetSize);

	const size_t maxIterationCount = 500;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = false;

	std::cout << "********* RANSAC of Circle2" << std::endl;
	{
		const double threshold = 0.2;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		if (std::abs(ransac.getA()) > eps)
			std::cout << "\tEstimated circle model: " << "x^2 + y^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
		else
			std::cout << "\tEstimated circle model: " << ransac.getA() << " * x^2 + " << ransac.getA() << " * y^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
		std::cout << "\tTrue circle model:      " << "x^2 + y^2 + " << (CIRCLE_EQN[1] / CIRCLE_EQN[0]) << " * x + " << (CIRCLE_EQN[2] / CIRCLE_EQN[0]) << " * y + " << (CIRCLE_EQN[3] / CIRCLE_EQN[0]) << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		std::cout << "\tIndices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}

	std::cout << "********* MLESAC of Circle2" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.15;
		const double outlierUniformProbability = 0.1;
		const size_t maxEMIterationCount = 10;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, outlierUniformProbability, maxEMIterationCount);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		if (std::abs(ransac.getA()) > eps)
			std::cout << "\tEstimated circle model: " << "x^2 + y^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
		else
			std::cout << "\tEstimated circle model: " << ransac.getA() << " * x^2 + " << ransac.getA() << " * y^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
		std::cout << "\tTrue circle model:      " << "x^2 + y^2 + " << (CIRCLE_EQN[1] / CIRCLE_EQN[0]) << " * x + " << (CIRCLE_EQN[2] / CIRCLE_EQN[0]) << " * y + " << (CIRCLE_EQN[3] / CIRCLE_EQN[0]) << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		std::cout << "\tIndices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}
}

void quadratic2_estimation()
{
	const double QUADRATIC_EQN[4] = { 1, -1, 1, -2 };  // x^2 - x + y - 2 = 0.
	const size_t NUM_QUADRATIC = 100;
	const size_t NUM_NOISE = 500;
	const double eps = 1.0e-10;

	// Generate random points.
	std::vector<local::Point2> samples;
	samples.reserve(NUM_QUADRATIC + NUM_NOISE);
	{
		for (size_t i = 0; i < NUM_QUADRATIC; ++i)
		{
			const double x = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3].
			const double y = -(QUADRATIC_EQN[0] * x * x + QUADRATIC_EQN[1] * x + QUADRATIC_EQN[3]) / QUADRATIC_EQN[2];
			samples.push_back(local::Point2(x, y));
		}

		for (size_t i = 0; i < NUM_NOISE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			const double y = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			samples.push_back(local::Point2(x, y));
		}
	}

	const size_t minimalSampleSetSize = 3;
	local::Quadratic2RansacEstimator ransac(samples, minimalSampleSetSize);

	const size_t maxIterationCount = 500;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = false;

	std::cout << "********* RANSAC of Quadratic2" << std::endl;
	{
		const double threshold = 0.2;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		if (std::abs(ransac.getA()) > eps)
			std::cout << "\tEstimated quadratic curve model: " << "x^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
		else
			std::cout << "\tEstimated quadratic curve model: " << ransac.getA() << " * x^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
		std::cout << "\tTrue quadratic curve model:      " << "x^2 + " << (QUADRATIC_EQN[1] / QUADRATIC_EQN[0]) << " * x + " << (QUADRATIC_EQN[2] / QUADRATIC_EQN[0]) << " * y + " << (QUADRATIC_EQN[3] / QUADRATIC_EQN[0]) << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		std::cout << "\tIndices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}

	std::cout << "********* MLESAC of Quadratic2" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.15;
		const double outlierUniformProbability = 0.1;
		const size_t maxEMIterationCount = 10;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, outlierUniformProbability, maxEMIterationCount);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		if (std::abs(ransac.getA()) > eps)
			std::cout << "\tEstimated quadratic curve model: " << "x^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
		else
			std::cout << "\tEstimated quadratic curve model: " << ransac.getA() << " * x^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
		std::cout << "\tTrue quadratic curve model:      " << "x^2 + " << (QUADRATIC_EQN[1] / QUADRATIC_EQN[0]) << " * x + " << (QUADRATIC_EQN[2] / QUADRATIC_EQN[0]) << " * y + " << (QUADRATIC_EQN[3] / QUADRATIC_EQN[0]) << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		std::cout << "\tIndices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}
}

void plane3_estimation()
{
	//const double PLANE_EQN[4] = { 0.5774, -0.5774, 0.5774, -1.1547 };  // x - y + z - 2 = 0.
	const double PLANE_EQN[4] = { 1, -1, 1, -2 };  // x - y + z - 2 = 0.
	const size_t NUM_PLANE = 30;
	const size_t NUM_NOISE = 100;
	const double eps = 1.0e-10;

	// Generate random points.
	std::vector<local::Point3> samples;
	samples.reserve(NUM_PLANE + NUM_NOISE);
	{
		for (size_t i = 0; i < NUM_PLANE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3].
			const double y = std::rand() % 10001 * 0.0006 - 3.0;  // [-3, 3].
			const double z = -(PLANE_EQN[0] * x + PLANE_EQN[1] * y + PLANE_EQN[3]) / PLANE_EQN[2];
			samples.push_back(local::Point3(x, y, z));
		}

		for (size_t i = 0; i < NUM_NOISE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			const double y = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			const double z = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			samples.push_back(local::Point3(x, y, z));
		}
	}

	const size_t minimalSampleSetSize = 3;
	local::Plane3RansacEstimator ransac(samples, minimalSampleSetSize);

	const size_t maxIterationCount = 500;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = false;

	std::cout << "********* RANSAC of Plane3" << std::endl;
	{
		const double threshold = 0.2;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		if (std::abs(ransac.getA()) > eps)
			std::cout << "\tEstimated plane model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " * z + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
		else 
			std::cout << "\tEstimated plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;
		std::cout << "\tTrue plane model:      " << "x + " << (PLANE_EQN[1] / PLANE_EQN[0]) << " * y + " << (PLANE_EQN[2] / PLANE_EQN[0]) << " * z + " << (PLANE_EQN[3] / PLANE_EQN[0]) << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		std::cout << "\tIndices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}

	std::cout << "********* MLESAC of Plane3" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.15;
		const double outlierUniformProbability = 0.1;
		const size_t maxEMIterationCount = 10;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, outlierUniformProbability, maxEMIterationCount);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		if (std::abs(ransac.getA()) > eps)
			std::cout << "\tEstimated plane model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " * z + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
		else
			std::cout << "\tEstimated plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;
		std::cout << "\tTrue plane model:      " << "x + " << (PLANE_EQN[1] / PLANE_EQN[0]) << " * y + " << (PLANE_EQN[2] / PLANE_EQN[0]) << " * z + " << (PLANE_EQN[3] / PLANE_EQN[0]) << " = 0" << std::endl;

		const std::vector<bool> &inliers = ransac.getInliers();
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		std::cout << "\tIndices of inliers: ";
		int k = 0;
		for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++k)
			if (*it) std::cout << k << ", ";
		std::cout << std::endl;
	}
}

}  // namespace my_ransac
