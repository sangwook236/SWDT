//#include "stdafx.h"
#include "Ransac.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <list>
#include <limits>
#include <cmath>
#include <random>


#if defined(max)
#undef max
#endif


namespace {
namespace local {

const double PI = 4.0 * std::atan(1.0);

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

class Plane3RansacEstimator: public Ransac
{
public:
	typedef Ransac base_type;

public:
	Plane3RansacEstimator(const std::vector<Point3> &samples, const size_t minimalSampleSize, const size_t usedSampleSize = 0, const std::shared_ptr<std::vector<double>> &scores = nullptr)
	: base_type(samples.size(), minimalSampleSize, usedSampleSize, scores), samples_(samples)
	{}

public:
	double getA() const  {  return a_;  }
	double getB() const  {  return b_;  }
	double getC() const  {  return c_;  }
	double getD() const  {  return d_;  }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices) override;
	/*virtual*/ bool verifyModel() const override;
	/*virtual*/ bool estimateModelFromInliers() override;

	// For RANSAC.
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inlierFlags, const double threshold) const override;
	// For MLESAC.
	/*virtual*/ void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const override;
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inlierFlags, const std::vector<double> &inlierProbs, const double inlierThresholdProbability) const override;

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
	if (indices.size() < minimalSampleSize_) return false;

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
	// TODO [improve] >> Check the validity of the estimated model.
	return true;
}

bool Plane3RansacEstimator::estimateModelFromInliers()
{
	// TODO [improve] >> For example, estimate the least squares solution from inliers.
	return true;
}

size_t Plane3RansacEstimator::lookForInliers(std::vector<bool> &inlierFlags, const double threshold) const
{
	const double denom = std::sqrt(a_*a_ + b_*b_ + c_*c_);
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute distance from a point to a model.
		const double dist = std::abs(a_ * it->x + b_ * it->y + c_ * it->z + d_) / denom;

		inlierFlags[k] = dist < threshold;
		if (inlierFlags[k]) ++inlierCount;
	}

	return inlierCount;
	//return std::count(inlierFlags.begin(), inlierFlags.end(), true);
}

void Plane3RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double denom = std::sqrt(a_*a_ + b_*b_ + c_*c_);
	const double factor = 1.0 / std::sqrt(2.0 * PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute distance from a point to a model.
		const double dist = (a_ * it->x + b_ * it->y + c_ * it->z + d_) / denom;

		// Compute inliers' probabilities.
		inlierProbs[k] = factor * std::exp(-0.5 * dist * dist / inlierSquaredStandardDeviation);
	}
}

size_t Plane3RansacEstimator::lookForInliers(std::vector<bool> &inlierFlags, const std::vector<double> &inlierProbs, const double inlierThresholdProbability) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inlierFlags[k] = inlierProbs[k] >= inlierThresholdProbability;
		if (inlierFlags[k]) ++inlierCount;
	}

	return inlierCount;
}

}  // namespace local
}  // unnamed namespace

namespace my_ransac {

void plane3_estimation()
{
	//const double PLANE_EQN[4] = { 0.5774, -0.5774, 0.5774, -1.1547 };  // x - y + z - 2 = 0.
	const double PLANE_EQN[4] = { 1, -1, 1, -2 };  // x - y + z - 2 = 0.
	const size_t NUM_PLANE = 100;
	const size_t NUM_NOISE = 500;
	const double eps = 1.0e-10;

	// Generate random points.
	std::vector<local::Point3> samples;
	samples.reserve(NUM_PLANE + NUM_NOISE);
	{
		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> unifDistInlier(-3, 3);  // [-3, 3].
		const double sigma = 0.1;
		//const double sigma = 0.2;  // Much harder.
		std::normal_distribution<double> noiseDist(0.0, sigma);
		for (size_t i = 0; i < NUM_PLANE; ++i)
		{
			const double x = unifDistInlier(RNG), y = unifDistInlier(RNG), z = -(PLANE_EQN[0] * x + PLANE_EQN[1] * y + PLANE_EQN[3]) / PLANE_EQN[2];
			samples.push_back(local::Point3(x + noiseDist(RNG), y + noiseDist(RNG), z + noiseDist(RNG)));
		}

		std::uniform_real_distribution<double> unifDistOutlier(-5, 5);  // [-5, 5].
		for (size_t i = 0; i < NUM_NOISE; ++i)
			samples.push_back(local::Point3(unifDistOutlier(RNG), unifDistOutlier(RNG), unifDistOutlier(RNG)));

		std::random_shuffle(samples.begin(), samples.end());
	}

	// RANSAC.
	const size_t minimalSampleSize = 3;
	local::Plane3RansacEstimator ransac(samples, minimalSampleSize);

	const size_t maxIterationCount = 500;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = true;

	std::cout << "********* RANSAC of Plane3" << std::endl;
	{
		const double distanceThreshold = 0.1;  // Distance threshold.

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, distanceThreshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		if (inlierCount != (size_t)-1 && inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated plane model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " * z + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else 
				std::cout << "\tEstimated plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue plane model:      " << "x + " << (PLANE_EQN[1] / PLANE_EQN[0]) << " * y + " << (PLANE_EQN[2] / PLANE_EQN[0]) << " * z + " << (PLANE_EQN[3] / PLANE_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inlierFlags = ransac.getInlierFlags();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inlierFlags.begin(); it != inlierFlags.end(); ++it, ++idx)
				if (*it) std::cout << idx << ", ";
			std::cout << std::endl;
		}
		else
			std::cout << "\tRANSAC failed" << std::endl;
	}

	std::cout << "********* MLESAC of Plane3" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.001;  // Inliers' squared standard deviation. Assume that inliers follow normal distribution.
		const double inlierThresholdProbability = 0.1;  // Inliers' threshold probability. Assume that outliers follow uniform distribution.
		const size_t maxEMIterationCount = 50;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, inlierThresholdProbability, maxEMIterationCount);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		if (inlierCount != (size_t)-1 && inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated plane model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " * z + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue plane model:      " << "x + " << (PLANE_EQN[1] / PLANE_EQN[0]) << " * y + " << (PLANE_EQN[2] / PLANE_EQN[0]) << " * z + " << (PLANE_EQN[3] / PLANE_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inlierFlags = ransac.getInlierFlags();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inlierFlags.begin(); it != inlierFlags.end(); ++it, ++idx)
				if (*it) std::cout << idx << ", ";
			std::cout << std::endl;
		}
		else
			std::cout << "\tMLESAC failed" << std::endl;
	}
}

}  // namespace my_ransac
