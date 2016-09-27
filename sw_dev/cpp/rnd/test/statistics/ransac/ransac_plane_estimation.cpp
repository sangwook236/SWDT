//#include "stdafx.h"
#include "Ransac.h"
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
	// TODO [improve] >> For example, estimate the least squares solution from inliers.
	return true;
}

size_t Plane3RansacEstimator::lookForInliers(std::vector<bool> &inliers, const double threshold) const
{
	const double denom = std::sqrt(a_*a_ + b_*b_ + c_*c_);
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point3>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute distance from a point to a model.
		const double dist = std::abs(a_ * it->x + b_ * it->y + c_ * it->z + d_) / denom;

		inliers[k] = dist < threshold;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
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
	const bool isProsacSampling = true;

	std::cout << "********* RANSAC of Plane3" << std::endl;
	{
		const double threshold = 0.05;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		//if (inlierCount != (size_t)-1)
		if (inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated plane model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " * z + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else 
				std::cout << "\tEstimated plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue plane model:      " << "x + " << (PLANE_EQN[1] / PLANE_EQN[0]) << " * y + " << (PLANE_EQN[2] / PLANE_EQN[0]) << " * z + " << (PLANE_EQN[3] / PLANE_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inliers = ransac.getInliers();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++idx)
				if (*it) std::cout << idx << ", ";
			std::cout << std::endl;
		}
		else
			std::cout << "\tRANSAC failed" << std::endl;
	}

	std::cout << "********* MLESAC of Plane3" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.01;
		const double outlierUniformProbability = 0.1;
		const size_t maxEMIterationCount = 50;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, outlierUniformProbability, maxEMIterationCount);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		//if (inlierCount != (size_t)-1)
		if (inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated plane model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " * z + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated plane model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " * z + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue plane model:      " << "x + " << (PLANE_EQN[1] / PLANE_EQN[0]) << " * y + " << (PLANE_EQN[2] / PLANE_EQN[0]) << " * z + " << (PLANE_EQN[3] / PLANE_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inliers = ransac.getInliers();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++idx)
				if (*it) std::cout << idx << ", ";
			std::cout << std::endl;
		}
		else
			std::cout << "\tMLESAC failed" << std::endl;
	}
}

}  // namespace my_ransac
