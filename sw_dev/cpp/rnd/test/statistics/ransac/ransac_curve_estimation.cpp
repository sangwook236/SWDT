#define __USE_OPENCV 1
//#include "stdafx.h"
#include "Ransac.h"
#if defined(__USE_OPENCV)
#include <opencv2/opencv.hpp>
#endif
//#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <gsl/gsl_poly.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <list>
#include <limits>
#include <cmath>
#include <random>
#include <cassert>


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

class Circle2RansacEstimator : public Ransac
{
public:
	typedef Ransac base_type;

public:
	Circle2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSize, const size_t usedSampleSize = 0, const std::shared_ptr<std::vector<double>> &scores = nullptr)
	: base_type(samples.size(), minimalSampleSize, usedSampleSize, scores), samples_(samples)
	{}

public:
	double getA() const { return a_; }
	double getB() const { return b_; }
	double getC() const { return c_; }
	double getD() const { return d_; }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices) override;
	/*virtual*/ bool verifyModel() const override;
	/*virtual*/ bool estimateModelFromInliers() override;

	// For RANSAC.
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inlierFlags, const double threshold) const override;
	// For MLESAC.
	/*virtual*/ void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const override;
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inlierFlags, const std::vector<double> &inlierProbs, const double inlierThresholdProbability) const override;

private:
	const std::vector<Point2> &samples_;

	// Circle equation: a * x^2 + a * y^2 + b * x + c * y + d = 0.
	double a_, b_, c_, d_;
};

bool Circle2RansacEstimator::estimateModel(const std::vector<size_t> &indices)
{
	if (indices.size() < minimalSampleSize_) return false;

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
	// TODO [improve] >> Check the validity of the estimated model.
	return true;
}

bool Circle2RansacEstimator::estimateModelFromInliers()
{
	// TODO [improve] >> For example, estimate the least squares solution from inliers.
	return true;
}

size_t Circle2RansacEstimator::lookForInliers(std::vector<bool> &inlierFlags, const double threshold) const
{
	const double cx = -0.5 * b_ / a_, cy = -0.5 * c_ / a_;
	const double radius = std::sqrt(0.25 * (b_*b_ + c_*c_) / (a_*a_) - d_ / a_);
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute distance from a point to a model.
		const double dist = std::abs(std::sqrt((it->x - cx)*(it->x - cx) + (it->y - cy)*(it->y - cy)) - radius);

		inlierFlags[k] = dist < threshold;
		if (inlierFlags[k]) ++inlierCount;
	}

	return inlierCount;
	//return std::count(inlierFlags.begin(), inlierFlags.end(), true);
}

void Circle2RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double cx = -0.5 * b_ / a_, cy = -0.5 * c_ / a_;
	const double radius = std::sqrt(0.25 * (b_*b_ + c_*c_) / (a_*a_) - d_ / a_);
	const double factor = 1.0 / std::sqrt(2.0 * PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute distance from a point to a model.
		const double dist = std::sqrt((it->x - cx)*(it->x - cx) + (it->y - cy)*(it->y - cy)) - radius;

		// Compute inliers' probabilities.
		inlierProbs[k] = factor * std::exp(-0.5 * dist * dist / inlierSquaredStandardDeviation);
	}
}

size_t Circle2RansacEstimator::lookForInliers(std::vector<bool> &inlierFlags, const std::vector<double> &inlierProbs, const double inlierThresholdProbability) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inlierFlags[k] = inlierProbs[k] >= inlierThresholdProbability;
		if (inlierFlags[k]) ++inlierCount;
	}

	return inlierCount;
}

class Quadratic2RansacEstimator : public Ransac
{
public:
	typedef Ransac base_type;

public:
	Quadratic2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSize, const size_t usedSampleSize = 0, const std::shared_ptr<std::vector<double>> &scores = nullptr, const std::shared_ptr<std::vector<Point2>> &anchorPoints = nullptr)
	: base_type(samples.size(), minimalSampleSize, usedSampleSize, scores), samples_(samples), anchorPoints_(anchorPoints)
	{}

public:
	double getA() const { return a_; }
	double getB() const { return b_; }
	double getC() const { return c_; }
	double getD() const { return d_; }

private:
	/*virtual*/ bool estimateModel(const std::vector<size_t> &indices) override;
	/*virtual*/ bool verifyModel() const override;
	/*virtual*/ bool estimateModelFromInliers() override;

	// For RANSAC.
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inlierFlags, const double threshold) const override;
	// For MLESAC.
	/*virtual*/ void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const override;
	/*virtual*/ size_t lookForInliers(std::vector<bool> &inlierFlags, const std::vector<double> &inlierProbs, const double inlierThresholdProbability) const override;

	double computeSquaredMinDistanceFromModel(const double x0, const double y0) const;
	bool estimateModelUsingLeastSqaures(const size_t sampleSize, const std::vector<size_t> &indices);

private:
	const std::vector<Point2> &samples_;
	const std::shared_ptr<std::vector<Point2>> anchorPoints_;

	// Quadratic curve equation: a * x^2 + b * x + c * y + d = 0.
	double a_, b_, c_, d_;
};

bool Quadratic2RansacEstimator::estimateModel(const std::vector<size_t> &indices)
{
	const size_t sampleSize = indices.size();
	if (sampleSize < minimalSampleSize_) return false;
	else if (minimalSampleSize_ == sampleSize)
	{
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
	else  // When #sample >= 4.
		return estimateModelUsingLeastSqaures(sampleSize, indices);
}

bool Quadratic2RansacEstimator::verifyModel() const
{
	if (!anchorPoints_) return true;

	const double distanceThreshold = 0.3;
	for (std::vector<local::Point2>::const_iterator cit = anchorPoints_->begin(); cit != anchorPoints_->end(); ++cit)
	{
		// 1. Use distance threshold;
		if (computeSquaredMinDistanceFromModel(cit->x, cit->y) > distanceThreshold) return false;

		// 2. Neighbors (k-nearest neighbors or epsilon-ball neighbors) of the anchor points are probably inliers.
	}

	return true;
}

bool Quadratic2RansacEstimator::estimateModelFromInliers()
{
	const size_t inlierCount = std::count(inlierFlags_.begin(), inlierFlags_.end(), true);
	if (inlierCount < minimalSampleSize_) return false;

	std::vector<size_t> indices;
	indices.reserve(totalSampleSize_);

	// TODO [imporve] >> Not-so-good implementation.
	size_t k = 0;
	for (const auto& flag : inlierFlags_)
	{
		if (flag) indices.push_back(k);
		++k;
	}
	
	return estimateModelUsingLeastSqaures(inlierCount, indices);
}

size_t Quadratic2RansacEstimator::lookForInliers(std::vector<bool> &inlierFlags, const double threshold) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inlierFlags[k] = std::sqrt(computeSquaredMinDistanceFromModel(it->x, it->y)) < threshold;
		if (inlierFlags[k]) ++inlierCount;
	}

	return inlierCount;
	//return std::count(inlierFlags.begin(), inlierFlags.end(), true);
}

void Quadratic2RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double factor = 1.0 / std::sqrt(2.0 * PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute inliers' probabilities.
		inlierProbs[k] = factor * std::exp(-0.5 * computeSquaredMinDistanceFromModel(it->x, it->y) / inlierSquaredStandardDeviation);
	}
}

size_t Quadratic2RansacEstimator::lookForInliers(std::vector<bool> &inlierFlags, const std::vector<double> &inlierProbs, const double inlierThresholdProbability) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		inlierFlags[k] = inlierProbs[k] >= inlierThresholdProbability;
		if (inlierFlags[k]) ++inlierCount;
	}

	return inlierCount;
}

double Quadratic2RansacEstimator::computeSquaredMinDistanceFromModel(const double x0, const double y0) const
{
	const double eps = 1.0e-10;

	// Compute distance from a point to a model.
	const double c_2 = c_ * c_;
	assert(c_2 > eps);
	const double aa = 4.0*a_*a_ / c_2, bb = 6.0*a_*b_ / c_2, cc = 2.0*(b_*b_ / c_2 + 2.0*a_*(d_ + c_ * y0) / c_2 + 1.0), dd = 2.0*(b_*(d_ + c_ * y0) / c_2 - x0);
	assert(aa > eps);

	double minDist2 = std::numeric_limits<double>::max();
	double roots[3] = { 0.0, };
	switch (gsl_poly_solve_cubic(bb / aa, cc / aa, dd / aa, &roots[0], &roots[1], &roots[2]))
	{
	case 1:
		{
			const double xx = roots[0], yy = -(a_ * xx*xx + b_ * xx + d_) / c_;
			minDist2 = (xx - x0)*(xx - x0) + (yy - y0)*(yy - y0);
		}
		break;
	case 3:
		for (int i = 0; i < 3; ++i)
		{
			const double xx = roots[i], yy = -(a_ * xx*xx + b_ * xx + d_) / c_;
			const double dist2 = (xx - x0)*(xx - x0) + (yy - y0)*(yy - y0);
			if (dist2 < minDist2)
				minDist2 = dist2;
		}
		break;
	default:
		assert(false);
		break;
	}

	return minDist2;
}

bool Quadratic2RansacEstimator::estimateModelUsingLeastSqaures(const size_t sampleSize, const std::vector<size_t> &indices)
{
#if 0
	// Using SVD for linear least squares.

	const size_t dim = 4;
	Eigen::MatrixXd m(sampleSize, dim);
	{
		size_t k = 0;
		for (const auto& idx : indices)
		{
			const Point2& pt = samples_[idx];
			m(k, 0) = pt.x * pt.x; m(k, 1) = pt.x; m(k, 2) = pt.y; m(k, 3) = 1.0;
			++k;
		}
	}

	// MxN matrix, K=min(M,N), M>=N.
	//const Eigen::SVD<Eigen::MatrixXd> svd(m);
	const Eigen::JacobiSVD<Eigen::MatrixXd> svd = m.jacobiSvd(Eigen::ComputeThinV);
	// Right singular vectors: KxN matrix.
	const Eigen::JacobiSVD<Eigen::MatrixXd>::MatrixVType& V = svd.matrixV();
	assert(dim == V.rows());

	const size_t last = V.rows() - 1;
	a_ = V(last, 0);
	b_ = V(last, 1);
	c_ = V(last, 2);
	d_ = V(last, 3);
#elif 1
	// Using SVD for linear least squares.

	const size_t dim = 3;
	Eigen::MatrixXd m(sampleSize, dim);
	Eigen::VectorXd b(sampleSize);
	{
		size_t k = 0;
		for (const auto& idx : indices)
		{
			const Point2& pt = samples_[idx];
			m(k, 0) = pt.x * pt.x; m(k, 1) = pt.x; m(k, 2) = 1.0;
			b(k) = pt.y;
			++k;
		}
	}

	const Eigen::VectorXd& sol = m.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
	assert(dim == sol.size());

	a_ = sol(0);
	b_ = sol(1);
	c_ = -1.0;
	d_ = sol(2);
#else
	// Using traditional linear least squares.

	const size_t dim = 3;
	Eigen::MatrixXd m(sampleSize, dim);
	Eigen::VectorXd b(sampleSize);
	{
		size_t k = 0;
		for (const auto& idx : indices)
		{
			const Point2& pt = samples_[idx];
			m(k, 0) = pt.x * pt.x; m(k, 1) = pt.x; m(k, 2) = 1.0;
			b(k) = pt.y;
			++k;
		}
	}

	//const Eigen::VectorXd& sol = (m.transpose() * m).inverse() * m.transpose() * b;  // Slow.
	const Eigen::VectorXd& sol = (m.transpose() * m).ldlt().solve(m.transpose() * b);
	assert(dim == sol.size());

	a_ = sol(0);
	b_ = sol(1);
	c_ = -1.0;
	d_ = sol(2);
#endif

	return true;
}

#if defined(__USE_OPENCV)
// Quadratic equation: y = a * x^2 + b * x + c.
void drawQuadratic(cv::Mat& rgb, const double a, const double b, const double c, const double xmin, const double xmax, const double sx, const double sy, const double scale, const cv::Scalar& color, const int thickness)
{
	const double inc = (xmax - xmin) * 0.01;
	double x0 = xmin;
	do
	{
		const double x1 = x0 + inc;
		const double y0 = a * x0*x0 + b * x0 + c, y1 = a * x1*x1 + b * x1 + c;
		cv::line(rgb, cv::Point((int)std::floor(x0 * scale + sx + 0.5), rgb.cols - (int)std::floor(y0 * scale + sy + 0.5)), cv::Point((int)std::floor(x1 * scale + sx + 0.5), rgb.cols - (int)std::floor(y1 * scale + sy + 0.5)), color, thickness, cv::LINE_AA);

		x0 = x1;
	} while (x0 <= xmax);
}
#endif

}  // namespace local
}  // unnamed namespace

namespace my_ransac {

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

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> unifDistInlier(-2, 4);  // [-2, 4].
		const double sigma = 0.1;
		//const double sigma = 0.2;  // Much harder.
		std::normal_distribution<double> noiseDist(0.0, sigma);
		for (size_t i = 0; i < NUM_CIRCLE; ++i)
		{
			const double x = unifDistInlier(RNG);
			const double y = (std::rand() % 2) ? (std::sqrt(0.25*c*c - x*x - b*x - d) - 0.5*c) : (-std::sqrt(0.25*c*c - x*x - b*x - d) - 0.5*c);
			samples.push_back(local::Point2(x + noiseDist(RNG), y + noiseDist(RNG)));
		}

		std::uniform_real_distribution<double> unifDistOutlier(-6, 6);  // [-6, 6].
		//std::uniform_real_distribution<double> unifDistOutlier1(-4, 6);  // [-4, 6].
		//std::uniform_real_distribution<double> unifDistOutlier2(-7, 3);  // [-7, 3].
		for (size_t i = 0; i < NUM_NOISE; ++i)
		{
			samples.push_back(local::Point2(unifDistOutlier(RNG), unifDistOutlier(RNG)));
			//samples.push_back(local::Point2(unifDistOutlier1(RNG), unifDistOutlier2(RNG)));
		}

		std::random_shuffle(samples.begin(), samples.end());
	}

	// RANSAC.
	const size_t minimalSampleSize = 3;
	local::Circle2RansacEstimator ransac(samples, minimalSampleSize);

	const size_t maxIterationCount = 1000;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = true;

	std::cout << "********* RANSAC of Circle2" << std::endl;
	{
		const double distanceThreshold = 0.1;  // Distance threshold.

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, distanceThreshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		if (inlierCount != (size_t)-1 && inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated circle model: " << "x^2 + y^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated circle model: " << ransac.getA() << " * x^2 + " << ransac.getA() << " * y^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue circle model:      " << "x^2 + y^2 + " << (CIRCLE_EQN[1] / CIRCLE_EQN[0]) << " * x + " << (CIRCLE_EQN[2] / CIRCLE_EQN[0]) << " * y + " << (CIRCLE_EQN[3] / CIRCLE_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inlierFlags = ransac.getInlierFlags();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inlierFlags.begin(); it != inlierFlags.end(); ++it, ++idx)
				if (*it) std::cout << idx << ", ";
			std::cout << std::endl;

#if defined(__USE_OPENCV)
			// For visualization.
			{
				// Draw samples and inliners.
				const int IMG_SIZE = 600;
				const double sx = 300.0, sy = 300.0, scale = 50.0;
				cv::Mat rgb(IMG_SIZE, IMG_SIZE, CV_8UC3);
				rgb.setTo(cv::Scalar::all(255));
				size_t idx = 0;
				for (std::vector<bool>::const_iterator cit = inlierFlags.begin(); cit != inlierFlags.end(); ++cit, ++idx)
					cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), IMG_SIZE - (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, *cit ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				const double b = ransac.getB() / ransac.getA(), c = ransac.getC() / ransac.getA(), d = ransac.getD() / ransac.getA();
				const double cxe = -0.5 * b, cye = -0.5 * c, re = std::sqrt(0.25*(b*b + c*c) - d);
				cv::circle(rgb, cv::Point((int)std::floor(cxe * scale + sx + 0.5), IMG_SIZE - (int)std::floor(cye * scale + sy + 0.5)), (int)std::floor(re * scale + 0.5), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
				// Draw the true model.
				const double cxt = 1.0, cyt = -2.0, rt = 3.0;
				cv::circle(rgb, cv::Point((int)std::floor(cxt * scale + sx + 0.5), IMG_SIZE - (int)std::floor(cyt * scale + sy + 0.5)),  (int)std::floor(rt * scale + 0.5), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				cv::imshow("RANSAC - Circle estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tRANSAC failed" << std::endl;
	}

	std::cout << "********* MLESAC of Circle2" << std::endl;
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
				std::cout << "\tEstimated circle model: " << "x^2 + y^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated circle model: " << ransac.getA() << " * x^2 + " << ransac.getA() << " * y^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue circle model:      " << "x^2 + y^2 + " << (CIRCLE_EQN[1] / CIRCLE_EQN[0]) << " * x + " << (CIRCLE_EQN[2] / CIRCLE_EQN[0]) << " * y + " << (CIRCLE_EQN[3] / CIRCLE_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inlierFlags = ransac.getInlierFlags();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inlierFlags.begin(); it != inlierFlags.end(); ++it, ++idx)
				if (*it) std::cout << idx << ", ";
			std::cout << std::endl;

#if defined(__USE_OPENCV)
			// For visualization.
			{
				// Draw samples and inliners.
				const int IMG_SIZE = 600;
				const double sx = 300.0, sy = 300.0, scale = 50.0;
				cv::Mat rgb(IMG_SIZE, IMG_SIZE, CV_8UC3);
				rgb.setTo(cv::Scalar::all(255));
				size_t idx = 0;
				for (std::vector<bool>::const_iterator cit = inlierFlags.begin(); cit != inlierFlags.end(); ++cit, ++idx)
					cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), IMG_SIZE - (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, *cit ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				const double b = ransac.getB() / ransac.getA(), c = ransac.getC() / ransac.getA(), d = ransac.getD() / ransac.getA();
				const double cxe = -0.5 * b, cye = -0.5 * c, re = std::sqrt(0.25*(b*b + c*c) - d);
				cv::circle(rgb, cv::Point((int)std::floor(cxe * scale + sx + 0.5), IMG_SIZE - (int)std::floor(cye * scale + sy + 0.5)), (int)std::floor(re * scale + 0.5), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
				// Draw the true model.
				const double cxt = 1.0, cyt = -2.0, rt = 3.0;
				cv::circle(rgb, cv::Point((int)std::floor(cxt * scale + sx + 0.5), IMG_SIZE - (int)std::floor(cyt * scale + sy + 0.5)), (int)std::floor(rt * scale + 0.5), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				cv::imshow("MLESAC - Circle estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tMLESAC failed" << std::endl;
	}

#if defined(__USE_OPENCV)
	std::cout << "Press any key to continue ..." << std::endl;
	cv::waitKey(0);
	cv::destroyAllWindows();
#endif
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
		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> unifDistInlier(-4, 4);  // [-4, 4].
		const double sigma = 0.1;
		//const double sigma = 0.2;  // Much harder.
		std::normal_distribution<double> noiseDist(0.0, sigma);
		for (size_t i = 0; i < NUM_QUADRATIC; ++i)
		{
			const double x = unifDistInlier(RNG);
			const double y = -(QUADRATIC_EQN[0] * x * x + QUADRATIC_EQN[1] * x + QUADRATIC_EQN[3]) / QUADRATIC_EQN[2];
			samples.push_back(local::Point2(x + noiseDist(RNG), y + noiseDist(RNG)));
		}

		std::uniform_real_distribution<double> unifDistOutlier1(-4, 4);  // [-4, 4].
		std::uniform_real_distribution<double> unifDistOutlier2(-20, 5);  // [-20, 5].
		for (size_t i = 0; i < NUM_NOISE; ++i)
			samples.push_back(local::Point2(unifDistOutlier1(RNG), unifDistOutlier2(RNG)));

		std::random_shuffle(samples.begin(), samples.end());
	}

	// Anchor points.
	std::shared_ptr<std::vector<local::Point2>> anchorPoints(new std::vector<local::Point2>);
	{
		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		const double sigma = 0.1;
		//const double sigma = 0.2;  // Much harder.
		std::normal_distribution<double> noiseDist(0.0, sigma);
		const double x1 = -3.0;
		const double y1 = -(QUADRATIC_EQN[0] * x1 * x1 + QUADRATIC_EQN[1] * x1 + QUADRATIC_EQN[3]) / QUADRATIC_EQN[2];
		anchorPoints->push_back(local::Point2(x1 + noiseDist(RNG), y1 + noiseDist(RNG)));
		const double x2 = 1.0;
		const double y2 = -(QUADRATIC_EQN[0] * x2 * x2 + QUADRATIC_EQN[1] * x2 + QUADRATIC_EQN[3]) / QUADRATIC_EQN[2];
		anchorPoints->push_back(local::Point2(x2 + noiseDist(RNG), y2 + noiseDist(RNG)));
	}

	// RANSAC.
	const size_t minimalSampleSize = 3;
	const size_t usedSampleSize = 10;
	local::Quadratic2RansacEstimator ransac(samples, minimalSampleSize, usedSampleSize, nullptr, anchorPoints);

	const size_t maxIterationCount = 1000;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = true;

	std::cout << "********* RANSAC of Quadratic2" << std::endl;
	{
		const double distanceThreshold = 0.1;  // Distance threshold.

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, distanceThreshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		if (inlierCount != (size_t)-1 && inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated quadratic curve model: " << "x^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated quadratic curve model: " << ransac.getA() << " * x^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue quadratic curve model:      " << "x^2 + " << (QUADRATIC_EQN[1] / QUADRATIC_EQN[0]) << " * x + " << (QUADRATIC_EQN[2] / QUADRATIC_EQN[0]) << " * y + " << (QUADRATIC_EQN[3] / QUADRATIC_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inlierFlags = ransac.getInlierFlags();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inlierFlags.begin(); it != inlierFlags.end(); ++it, ++idx)
				if (*it) std::cout << idx << ", ";
			std::cout << std::endl;

#if defined(__USE_OPENCV)
			// For visualization.
			{
				// Draw samples and inliners.
				const int IMG_SIZE = 800;
				const double sx = 400.0, sy = 600.0, scale = 20.0;
				cv::Mat rgb(IMG_SIZE, IMG_SIZE, CV_8UC3);
				rgb.setTo(cv::Scalar::all(255));
				size_t idx = 0;
				for (std::vector<bool>::const_iterator cit = inlierFlags.begin(); cit != inlierFlags.end(); ++cit, ++idx)
					cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), IMG_SIZE - (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, *cit ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				local::drawQuadratic(rgb, -ransac.getA() / ransac.getC(), -ransac.getB() / ransac.getC(), -ransac.getD() / ransac.getC(), -5.0, 5.0, sx, sy, scale, cv::Scalar(255, 0, 0), 1);
				// Draw the true model.
				local::drawQuadratic(rgb, -QUADRATIC_EQN[0] / QUADRATIC_EQN[2], -QUADRATIC_EQN[1] / QUADRATIC_EQN[2], -QUADRATIC_EQN[3] / QUADRATIC_EQN[2], -5.0, 5.0, sx, sy, scale, cv::Scalar(0, 0, 255), 1);

				cv::imshow("RANSAC - Quadratic curve estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tRANSAC failed" << std::endl;
	}

	std::cout << "********* MLESAC of Quadratic2" << std::endl;
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
				std::cout << "\tEstimated quadratic curve model: " << "x^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated quadratic curve model: " << ransac.getA() << " * x^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue quadratic curve model:      " << "x^2 + " << (QUADRATIC_EQN[1] / QUADRATIC_EQN[0]) << " * x + " << (QUADRATIC_EQN[2] / QUADRATIC_EQN[0]) << " * y + " << (QUADRATIC_EQN[3] / QUADRATIC_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inlierFlags = ransac.getInlierFlags();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inlierFlags.begin(); it != inlierFlags.end(); ++it, ++idx)
				if (*it) std::cout << idx << ", ";
			std::cout << std::endl;

#if defined(__USE_OPENCV)
			// For visualization.
			{
				// Draw samples and inliners.
				const int IMG_SIZE = 800;
				const double sx = 400.0, sy = 600.0, scale = 20.0;
				cv::Mat rgb(IMG_SIZE, IMG_SIZE, CV_8UC3);
				rgb.setTo(cv::Scalar::all(255));
				size_t idx = 0;
				for (std::vector<bool>::const_iterator cit = inlierFlags.begin(); cit != inlierFlags.end(); ++cit, ++idx)
					cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), IMG_SIZE - (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, *cit ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				local::drawQuadratic(rgb, -ransac.getA() / ransac.getC(), -ransac.getB() / ransac.getC(), -ransac.getD() / ransac.getC(), -5.0, 5.0, sx, sy, scale, cv::Scalar(255, 0, 0), 1);
				// Draw the true model.
				local::drawQuadratic(rgb, -QUADRATIC_EQN[0] / QUADRATIC_EQN[2], -QUADRATIC_EQN[1] / QUADRATIC_EQN[2], -QUADRATIC_EQN[3] / QUADRATIC_EQN[2], -5.0, 5.0, sx, sy, scale, cv::Scalar(0, 0, 255), 1);

				cv::imshow("MLESAC - Quadratic curve estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tMLESAC failed" << std::endl;
	}

#if defined(__USE_OPENCV)
	std::cout << "Press any key to continue ..." << std::endl;
	cv::waitKey(0);
	cv::destroyAllWindows();
#endif
}

}  // namespace my_ransac
