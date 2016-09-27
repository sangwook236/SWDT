#define __USE_OPENCV 1
//#include "stdafx.h"
#include "Ransac.h"
#if defined(__USE_OPENCV)
#include <opencv2/opencv.hpp>
#endif
#include <gsl/gsl_poly.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <list>
#include <vector>
#include <limits>
#include <cmath>
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
	// TODO [improve] >> For example, estimate the least squares solution from inliers.
	return true;
}

size_t Circle2RansacEstimator::lookForInliers(std::vector<bool> &inliers, const double threshold) const
{
	const double cx = -0.5 * b_ / a_, cy = -0.5 * c_ / a_;
	const double radius = 0.25 * (b_*b_ + c_*c_) - d_;
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute distance from a point to a model.
		const double dist = std::abs(std::sqrt((it->x - cx)*(it->x - cx) + (it->y - cy)*(it->y - cy)) - radius);

		inliers[k] = dist < threshold;
		if (inliers[k]) ++inlierCount;
	}

	return inlierCount;
}

void Circle2RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double cx = -0.5 * b_ / a_, cy = -0.5 * c_ / a_;
	const double radius = 0.25 * (b_*b_ + c_*c_) - d_;
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
	// TODO [improve] >> For example, estimate the least squares solution from inliers.
	return true;
}

size_t Quadratic2RansacEstimator::lookForInliers(std::vector<bool> &inliers, const double threshold) const
{
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		const double eps = 1.0e-10;

		// Compute distance from a point to a model.
		const double x0 = it->x, y0 = it->y;
		const double c_2 = c_ * c_;
		assert(c_2 > eps);
		const double aa = 4.0*a_*a_ / c_2, bb = 6.0*a_*b_ / c_2, cc = 2.0*(b_*b_ / c_2 + 2.0*a_*(d_ + y0) / c_2 + 1.0), dd = 2.0*(b_*(d_ + y0) / c_2 - x0);
		assert(std::abs(aa) > eps);

		gsl_complex z[3];
		gsl_poly_complex_solve_cubic(bb / aa, cc / aa, dd / aa, &z[0], &z[1], &z[2]);

		double dist;
		bool exist = false;
		for (int i = 0; i < 3; ++i)
			if (std::abs(z[i].dat[1]) < eps)
			{
				const double xf = z[i].dat[0], yf = -(a_ * xf*xf + b_ * xf + d_) / c_;
				dist = std::sqrt((xf - x0)*(xf - x0) + (yf - y0)*(yf - y0));
				exist = true;
				break;
			}
		assert(exist);

		inliers[k] = dist < threshold;
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
		const double eps = 1.0e-10;

		// Compute distance from a point to a model.
		const double x0 = it->x, y0 = it->y;
		const double c_2 = c_ * c_;
		assert(c_2 > eps);
		const double aa = 4.0*a_*a_ / c_2, bb = 6.0*a_*b_ / c_2, cc = 2.0*(b_*b_ / c_2 + 2.0*a_*(d_ + y0) / c_2 + 1.0), dd = 2.0*(b_*(d_ + y0) / c_2 - x0);
		assert(std::abs(aa) > eps);

		gsl_complex z[3];
		gsl_poly_complex_solve_cubic(bb / aa, cc / aa, dd / aa, &z[0], &z[1], &z[2]);

		double dist2;
		bool exist = false;
		for (int i = 0; i < 3; ++i)
			if (std::abs(z[i].dat[1]) < eps)
			{
				const double xf = z[i].dat[0], yf = -(a_ * xf*xf + b_ * xf + d_) / c_;
				dist2 = (xf - x0)*(xf - x0) + (yf - y0)*(yf - y0);
				exist = true;
				break;
			}
		assert(exist);

		// Compute inliers' probabilities.
		inlierProbs[k] = factor * std::exp(-0.5 * dist2 / inlierSquaredStandardDeviation);
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

}  // namespace local
}  // unnamed namespace

namespace my_ransac {

void circle2_estimation()
{
	const double CIRCLE_EQN[4] = { 1, -2, 4, -4 };  // (x - 1)^2 + (y + 2)^2 = 3^2 <=> x^2 + y^2 - 2 * x + 4 * y - 4 = 0.
	const size_t NUM_CIRCLE = 300;
	const size_t NUM_NOISE = 500;
	const double eps = 1.0e-10;

	// Generate random points.
	std::vector<local::Point2> samples;
	samples.reserve(NUM_CIRCLE + NUM_NOISE);
	{
		const double b = CIRCLE_EQN[1] / CIRCLE_EQN[0], c = CIRCLE_EQN[2] / CIRCLE_EQN[0], d = CIRCLE_EQN[3] / CIRCLE_EQN[0];

		for (size_t i = 0; i < NUM_CIRCLE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0010 - 5.0;  // [-5, 5].
			const double y = (std::rand() % 2) ? (std::sqrt(0.25*c*c - x*x - b*x - d) - 0.5*c) : (-std::sqrt(0.25*c*c - x*x - b*x - d) - 0.5*c);
			samples.push_back(local::Point2(x, y));
		}

		for (size_t i = 0; i < NUM_NOISE; ++i)
		{
			const double x = std::rand() % 10001 * 0.0014 - 7.0;  // [-7, 7].
			const double y = std::rand() % 10001 * 0.0014 - 7.0;  // [-7, 7].
			samples.push_back(local::Point2(x, y));
		}
	}

	const size_t minimalSampleSetSize = 3;
	local::Circle2RansacEstimator ransac(samples, minimalSampleSetSize);

	const size_t maxIterationCount = 1000;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.8;
	const bool isProsacSampling = true;

	std::cout << "********* RANSAC of Circle2" << std::endl;
	{
		const double threshold = 0.01;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		//if (inlierCount != (size_t)-1)
		if (inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated circle model: " << "x^2 + y^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated circle model: " << ransac.getA() << " * x^2 + " << ransac.getA() << " * y^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue circle model:      " << "x^2 + y^2 + " << (CIRCLE_EQN[1] / CIRCLE_EQN[0]) << " * x + " << (CIRCLE_EQN[2] / CIRCLE_EQN[0]) << " * y + " << (CIRCLE_EQN[3] / CIRCLE_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inliers = ransac.getInliers();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++idx)
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
				for (std::vector<local::Point2>::const_iterator cit = samples.begin(); cit != samples.end(); ++cit)
					cv::circle(rgb, cv::Point((int)std::floor(cit->x * scale + sx + 0.5), (int)std::floor(cit->y * scale + sy + 0.5)), 2, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);
				idx = 0;
				for (std::vector<bool>::const_iterator cit = inliers.begin(); cit != inliers.end(); ++cit, ++idx)
					if (*cit)
						cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				const double b = ransac.getB() / ransac.getA(), c = ransac.getC() / ransac.getA(), d = ransac.getD() / ransac.getA();
				const double cxe = -0.5 * b, cye = -0.5 * c, re = std::sqrt(0.25*(b*b + c*c) - d);
				cv::circle(rgb, cv::Point((int)std::floor(cxe * scale + sx + 0.5), (int)std::floor(cye * scale + sy + 0.5)), (int)std::floor(re * scale + 0.5), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
				// Draw the true model.
				const double cxt = 1.0, cyt = -2.0, rt = 3.0;
				cv::circle(rgb, cv::Point((int)std::floor(cxt * scale + sx + 0.5), (int)std::floor(cyt * scale + sy + 0.5)), (int)std::floor(rt * scale + 0.5), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				cv::imshow("RANSAC - Circle estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tRANSAC failed" << std::endl;
	}

	std::cout << "********* MLESAC of Circle2" << std::endl;
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
				std::cout << "\tEstimated circle model: " << "x^2 + y^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated circle model: " << ransac.getA() << " * x^2 + " << ransac.getA() << " * y^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue circle model:      " << "x^2 + y^2 + " << (CIRCLE_EQN[1] / CIRCLE_EQN[0]) << " * x + " << (CIRCLE_EQN[2] / CIRCLE_EQN[0]) << " * y + " << (CIRCLE_EQN[3] / CIRCLE_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inliers = ransac.getInliers();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++idx)
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
				for (std::vector<local::Point2>::const_iterator cit = samples.begin(); cit != samples.end(); ++cit)
					cv::circle(rgb, cv::Point((int)std::floor(cit->x * scale + sx + 0.5), (int)std::floor(cit->y * scale + sy + 0.5)), 2, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);
				idx = 0;
				for (std::vector<bool>::const_iterator cit = inliers.begin(); cit != inliers.end(); ++cit, ++idx)
					if (*cit)
						cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				const double b = ransac.getB() / ransac.getA(), c = ransac.getC() / ransac.getA(), d = ransac.getD() / ransac.getA();
				const double cxe = -0.5 * b, cye = -0.5 * c, re = std::sqrt(0.25*(b*b + c*c) - d);
				cv::circle(rgb, cv::Point((int)std::floor(cxe * scale + sx + 0.5), (int)std::floor(cye * scale + sy + 0.5)), (int)std::floor(re * scale + 0.5), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
				// Draw the true model.
				const double cxt = 1.0, cyt = -2.0, rt = 3.0;
				cv::circle(rgb, cv::Point((int)std::floor(cxt * scale + sx + 0.5), (int)std::floor(cyt * scale + sy + 0.5)), (int)std::floor(rt * scale + 0.5), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				cv::imshow("MLESAC - Circle estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tMLESAC failed" << std::endl;
	}

#if defined(__USE_OPENCV)
	cv::waitKey(0);
	cv::destroyAllWindows();
#endif
}

void quadratic2_estimation()
{
	const double QUADRATIC_EQN[4] = { 1, -1, 1, -2 };  // x^2 - x + y - 2 = 0.
	const size_t NUM_QUADRATIC = 300;
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

	const size_t maxIterationCount = 1000;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.1;
	const bool isProsacSampling = true;

	std::cout << "********* RANSAC of Quadratic2" << std::endl;
	{
		const double threshold = 0.05;

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, threshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		//if (inlierCount != (size_t)-1)
		if (inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated quadratic curve model: " << "x^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated quadratic curve model: " << ransac.getA() << " * x^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue quadratic curve model:      " << "x^2 + " << (QUADRATIC_EQN[1] / QUADRATIC_EQN[0]) << " * x + " << (QUADRATIC_EQN[2] / QUADRATIC_EQN[0]) << " * y + " << (QUADRATIC_EQN[3] / QUADRATIC_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inliers = ransac.getInliers();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++idx)
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
				for (std::vector<local::Point2>::const_iterator cit = samples.begin(); cit != samples.end(); ++cit)
					cv::circle(rgb, cv::Point((int)std::floor(cit->x * scale + sx + 0.5), (int)std::floor(cit->y * scale + sy + 0.5)), 2, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);
				idx = 0;
				for (std::vector<bool>::const_iterator cit = inliers.begin(); cit != inliers.end(); ++cit, ++idx)
					if (*cit)
						cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				//const double b = ransac.getB() / ransac.getA(), c = ransac.getC() / ransac.getA(), d = ransac.getD() / ransac.getA();
				//const double cxe = -0.5 * b, cye = -0.5 * c, re = std::sqrt(0.25*(b*b + c*c) - d);
				//cv::circle(rgb, cv::Point((int)std::floor(cxe * scale + sx + 0.5), (int)std::floor(cye * scale + sy + 0.5)), (int)std::floor(re * scale + 0.5), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
				// Draw the true model.
				//const double cxt = 1.0, cyt = -2.0, rt = 3.0;
				//cv::circle(rgb, cv::Point((int)std::floor(cxt * scale + sx + 0.5), (int)std::floor(cyt * scale + sy + 0.5)), (int)std::floor(rt * scale + 0.5), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				cv::imshow("RANSAC - Quadratic curve estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tRANSAC failed" << std::endl;
	}

	std::cout << "********* MLESAC of Quadratic2" << std::endl;
	{
		const double inlierSquaredStandardDeviation = 0.1;
		const double outlierUniformProbability = 0.1;
		const size_t maxEMIterationCount = 50;

		const size_t inlierCount = ransac.runMLESAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, inlierSquaredStandardDeviation, outlierUniformProbability, maxEMIterationCount);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		//if (inlierCount != (size_t)-1)
		if (inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated quadratic curve model: " << "x^2 + " << (ransac.getB() / ransac.getA()) << " * x + " << (ransac.getC() / ransac.getA()) << " * y + " << (ransac.getD() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated quadratic curve model: " << ransac.getA() << " * x^2 + " << ransac.getB() << " * x + " << ransac.getC() << " * y + " << ransac.getD() << " = 0" << std::endl;
			std::cout << "\tTrue quadratic curve model:      " << "x^2 + " << (QUADRATIC_EQN[1] / QUADRATIC_EQN[0]) << " * x + " << (QUADRATIC_EQN[2] / QUADRATIC_EQN[0]) << " * y + " << (QUADRATIC_EQN[3] / QUADRATIC_EQN[0]) << " = 0" << std::endl;

			const std::vector<bool> &inliers = ransac.getInliers();
			std::cout << "\tIndices of inliers: ";
			size_t idx = 0;
			for (std::vector<bool>::const_iterator it = inliers.begin(); it != inliers.end(); ++it, ++idx)
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
				for (std::vector<local::Point2>::const_iterator cit = samples.begin(); cit != samples.end(); ++cit)
					cv::circle(rgb, cv::Point((int)std::floor(cit->x * scale + sx + 0.5), (int)std::floor(cit->y * scale + sy + 0.5)), 2, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);
				idx = 0;
				for (std::vector<bool>::const_iterator cit = inliers.begin(); cit != inliers.end(); ++cit, ++idx)
					if (*cit)
						cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				//const double b = ransac.getB() / ransac.getA(), c = ransac.getC() / ransac.getA(), d = ransac.getD() / ransac.getA();
				//const double cxe = -0.5 * b, cye = -0.5 * c, re = std::sqrt(0.25*(b*b + c*c) - d);
				//cv::circle(rgb, cv::Point((int)std::floor(cxe * scale + sx + 0.5), (int)std::floor(cye * scale + sy + 0.5)), (int)std::floor(re * scale + 0.5), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
				// Draw the true model.
				//const double cxt = 1.0, cyt = -2.0, rt = 3.0;
				//cv::circle(rgb, cv::Point((int)std::floor(cxt * scale + sx + 0.5), (int)std::floor(cyt * scale + sy + 0.5)), (int)std::floor(rt * scale + 0.5), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				cv::imshow("MLESAC - Quadratic curve estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tMLESAC failed" << std::endl;
	}

#if defined(__USE_OPENCV)
	cv::waitKey(0);
	cv::destroyAllWindows();
#endif
}

}  // namespace my_ransac
