#define __USE_OPENCV 1
//#include "stdafx.h"
#include "Ransac.h"
#if defined(__USE_OPENCV)
#include <opencv2/opencv.hpp>
#endif
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

class Line2RansacEstimator : public Ransac
{
public:
	typedef Ransac base_type;

public:
	Line2RansacEstimator(const std::vector<Point2> &samples, const size_t minimalSampleSize, const size_t usedSampleSize = 0, const std::shared_ptr<std::vector<double>> &scores = nullptr)
	: base_type(samples.size(), minimalSampleSize, usedSampleSize, scores), samples_(samples)
	{}

public:
	double getA() const { return a_; }
	double getB() const { return b_; }
	double getC() const { return c_; }

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

	// Line equation: a * x + b * y + c = 0.
	double a_, b_, c_;
};

bool Line2RansacEstimator::estimateModel(const std::vector<size_t> &indices)
{
	if (indices.size() < minimalSampleSize_) return false;

	const Point2 &pt1 = samples_[indices[0]];
	const Point2 &pt2 = samples_[indices[1]];

	a_ = pt2.y - pt1.y;
	b_ = pt1.x - pt2.x;
	c_ = -a_ * pt1.x - b_ * pt1.y;

	return true;
}

bool Line2RansacEstimator::verifyModel() const
{
	// TODO [improve] >> Check the validity of the estimated model.
	return true;
}

bool Line2RansacEstimator::estimateModelFromInliers()
{
	// TODO [improve] >> For example, estimate the least squares solution from inliers.
	return true;
}

size_t Line2RansacEstimator::lookForInliers(std::vector<bool> &inlierFlags, const double threshold) const
{
	const double denom = std::sqrt(a_*a_ + b_*b_);
	size_t inlierCount = 0;
	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute distance from a model to a point.
		const double dist = std::abs(a_ * it->x + b_ * it->y + c_) / denom;

		inlierFlags[k] = dist < threshold;
		if (inlierFlags[k]) ++inlierCount;
	}

	return inlierCount;
	//return std::count(inlierFlags.begin(), inlierFlags.end(), true);
}

void Line2RansacEstimator::computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const
{
	const double denom = std::sqrt(a_*a_ + b_*b_);
	const double factor = 1.0 / std::sqrt(2.0 * PI * inlierSquaredStandardDeviation);

	int k = 0;
	for (std::vector<Point2>::const_iterator it = samples_.begin(); it != samples_.end(); ++it, ++k)
	{
		// Compute distance from a point to a model.
		const double dist = (a_ * it->x + b_ * it->y + c_) / denom;

		// Compute inliers' probabilities.
		inlierProbs[k] = factor * std::exp(-0.5 * dist * dist / inlierSquaredStandardDeviation);
	}
}

size_t Line2RansacEstimator::lookForInliers(std::vector<bool> &inlierFlags, const std::vector<double> &inlierProbs, const double inlierThresholdProbability) const
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
		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> unifDistInlier(-3, 3);  // [-3, 3].
		const double sigma = 0.1;
		//const double sigma = 0.2;  // Much harder.
		std::normal_distribution<double> noiseDist(0.0, sigma);
		for (size_t i = 0; i < NUM_LINE; ++i)
		{
			const double x = unifDistInlier(RNG), y = -(LINE_EQN[0] * x + LINE_EQN[2]) / LINE_EQN[1];
			samples.push_back(local::Point2(x + noiseDist(RNG), y + noiseDist(RNG)));
		}

		std::uniform_real_distribution<double> unifDistOutlier(-5, 5);  // [-5, 5].
		for (size_t i = 0; i < NUM_NOISE; ++i)
			samples.push_back(local::Point2(unifDistOutlier(RNG), unifDistOutlier(RNG)));

		std::random_shuffle(samples.begin(), samples.end());
	}

	// RANSAC.
	const size_t minimalSampleSize = 2;
	local::Line2RansacEstimator ransac(samples, minimalSampleSize);

	const size_t maxIterationCount = 500;
	const size_t minInlierCount = 50;
	const double alarmRatio = 0.5;
	const bool isProsacSampling = true;

	std::cout << "********* RANSAC of Line2" << std::endl;
	{
		const double distanceThreshold = 0.1;  // Distance threshold.

		const size_t inlierCount = ransac.runRANSAC(maxIterationCount, minInlierCount, alarmRatio, isProsacSampling, distanceThreshold);

		std::cout << "\tThe number of iterations: " << ransac.getIterationCount() << std::endl;
		std::cout << "\tThe number of inliers: " << inlierCount << std::endl;
		if (inlierCount != (size_t)-1 && inlierCount >= minInlierCount)
		{
			if (std::abs(ransac.getA()) > eps)
				std::cout << "\tEstimated line model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated line model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " = 0" << std::endl;
			std::cout << "\tTrue line model:      " << "x + " << (LINE_EQN[1] / LINE_EQN[0]) << " * y + " << (LINE_EQN[2] / LINE_EQN[0]) << " = 0" << std::endl;

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
				const double sx = 300.0, sy = 300.0, scale = 100.0;
				cv::Mat rgb(IMG_SIZE, IMG_SIZE, CV_8UC3);
				rgb.setTo(cv::Scalar::all(255));
				size_t idx = 0;
				for (std::vector<bool>::const_iterator cit = inlierFlags.begin(); cit != inlierFlags.end(); ++cit, ++idx)
					cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), IMG_SIZE - (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, *cit ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				const double xe0 = -3.0, ye0 = -(ransac.getA() * xe0 + ransac.getC()) / ransac.getB();
				const double xe1 = 3.0, ye1 = -(ransac.getA() * xe1 + ransac.getC()) / ransac.getB();
				cv::line(rgb, cv::Point((int)std::floor(xe0 * scale + sx + 0.5), IMG_SIZE - (int)std::floor(ye0 * scale + sy + 0.5)), cv::Point((int)std::floor(xe1 * scale + sx + 0.5), IMG_SIZE - (int)std::floor(ye1 * scale + sy + 0.5)), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
				// Draw the true model.
				const double xt0 = -3.0, yt0 = -(LINE_EQN[0] * xt0 + LINE_EQN[2]) / LINE_EQN[1];
				const double xt1 = 3.0, yt1 = -(LINE_EQN[0] * xt1 + LINE_EQN[2]) / LINE_EQN[1];
				cv::line(rgb, cv::Point((int)std::floor(xt0 * scale + sx + 0.5), IMG_SIZE - (int)std::floor(yt0 * scale + sy + 0.5)), cv::Point((int)std::floor(xt1 * scale + sx + 0.5), IMG_SIZE - (int)std::floor(yt1 * scale + sy + 0.5)), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				cv::imshow("RANSAC - Line estimation", rgb);
			}
#endif
		}
		else
			std::cout << "\tRANSAC failed" << std::endl;
	}

	std::cout << "********* MLESAC of Line2" << std::endl;
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
				std::cout << "\tEstimated line model: " << "x + " << (ransac.getB() / ransac.getA()) << " * y + " << (ransac.getC() / ransac.getA()) << " = 0" << std::endl;
			else
				std::cout << "\tEstimated line model: " << ransac.getA() << " * x + " << ransac.getB() << " * y + " << ransac.getC() << " = 0" << std::endl;
			std::cout << "\tTrue line model:      " << "x + " << (LINE_EQN[1] / LINE_EQN[0]) << " * y + " << (LINE_EQN[2] / LINE_EQN[0]) << " = 0" << std::endl;

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
				const double sx = 300.0, sy = 300.0, scale = 100.0;
				cv::Mat rgb(IMG_SIZE, IMG_SIZE, CV_8UC3);
				rgb.setTo(cv::Scalar::all(255));
				size_t idx = 0;
				for (std::vector<bool>::const_iterator cit = inlierFlags.begin(); cit != inlierFlags.end(); ++cit, ++idx)
					cv::circle(rgb, cv::Point((int)std::floor(samples[idx].x * scale + sx + 0.5), IMG_SIZE - (int)std::floor(samples[idx].y * scale + sy + 0.5)), 2, *cit ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);

				// Draw the estimated model.
				const double xe0 = -3.0, ye0 = -(ransac.getA() * xe0 + ransac.getC()) / ransac.getB();
				const double xe1 = 3.0, ye1 = -(ransac.getA() * xe1 + ransac.getC()) / ransac.getB();
				cv::line(rgb, cv::Point((int)std::floor(xe0 * scale + sx + 0.5), IMG_SIZE - (int)std::floor(ye0 * scale + sy + 0.5)), cv::Point((int)std::floor(xe1 * scale + sx + 0.5), IMG_SIZE - (int)std::floor(ye1 * scale + sy + 0.5)), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
				// Draw the true model.
				const double xt0 = -3.0, yt0 = -(LINE_EQN[0] * xt0 + LINE_EQN[2]) / LINE_EQN[1];
				const double xt1 = 3.0, yt1 = -(LINE_EQN[0] * xt1 + LINE_EQN[2]) / LINE_EQN[1];
				cv::line(rgb, cv::Point((int)std::floor(xt0 * scale + sx + 0.5), IMG_SIZE - (int)std::floor(yt0 * scale + sy + 0.5)), cv::Point((int)std::floor(xt1 * scale + sx + 0.5), IMG_SIZE - (int)std::floor(yt1 * scale + sy + 0.5)), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				cv::imshow("MLESAC - Line estimation", rgb);
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
