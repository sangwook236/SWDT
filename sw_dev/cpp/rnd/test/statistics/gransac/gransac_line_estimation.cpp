#define __USE_OPENCV 1
#include "../gransac_lib/AbstractModel.hpp"
#include "../gransac_lib/GRANSAC.hpp"
#if defined(__USE_OPENCV)
#include <opencv2/opencv.hpp>
#endif
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>


namespace {
namespace local {

typedef std::array<GRANSAC::VPFloat, 2> Vector2VP;

// REF [site] >> https://github.com/srinath1905/GRANSAC/blob/master/examples/LineModel.hpp
class Point2D : public GRANSAC::AbstractParameter
{
public:
	Point2D(GRANSAC::VPFloat x, GRANSAC::VPFloat y)
	{
		m_Point2D[0] = x;
		m_Point2D[1] = y;
	}

	Vector2VP m_Point2D;
};

// REF [site] >> https://github.com/srinath1905/GRANSAC/blob/master/examples/LineModel.hpp
class Line2DModel : public GRANSAC::AbstractModel<2>
{
public:
	Line2DModel(std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> inputParams)
	{
		Initialize(inputParams);
	}

	/*virtual*/ void Initialize(std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> inputParams) override
	{
		if (inputParams.size() != 2)
			throw std::runtime_error("Line2DModel - Number of input parameters does not match minimum number required for this model.");

		// Check for AbstractParamter types.
		const auto point1 = std::dynamic_pointer_cast<Point2D>(inputParams[0]);
		const auto point2 = std::dynamic_pointer_cast<Point2D>(inputParams[1]);
		if (nullptr == point1 || nullptr == point2)
			throw std::runtime_error("Line2DModel - InputParams type mismatch. It is not a Point2D.");

		std::copy(inputParams.begin(), inputParams.end(), m_MinModelParams.begin());

		// Compute the line parameters.
		m_m = (point2->m_Point2D[1] - point1->m_Point2D[1]) / (point2->m_Point2D[0] - point1->m_Point2D[0]);  // Slope.
		m_d = point1->m_Point2D[1] - m_m * point1->m_Point2D[0];  // Intercept.
		//m_d = point2->m_Point2D[1] - m_m * point2->m_Point2D[0];  // Intercept - alternative should be the same as above.

		// mx - y + d = 0.
		m_a = m_m;
		m_b = -1.0;
		m_c = m_d;

		m_DistDenominator = std::sqrt(m_a * m_a + m_b * m_b);  // Cache square root for efficiency.
	}

	/*virtual*/ std::pair<GRANSAC::VPFloat, std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>> Evaluate(std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> evaluateParams, GRANSAC::VPFloat threshold)	override
	{
		std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> inliers;
		const int nTotalParams = evaluateParams.size();
		int nInliers = 0;

		for (const auto& param : evaluateParams)
		{
			if (ComputeDistanceMeasure(param) < threshold)
			{
				inliers.push_back(param);
				++nInliers;
			}
		}

		const GRANSAC::VPFloat inlierFraction = GRANSAC::VPFloat(nInliers) / GRANSAC::VPFloat(nTotalParams);  // This is the inlier fraction.

		return std::make_pair(inlierFraction, inliers);
	}

protected:
	/*virtual*/ GRANSAC::VPFloat ComputeDistanceMeasure(std::shared_ptr<GRANSAC::AbstractParameter> param) override
	{
		const auto extPoint2D = std::dynamic_pointer_cast<Point2D>(param);
		if (nullptr == extPoint2D)
			throw std::runtime_error("Line2DModel::ComputeDistanceMeasure() - Passed parameter are not of type Point2D.");

		// Return distance between passed "point" and this line.
		// http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
		const GRANSAC::VPFloat numer = std::fabs(m_a * extPoint2D->m_Point2D[0] + m_b * extPoint2D->m_Point2D[1] + m_c);
		const GRANSAC::VPFloat dist = numer / m_DistDenominator;

		// // Debug.
		// std::cout << "Point: " << extPoint2D->m_Point2D[0] << ", " << extPoint2D->m_Point2D[1] << std::endl;
		// std::cout << "Line: " << m_a << " x + " << m_b << " y + "  << m_c << std::endl;
		// std::cout << "Distance: " << dist << std::endl << std::endl;

		return dist;
	}

protected:
	// Parametric form.
	GRANSAC::VPFloat m_a, m_b, m_c;  // ax + by + c = 0.
	GRANSAC::VPFloat m_DistDenominator;  // = sqrt(a^2 + b^2). Stored for efficiency reasons.

	// Another parametrization y = mx + d.
	GRANSAC::VPFloat m_m;  // Slope.
	GRANSAC::VPFloat m_d;  // Intercept.
};

// REF [site] >> https://github.com/srinath1905/GRANSAC/blob/master/examples/LineFittingSample.cpp
GRANSAC::VPFloat computeSlope(int x0, int y0, int x1, int y1)
{
	return x0 == x1 ? std::numeric_limits<GRANSAC::VPFloat>::max() : (GRANSAC::VPFloat)(y1 - y0) / (x1 - x0);
}

#if defined(__USE_OPENCV)
// REF [site] >> https://github.com/srinath1905/GRANSAC/blob/master/examples/LineFittingSample.cpp
void drawFullLine(cv::Mat& img, cv::Point a, cv::Point b, cv::Scalar color, int lineWidth)
{
	const GRANSAC::VPFloat slope = computeSlope(a.x, a.y, b.x, b.y);

	cv::Point p(0, 0), q(img.cols, img.rows);

	p.y = -(a.x - p.x) * slope + a.y;
	q.y = -(b.x - q.x) * slope + b.y;

	cv::line(img, p, q, color, lineWidth, cv::LINE_8, 0);
}
#endif

}  // namespace local
}  // unnamed namespace

namespace my_gransac {

// REF [site] >> https://github.com/srinath1905/GRANSAC/blob/master/examples/LineFittingSample.cpp
void line_estimation()
{
	const int IMAGE_SIZE = 1000;  // Image size.
	const int NUM_POINTS = 500;

#if defined(__USE_OPENCV)
	cv::Mat canvas(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3);
	canvas.setTo(cv::Scalar::all(255));
#endif

	// Randomly generate points in a 2D plane roughly aligned in a line for testing.
	std::random_device seedDevice;
	std::mt19937 RNG = std::mt19937(seedDevice());

	std::uniform_int_distribution<int> uniDist(0, IMAGE_SIZE - 1);  // [Incl, Incl].
	const int perturb = 25;
	std::normal_distribution<GRANSAC::VPFloat> perturbDist(0, perturb);

	std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> dataPoints;
	for (int i = 0; i < NUM_POINTS; ++i)
	{
		const int diag = uniDist(RNG);  // Diagonal line.
		const local::Point2D pt(std::floor(diag + perturbDist(RNG)), std::floor(diag + perturbDist(RNG)));
#if defined(__USE_OPENCV)
		cv::circle(canvas, cv::Point(pt.m_Point2D[0], pt.m_Point2D[1]), 2, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);
#endif

		dataPoints.push_back(std::make_shared<local::Point2D>(pt.m_Point2D[0], pt.m_Point2D[1]));
	}

	GRANSAC::RANSAC<local::Line2DModel, 2> estimator;
	estimator.Initialize(20, 100);  // Threshold, iterations.
	{
#if defined(__USE_OPENCV)
		const int start = cv::getTickCount();
		estimator.Estimate(dataPoints);
		const int end = cv::getTickCount();
		std::cout << "GRANSAC took: " << GRANSAC::VPFloat(end - start) / GRANSAC::VPFloat(cv::getTickFrequency()) * 1000.0 << " ms." << std::endl;
#else
		estimator.Estimate(dataPoints);
#endif
	}

#if defined(__USE_OPENCV)
	const auto bestInliers = estimator.GetBestInliers();
	if (bestInliers.size() > 0)
	{
		for (const auto& inlier : bestInliers)
		{
			const auto pt = std::dynamic_pointer_cast<local::Point2D>(inlier);
			cv::circle(canvas, cv::Point(std::floor(pt->m_Point2D[0]), std::floor(pt->m_Point2D[1])), 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
		}
	}

	const auto bestLine = estimator.GetBestModel();
	if (bestLine)
	{
		const auto bestLinePt1 = std::dynamic_pointer_cast<local::Point2D>(bestLine->GetModelParams()[0]);
		const auto bestLinePt2 = std::dynamic_pointer_cast<local::Point2D>(bestLine->GetModelParams()[1]);
		if (bestLinePt1 && bestLinePt2)
		{
			const cv::Point pt1(bestLinePt1->m_Point2D[0], bestLinePt1->m_Point2D[1]);
			const cv::Point pt2(bestLinePt2->m_Point2D[0], bestLinePt2->m_Point2D[1]);
			local::drawFullLine(canvas, pt1, pt2, cv::Scalar(0, 0, 255), 1);
		}
	}

	cv::imshow("GRANSAC Example", canvas);
	cv::waitKey(0);
		
	//cv::imwrite("LineFitting.png", canvas);
#else
	const auto bestInliers = estimator.GetBestInliers();
	if (bestInliers.size() > 0)
	{
		std::cout << "#Inliers = " << bestInliers.size() << std::endl;
		std::cout << "Inliers = ";
		for (const auto& inlier : bestInliers)
		{
			const auto pt = std::dynamic_pointer_cast<local::Point2D>(inlier);
			const GRANSAC::VPFloat x = std::floor(pt->m_Point2D[0]), y = std::floor(pt->m_Point2D[1]);
			std::cout << "(" << x << "," << y << "), ";
		}
		std::cout << std::endl;
	}

	const auto bestLine = estimator.GetBestModel();
	if (bestLine)
	{
		const auto bestLinePt1 = std::dynamic_pointer_cast<local::Point2D>(bestLine->GetModelParams()[0]);
		const auto bestLinePt2 = std::dynamic_pointer_cast<local::Point2D>(bestLine->GetModelParams()[1]);
		if (bestLinePt1 && bestLinePt2)
		{
			const GRANSAC::VPFloat x1 = bestLinePt1->m_Point2D[0], y1 = bestLinePt1->m_Point2D[1];
			const GRANSAC::VPFloat x2 = bestLinePt2->m_Point2D[0], y2 = bestLinePt2->m_Point2D[1];

			const GRANSAC::VPFloat m = (y2 - y1) / (x2 - x1);  // Slope.
			const GRANSAC::VPFloat b = y1 - m * x1;  // Intercept.
			//const GRANSAC::VPFloat b = y2 - m * x2;  // Intercept - alternative should be the same as above.

			std::cout << "Estimated line model: " << m << " * x + y + " << b << std::endl;
		}
	}
#endif
}

}  // namespace my_gransac
