#define __USE_OPENCV 1
#include "../gransac_lib/AbstractModel.hpp"
#include "../gransac_lib/GRANSAC.hpp"
#if defined(__USE_OPENCV)
#include <opencv2/opencv.hpp>
#endif
#include <gsl/gsl_poly.h>
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <cassert>


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

class Quadratic2DModel : public GRANSAC::AbstractModel<3>
{
public:
	Quadratic2DModel(std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> inputParams)
	{
		Initialize(inputParams);
	}

	/*virtual*/ void Initialize(std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> inputParams) override
	{
		if (inputParams.size() != 3)
			throw std::runtime_error("Quadratic2DModel - Number of input parameters does not match minimum number required for this model.");

		// Check for AbstractParamter types.
		const auto point1 = std::dynamic_pointer_cast<Point2D>(inputParams[0]);
		const auto point2 = std::dynamic_pointer_cast<Point2D>(inputParams[1]);
		const auto point3 = std::dynamic_pointer_cast<Point2D>(inputParams[2]);
		if (nullptr == point1 || nullptr == point2 || nullptr == point3)
			throw std::runtime_error("Quadratic2DModel - InputParams type mismatch. It is not a Point2D.");

		std::copy(inputParams.begin(), inputParams.end(), m_MinModelParams.begin());

		// Compute the quadratic curve parameters.
		const GRANSAC::VPFloat x1 = point1->m_Point2D[0], y1 = point1->m_Point2D[1], x1_2 = x1 * x1;
		const GRANSAC::VPFloat x2 = point2->m_Point2D[0], y2 = point2->m_Point2D[1], x2_2 = x2 * x2;
		const GRANSAC::VPFloat x3 = point3->m_Point2D[0], y3 = point3->m_Point2D[1], x3_2 = x3 * x3;

		a_ = x1*(y3 - y2) - x2*y3 + x3*y2 + (x2 - x3)*y1;
		b_ = x1_2*(y3 - y2) - x2_2*y3 + x3_2*y2 + (x2_2 - x3_2)*y1;
		c_ = x1*(x3_2 - x2_2) - x2*x3_2 + x2_2*x3 + x1_2*(x2 - x3);
		d_ = x1*(x3_2*y2 - x2_2*y3) + x1_2*(x2*y3 - x3*y2) + (x2_2*x3 - x2*x3_2)*y1;

		aa_ = -a_ / c_;
		bb_ = -b_ / c_;
		cc_ = -d_ / c_;
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
			throw std::runtime_error("Quadratic2DModel::ComputeDistanceMeasure() - Passed parameter are not of type Point2D.");

		const double eps = 1.0e-10;
		const double x0 = extPoint2D->m_Point2D[0], y0 = extPoint2D->m_Point2D[1];

		// Return distance between passed "point" and this quadratic curve.
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

protected:
	// Parametric form: a * x^2 + b * x + c * y + d = 0.
	GRANSAC::VPFloat a_, b_, c_, d_;

	// Another parametrization: y = aa * x^2 + bb * x + cc.
	GRANSAC::VPFloat aa_, bb_, cc_;
};

}  // namespace local
}  // unnamed namespace

namespace my_gransac {

void quadratic2_estimation()
{
	const int RANGE = 5;  // Image size.
	const int NUM_POINTS = 500;

	// Randomly generate points in a 2D plane roughly aligned in a line for testing.
	std::random_device seedDevice;
	std::mt19937 RNG = std::mt19937(seedDevice());

	std::uniform_real_distribution<double> uniDist(-RANGE, RANGE);  // [Incl, Incl].
	const int perturb = 25;
	std::normal_distribution<GRANSAC::VPFloat> perturbDist(0, perturb);

	std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> dataPoints;
	for (int i = 0; i < NUM_POINTS; ++i)
	{
		// Quadratic curve equation: x^2 - x + y - 2 = 0.
		const GRANSAC::VPFloat x = uniDist(RNG);
		const GRANSAC::VPFloat y = -x*x + x + 2;

		dataPoints.push_back(std::make_shared<local::Point2D>(std::floor(x + perturbDist(RNG)), std::floor(y + perturbDist(RNG))));
	}

	// RANSAC.
	GRANSAC::RANSAC<local::Quadratic2DModel, 3> estimator;
	estimator.Initialize(1.0, 1000);  // Threshold, iterations.
	estimator.Estimate(dataPoints);

	// Output.
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
		const auto bestLinePt3 = std::dynamic_pointer_cast<local::Point2D>(bestLine->GetModelParams()[2]);
		if (bestLinePt1 && bestLinePt2 && bestLinePt3)
		{
			const GRANSAC::VPFloat x1 = bestLinePt1->m_Point2D[0], y1 = bestLinePt1->m_Point2D[1], x1_2 = x1 * x1;
			const GRANSAC::VPFloat x2 = bestLinePt2->m_Point2D[0], y2 = bestLinePt2->m_Point2D[1], x2_2 = x2 * x2;
			const GRANSAC::VPFloat x3 = bestLinePt3->m_Point2D[0], y3 = bestLinePt3->m_Point2D[1], x3_2 = x3 * x3;

			const GRANSAC::VPFloat a = x1*(y3 - y2) - x2*y3 + x3*y2 + (x2 - x3)*y1;
			const GRANSAC::VPFloat b = x1_2*(y3 - y2) - x2_2*y3 + x3_2*y2 + (x2_2 - x3_2)*y1;
			const GRANSAC::VPFloat c = x1*(x3_2 - x2_2) - x2*x3_2 + x2_2*x3 + x1_2*(x2 - x3);
			const GRANSAC::VPFloat d = x1*(x3_2*y2 - x2_2*y3) + x1_2*(x2*y3 - x3*y2) + (x2_2*x3 - x2*x3_2)*y1;

			std::cout << "Estimated quadratic curve model: " << a << " * x^2 + " << b << " * x + " << c << " * y + " << d << " = 0" << std::endl;
			std::cout << "True quadratic curve model:      " << 1 << " * x^2 + " << -1 << " * x + " << 1 << " * y + " << -2 << " = 0" << std::endl;
		}
	}
}

}  // namespace my_gransac
