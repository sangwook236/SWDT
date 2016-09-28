//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_ransac {

void line2_estimation();
void circle2_estimation();
void plane3_estimation();

void quadratic2_estimation();

}  // namespace my_ransac

int ransac_main(int argc, char *argv[])
{
	// ----------------------------------------------------
	//my_ransac::line2_estimation();
	//my_ransac::circle2_estimation();
	//my_ransac::plane3_estimation();

	// ----------------------------------------------------
	// Verify an estimated model based on anchor points: REF [function] >> Quadratic2RansacEstimator::verifyModel().
	// Refine an estimated model using inliers: REF [function] >> Quadratic2RansacEstimator::estimateModelFromInliers().
	my_ransac::quadratic2_estimation();

	return 0;
}
