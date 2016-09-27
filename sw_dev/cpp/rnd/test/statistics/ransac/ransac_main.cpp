//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_ransac {

void line2_estimation();
void circle2_estimation();
void quadratic2_estimation();
void plane3_estimation();

}  // namespace my_ransac

int ransac_main(int argc, char *argv[])
{
	//my_ransac::line2_estimation();
	my_ransac::circle2_estimation();
	my_ransac::quadratic2_estimation();
	//my_ransac::plane3_estimation();

	return 0;
}
