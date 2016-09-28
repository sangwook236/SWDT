//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gransac {

void line2_estimation();
void quadratic2_estimation();

}  // namespace my_gransac

int gransac_main(int argc, char *argv[])
{
	//my_gransac::line2_estimation();
	my_gransac::quadratic2_estimation();

	return 0;
}
