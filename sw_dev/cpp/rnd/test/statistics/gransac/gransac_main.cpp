//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gransac {

void line_estimation();
void quadratic_estimation();

}  // namespace my_gransac

int gransac_main(int argc, char *argv[])
{
	//my_gransac::line_estimation();
	my_gransac::quadratic_estimation();

	return 0;
}
