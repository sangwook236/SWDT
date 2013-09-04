//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_nyu_depth_toolbox_v2 {

void depth_filling_cross_bilateral_filter_example();

}  // namespace my_nyu_depth_toolbox_v2

/*
[ref] NYU Depth Dataset V2
	http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
*/

int nyu_depth_toolbox_v2_main(int argc, char *argv[])
{
	// depth-filling cross bilateral filter.
	my_nyu_depth_toolbox_v2::depth_filling_cross_bilateral_filter_example();

	return 0;
}
