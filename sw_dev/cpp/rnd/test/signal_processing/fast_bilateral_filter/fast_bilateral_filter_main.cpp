//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_fast_bilateral_filter {

void fast_bilateral_filter_example();
void truncated_kernel_bilateral_filter_example();
void color_bilateral_filter_example();
void cross_bilateral_filter_example();

void depth_filling_cross_bilateral_filter();

}  // namespace my_fast_bilateral_filter

/*
[ref]
	"A Fast Approximation of the Bilateral Filter using a Signal Processing Approach", S. Paris and F. Durand, ECCV, 2006.
	http://people.csail.mit.edu/sparis/bf/
*/

int fast_bilateral_filter_main(int argc, char *argv[])
{
	// fast bilateral filter.
	//my_fast_bilateral_filter::fast_bilateral_filter_example();

	// truncated kernel bilateral filter.
	//my_fast_bilateral_filter::truncated_kernel_bilateral_filter_example();

	// color bilateral filter.
	//my_fast_bilateral_filter::color_bilateral_filter_example();

	// joint/cross bilateral filter.
	// FIXME [fix] >> not correctly working.
	//	-. parameters have to be adjusted.
	//		filtering is sensitive to parameters.
	//	-. edge image may be not proper.
	my_fast_bilateral_filter::cross_bilateral_filter_example();

	//------------------------------------------------------------------
	// extension

	// depth-filling joint/cross bilateral filter.
	// FIXME [fix] >> not correctly working.
	//	-. additional implementation is required.
	//		depth impainting logic.
	//my_fast_bilateral_filter::depth_filling_cross_bilateral_filter();

	return 0;
}
