//include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_ivt {

void hough_line_example();
void hough_circle_example();
void klt_tracker_example();
void particle_filter_example();

}  // namespace my_ivt

int ivt_main(int argc, char *argv[])
{
	// Hough transform.
	//my_ivt::hough_line_example();
	//my_ivt::hough_circle_example();

	// KLT tracker.
	//my_ivt::klt_tracker_example();

	// particle filtering.
	my_ivt::particle_filter_example();

	return 0;
}
