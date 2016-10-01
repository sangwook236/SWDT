//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void fab_map_sample()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// REF [file] >> ${OPENCV_HOME}/sample/cpp/fabmap_sample.cpp
void slam()
{
	local::fab_map_sample();
}

}  // namespace my_opencv
