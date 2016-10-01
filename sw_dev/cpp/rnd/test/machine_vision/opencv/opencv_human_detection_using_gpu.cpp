//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void human_detection_using_gpu_hog()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void human_detection_using_gpu()
{
	local::human_detection_using_gpu_hog();
}

}  // namespace my_opencv
