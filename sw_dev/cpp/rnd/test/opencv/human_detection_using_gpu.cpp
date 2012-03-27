//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void human_detection_using_gpu_hog()
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

void human_detection_using_gpu()
{
	local::human_detection_using_gpu_hog();
}
