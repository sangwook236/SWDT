//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void brox_optical_flow()
{
	throw std::runtime_error("Not yet implemented");
}

void pyrlk_optical_flow()
{
	throw std::runtime_error("Not yet implemented");
}

void farneback_optical_flow()
{
	throw std::runtime_error("Not yet implemented");
}

void opticalflow_nvidia_api()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void optical_flow_using_gpu()
{
	local::brox_optical_flow();
	local::pyrlk_optical_flow();
	local::farneback_optical_flow();
	local::opticalflow_nvidia_api();
}

}  // namespace my_opencv
