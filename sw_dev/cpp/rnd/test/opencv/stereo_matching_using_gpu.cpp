//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void bm()
{
	throw std::runtime_error("not yet implemented");
}

void belief_propagation()
{
	throw std::runtime_error("not yet implemented");
}

void constant_space_belief_propagation()
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

void stereo_matching_using_gpu()
{
	local::bm();
	local::belief_propagation();
	local::constant_space_belief_propagation();
}
