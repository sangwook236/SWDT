//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

void inference_using_graphcut()
{
	throw std::runtime_error("not yet implemented");
}

void inference_using_belief_propagation()
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

void mrf()
{
	local::inference_using_graphcut();
	local::inference_using_belief_propagation();
}
