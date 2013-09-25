//#include "stdafx.h"
#include <ompl/config.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_ompl {

void rigid_body_planning_example();

}  // namespace my_ompl

int ompl_main(int argc, char *argv[])
{
	std::cout << "OMPL version: " << OMPL_VERSION << std::endl;

	my_ompl::rigid_body_planning_example();

	return 0;
}

