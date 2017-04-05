//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_osg {

void basic_geometry();

}  // namespace my_osg

int osg_main(int argc, char *argv[])
{
	// ---------------------------------------------------------------
	my_osg::basic_geometry();

	// Integration with Qt -------------------------------------------
	// REF [project] >> ${SWDT_C++_HOME}/ext/test/gui.

	return 0;
}

