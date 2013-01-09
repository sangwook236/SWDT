//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_atlas {

void cblas();
void clapack();

}  // namespace my_atlas

int atlas_main(int argc, char* argv[])
{
	my_atlas::cblas();
	my_atlas::clapack();

	return 0;
}
