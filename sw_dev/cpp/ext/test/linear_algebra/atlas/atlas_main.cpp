//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace atlas {

void cblas();
void clapack();

}  // namespace atlas

int atlas_main(int argc, char* argv[])
{
	atlas::cblas();
	atlas::clapack();

	return 0;
}
