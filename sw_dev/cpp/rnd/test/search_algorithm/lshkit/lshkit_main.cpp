//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_lshkit {

void scan_example();
void fitdata_example();

}  // namespace my_lshkit

int lshkit_main(int argc, char *argv[])
{
    my_lshkit::scan_example();
    my_lshkit::fitdata_example();

	return 0;
}
