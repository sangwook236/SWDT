//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libkdtreepp {

void find_within_range_example();
bool kdtree_example();
void hayne_example();

}  // namespace my_libkdtreepp

int libkdtreepp_main(int argc, char *argv[])
{
	my_libkdtreepp::find_within_range_example();
	my_libkdtreepp::kdtree_example();
	my_libkdtreepp::hayne_example();

	return 0;
}
