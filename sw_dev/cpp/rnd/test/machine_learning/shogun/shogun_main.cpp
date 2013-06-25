//#include "stdafx.h"
#include <shogun/base/init.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

void classification_example();
void regression_example();

}  // namespace my_shogun

int shogun_main(int argc, char *argv[])
{
	shogun::init_shogun_with_defaults();

	my_shogun::classification_example();
	my_shogun::regression_example();

	shogun::exit_shogun();

	return 0;
}
