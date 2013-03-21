//#include "stdafx.h"
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opennn {

void simple_function_regression_example();
void simple_pattern_recognition_example();

}  // namespace my_opennn

int opennn_main(int argc, char *argv[])
{
	my_opennn::simple_function_regression_example();
	//my_opennn::simple_pattern_recognition_example();

	return 0;
}
