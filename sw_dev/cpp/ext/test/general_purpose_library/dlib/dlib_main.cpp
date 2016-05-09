//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_dlib {

void svm_struct_example();
void graph_labeling_example();

}  // namespace my_dlib

int dlib_main(int argc, char *argv[])
{
	// matrix operation example.
	// REF [file] >> ${DLIB_HOME}/examples/matrix_ex.cpp

	// structured SVM example.
	my_dlib::svm_struct_example();

	// graph labeling example.
	my_dlib::graph_labeling_example();

	return 0;
}
