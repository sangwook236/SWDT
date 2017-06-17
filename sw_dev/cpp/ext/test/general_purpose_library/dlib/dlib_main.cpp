//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_dlib {

void max_cost_assignment_example();
void graph_labeling_example();

void svm_struct_example();
void dnn_example();

}  // namespace my_dlib

int dlib_main(int argc, char *argv[])
{
	// Matrix operation.
	// REF [file] >> ${DLIB_HOME}/examples/matrix_ex.cpp

	// Assignment problem (use Hungarian/Kuhn-Munkres algorithm).
	my_dlib::max_cost_assignment_example();

	// Graph labeling.
	//my_dlib::graph_labeling_example();

	// Structured SVM.
	//my_dlib::svm_struct_example();

	// Deep learning.
	//my_dlib::dnn_example();

	return 0;
}
