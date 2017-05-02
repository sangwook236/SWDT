//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libsvm {

void train_example();
void predict_example();
void simple_svdd();
void simple_one_class_svm();

}  // namespace my_libsvm

int libsvm_main(int argc, char *argv[])
{
	// Example.
	//my_libsvm::train_example();
	//my_libsvm::predict_example();

	// One-class classification.
	//	- Support vector data description.
	//	- One-class SVM.
	//my_libsvm::simple_svdd();
	my_libsvm::simple_one_class_svm();

	return 0;
}
