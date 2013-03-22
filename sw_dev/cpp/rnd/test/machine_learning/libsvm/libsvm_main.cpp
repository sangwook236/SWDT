//#include "stdafx.h"
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libsvm {

void train_example();
void predict_example();

}  // namespace my_libsvm

int libsvm_main(int argc, char *argv[])
{
	my_libsvm::train_example();
	my_libsvm::predict_example();

	return 0;
}
