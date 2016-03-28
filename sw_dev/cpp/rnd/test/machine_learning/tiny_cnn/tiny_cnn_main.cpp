//include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_tiny_cnn {

void mnist_example();
void cifar10_example();

}  // namespace my_tiny_cnn

int tiny_cnn_main(int argc, char *argv[])
{
	my_tiny_cnn::mnist_example();
	my_tiny_cnn::cifar10_example();

	return 0;
}
