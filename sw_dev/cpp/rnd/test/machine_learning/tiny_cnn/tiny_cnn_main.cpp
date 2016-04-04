//include "stdafx.h"
#include <tiny_cnn/tiny_cnn.h>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_tiny_cnn {

void convnet_sample(const std::string& data_dir_path);
void mlp_sample(const std::string& data_dir_path);
void denoising_auto_encoder_sample();
void dropout_sample(const std::string& data_dir_path);

void mnist_train_example();
void mnist_test_example();
void cifar10_train_example();

}  // namespace my_tiny_cnn

int tiny_cnn_main(int argc, char *argv[])
{
	try
	{
		{
			const std::string path_to_data("./data/machine_learning");
			my_tiny_cnn::convnet_sample(path_to_data);
			my_tiny_cnn::mlp_sample(path_to_data);
			my_tiny_cnn::denoising_auto_encoder_sample();
			my_tiny_cnn::dropout_sample(path_to_data);
		}

		my_tiny_cnn::mnist_train_example();
		my_tiny_cnn::mnist_test_example();
		my_tiny_cnn::cifar10_train_example();
	}
	catch (const tiny_cnn::nn_error& e)
	{
		std::cout << e.what() << std::endl;
	}

	return 0;
}
