//include "stdafx.h"
#include <tiny_dnn/tiny_dnn.h>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_tiny_dnn {

void convnet_sample(const std::string& data_dir_path);
void mlp_sample(const std::string& data_dir_path);
void denoising_auto_encoder_sample(const std::string& data_dir_path);
void dropout_sample(const std::string& data_dir_path);

void mnist_train_example();
void mnist_test_example();
void cifar10_train_example();
void cifar10_test_example();

}  // namespace my_tiny_dnn

int tiny_dnn_main(int argc, char *argv[])
{
	try
	{
		if (false)
		{
			const std::string path_to_dataset("./data/machine_learning/mnist");

			std::cout << "\tConvolutional neural networks (LeNet-5 like architecture) ---" << std::endl;
			my_tiny_dnn::convnet_sample(path_to_dataset);
			std::cout << "\t3-Layer Networks (MLP) --------------------------------------" << std::endl;
			my_tiny_dnn::mlp_sample(path_to_dataset);
			std::cout << "\tDenoising auto-encoder --------------------------------------" << std::endl;
			my_tiny_dnn::denoising_auto_encoder_sample(path_to_dataset);
			std::cout << "\tDropout -----------------------------------------------------" << std::endl;
			my_tiny_dnn::dropout_sample(path_to_dataset);
		}

		std::cout << "\tTrain MNIST -------------------------------------------------" << std::endl;
		//my_tiny_dnn::mnist_train_example();
		std::cout << "\tTest MNIST --------------------------------------------------" << std::endl;
		my_tiny_dnn::mnist_test_example();

		std::cout << "\tTrain CIFAR10 -----------------------------------------------" << std::endl;
		//my_tiny_dnn::cifar10_train_example();
		std::cout << "\tTest CIFAR10 ------------------------------------------------" << std::endl;
		my_tiny_dnn::cifar10_test_example();
	}
	catch (const tiny_dnn::nn_error &ex)
	{
		std::cout << "tiny_dnn::nn_error caught: " << ex.what() << std::endl;

		return 1;
	}

	return 0;
}
