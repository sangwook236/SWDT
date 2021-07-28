//#include "stdafx.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

// REF [site] >>
//	https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/convolutional_neural_network/inlcude/convnet.h
//	https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/convolutional_neural_network/src/convnet.cpp
class ConvNetImpl : public torch::nn::Module
{
public:
	explicit ConvNetImpl(int64_t num_classes = 10)
	: fc_(7 * 7 * 32, num_classes)
	{
		register_module("layer1", layer1_);
		register_module("layer2", layer2_);
		register_module("fc", fc_);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = layer1_->forward(x);
		x = layer2_->forward(x);
		x = x.view({-1, 7 * 7 * 32});
		return fc_->forward(x);
	}

private:
	torch::nn::Sequential layer1_{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)),
		torch::nn::BatchNorm2d(16),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
	};

	torch::nn::Sequential layer2_{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
		torch::nn::BatchNorm2d(32),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
	};

	torch::nn::Linear fc_;
};

TORCH_MODULE(ConvNet);

// REF [site] >> https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/convolutional_neural_network/src/main.cpp
void cnn_mnist_tutorial()
{
	std::cout << "Convolutional Neural Network." << std::endl << std::endl;

	// Device.
	const auto cuda_available = torch::cuda::is_available();
	const torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

	// Hyper parameters.
	const int64_t num_classes = 10;
	const int64_t batch_size = 100;
	const size_t num_epochs = 5;
	const double learning_rate = 0.001;

	const std::string MNIST_data_path = "/home/sangwook/work/dataset/text/mnist/";

	// MNIST dataset.
	auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	// Number of samples in the training set.
	const auto num_train_samples = train_dataset.size().value();

	auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	// Number of samples in the test set.
	const auto num_test_samples = test_dataset.size().value();

	// Data loader.
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);

	// Model.
	ConvNet model(num_classes);
	model->to(device);

	// Optimizer.
	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

	// Set floating point output precision.
	std::cout << std::fixed << std::setprecision(4);

	//--------------------
	// Train the model.
	std::cout << "Training..." << std::endl;
	model->train();

	for (size_t epoch = 0; epoch != num_epochs; ++epoch)
	{
		// Initialize running metrics.
		double running_loss = 0.0;
		size_t num_correct = 0;

		for (const auto &batch: *train_loader)
		{
			// Transfer images and target labels to device.
			const auto data = batch.data.to(device);
			const auto target = batch.target.to(device);

			// Forward pass.
			const auto output = model->forward(data);

			// Calculate loss.
			const auto loss = torch::nn::functional::cross_entropy(output, target);

			// Update running loss.
			running_loss += loss.item<double>() * data.size(0);

			// Calculate prediction.
			const auto prediction = output.argmax(1);

			// Update number of correctly classified samples.
			num_correct += prediction.eq(target).sum().item<int64_t>();

			// Backward pass and optimize.
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}

		const auto sample_mean_loss = running_loss / num_train_samples;
		const auto accuracy = static_cast<double>(num_correct) / num_train_samples;

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			<< sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
	}

	std::cout << "Training finished!" << std::endl << std::endl;

	//--------------------
	// Test the model.
	std::cout << "Testing..." << std::endl;
	model->eval();
	torch::NoGradGuard no_grad;

	double running_loss = 0.0;
	size_t num_correct = 0;
	for (const auto &batch: *test_loader)
	{
		const auto data = batch.data.to(device);
		const auto target = batch.target.to(device);

		const auto output = model->forward(data);

		const auto loss = torch::nn::functional::cross_entropy(output, target);
		running_loss += loss.item<double>() * data.size(0);

		const auto prediction = output.argmax(1);
		num_correct += prediction.eq(target).sum().item<int64_t>();
	}

	std::cout << "Testing finished!" << std::endl;

	const auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
	const auto test_sample_mean_loss = running_loss / num_test_samples;

	std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << std::endl;
}

void get_module_params(std::vector<at::Tensor> &parameters, const torch::jit::script::Module &script_module)
{
	for (const auto &params: script_module.parameters())
		parameters.push_back(params);
	// FIXME [fix] >> Segmentation fault.
	//for (const auto &childModule: script_module.modules())
	//	get_module_params(parameters, childModule);
}

// REF [function] >> cnn_mnist_tutorial().
void resnet_mnist_torch_script_example()
{
	// Device.
	const auto cuda_available = torch::cuda::is_available();
	const torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

	// Hyper parameters.
	const int64_t num_classes = 10;
	const int64_t batch_size = 100;
	const size_t num_epochs = 5;
	const double learning_rate = 0.001;

	const std::string MNIST_data_path = "/home/sangwook/work/dataset/text/mnist/";

	// MNIST dataset.
	auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	// Number of samples in the training set.
	const auto num_train_samples = train_dataset.size().value();

	auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	// Number of samples in the test set.
	const auto num_test_samples = test_dataset.size().value();

	// Data loader.
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);

	// Load a Torch Script model.
	// REF [python] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_torch_script.py
	const std::string script_module_filepath("./resnet_mnist_ts_model.pth");

	torch::jit::script::Module script_module;
	try
	{
		// Deserialize a ScriptModule from a file using torch::jit::load().
		script_module = torch::jit::load(script_module_filepath);
	}
	catch (const c10::Error &ex)
	{
		std::cerr << "[Error] Failed to load a script module, " << script_module_filepath << ": " << ex.what() << std::endl;
		return;
	}
	script_module.to(device);

	std::vector<at::Tensor> script_module_params;
	get_module_params(script_module_params, script_module);

	// Optimizer.
	torch::optim::Adam optimizer(script_module_params, torch::optim::AdamOptions(learning_rate));

	// Set floating point output precision.
	std::cout << std::fixed << std::setprecision(4);

	//--------------------
	// Train the model.
	std::cout << "Training..." << std::endl;
	script_module.train();

	for (size_t epoch = 0; epoch != num_epochs; ++epoch)
	{
		// Initialize running metrics.
		double running_loss = 0.0;
		size_t num_correct = 0;

		for (const auto &batch: *train_loader)
		{
			// Transfer images and target labels to device.
			const auto data = batch.data.to(device);
			const auto target = batch.target.to(device);

			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(data);
			//inputs.push_back(torch::ones({1, 1, 28, 28}));

			// Forward pass.
			const auto output = script_module.forward(inputs).toTensor();

			// Calculate loss.
			const auto loss = torch::nn::functional::cross_entropy(output, target);

			// Update running loss.
			running_loss += loss.item<double>() * data.size(0);

			// Calculate prediction.
			const auto prediction = output.argmax(1);

			// Update number of correctly classified samples.
			num_correct += prediction.eq(target).sum().item<int64_t>();

			// Backward pass and optimize.
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}

		const auto sample_mean_loss = running_loss / num_train_samples;
		const auto accuracy = static_cast<double>(num_correct) / num_train_samples;

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			<< sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
	}

	std::cout << "Training finished!" << std::endl << std::endl;

	//--------------------
	// Test the model.
	std::cout << "Testing..." << std::endl;
	script_module.eval();
	torch::NoGradGuard no_grad;

	double running_loss = 0.0;
	size_t num_correct = 0;
	for (const auto &batch: *test_loader)
	{
		const auto data = batch.data.to(device);
		const auto target = batch.target.to(device);

		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(data);

		const auto output = script_module.forward(inputs).toTensor();

		const auto loss = torch::nn::functional::cross_entropy(output, target);
		running_loss += loss.item<double>() * data.size(0);

		const auto prediction = output.argmax(1);
		num_correct += prediction.eq(target).sum().item<int64_t>();
	}

	std::cout << "Testing finished!" << std::endl;

	const auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
	const auto test_sample_mean_loss = running_loss / num_test_samples;

	std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_torch {

void training_example()
{
	local::cnn_mnist_tutorial();
	//local::resnet_mnist_torch_script_example();
}

}  // namespace my_torch
