//#include "stdafx.h"
#include <stdexcept>
#include <memory>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>


namespace {
namespace local {

#if 0

struct Net: torch::nn::Module
{
	Net(int64_t N, int64_t M)
	{
		// register_parameter() & register_module() are needed if we want to use the parameters() method later on.
		W = register_parameter("W", torch::randn({N, M}));
		b = register_parameter("b", torch::randn(M));
	}

	torch::Tensor forward(torch::Tensor input)
	{
		return torch::addmm(b, input, W);
	}

	torch::Tensor W, b;
};

#else

struct Net: torch::nn::Module
{
	Net(int64_t N, int64_t M)
	: linear(register_module("linear", torch::nn::Linear(N, M)))
	{
		another_bias = register_parameter("b", torch::randn(M));
	}

	torch::Tensor forward(torch::Tensor input)
	{
		return linear(input) + another_bias;
	}

	torch::nn::Linear linear{nullptr};  // Construct an empty holder.
	torch::Tensor another_bias;
};

#endif

struct Net2Impl: torch::nn::Module
{
	Net2Impl(int64_t in, int64_t out)
	: weight(register_parameter("weight", torch::randn({in, out})))
	{
		bias = register_parameter("bias", torch::randn(out));
	}

	torch::Tensor forward(const torch::Tensor &input)
	{
		return torch::addmm(bias, input, weight);
	}

	torch::Tensor weight, bias;
};
// Module holder.
//	This "generated" class is effectively a wrapper over a std::shared_ptr<LinearImpl>.
TORCH_MODULE(Net2);

void call_by_val(Net net) { }
void call_by_ref(Net &net) { }
void call_by_ptr(Net *net) { }
void call_by_shared_ptr(std::shared_ptr<Net> net) { }

// REF [site] >> https://pytorch.org/tutorials/advanced/cpp_frontend.html
void simple_frontend_tutorial()
{
	//------------------------------------------------------------
	const torch::Tensor tensor = torch::eye(3);
	std::cout << tensor << std::endl;

	//------------------------------------------------------------
	Net net(4, 5);

	for (const auto &p: net.parameters())
	{
		std::cout << p << std::endl;
	}
	for (const auto &pair: net.named_parameters())
	{
		std::cout << pair.key() << ": " << pair.value() << std::endl;
	}

	std::cout << net.forward(torch::ones({2, 4})) << std::endl;

	call_by_val(net);
	call_by_val(std::move(net));
	call_by_ref(net);
	call_by_ptr(&net);

	auto net_p = std::make_shared<Net>(4, 5);
	call_by_shared_ptr(net_p);

	//------------------------------------------------------------
	Net2 net2(4, 5);

	for (const auto &p: net2->parameters())
	{
		std::cout << p << std::endl;
	}
	for (const auto &pair: net2->named_parameters())
	{
		std::cout << pair.key() << ": " << pair.value() << std::endl;
	}

	std::cout << net2->forward(torch::ones({2, 4})) << std::endl;
}

// Define a new Module.
struct Mlp: torch::nn::Module
{
	Mlp()
	{
		// Construct and register three Linear submodules.
		fc1 = register_module("fc1", torch::nn::Linear(784, 64));
		fc2 = register_module("fc2", torch::nn::Linear(64, 32));
		fc3 = register_module("fc3", torch::nn::Linear(32, 10));
	}

	// Implement the Mlp's algorithm.
	torch::Tensor forward(torch::Tensor x)
	{
		// Use one of many tensor manipulation functions.
		x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
		x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		x = torch::relu(fc2->forward(x));
		x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
		return x;
	}

	// Use one of many "standard library" modules.
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

// REF [site] >> https://pytorch.org/cppdocs/frontend.html
void mlp_frontend_tutorial()
{
	const std::string MNIST_data_path("/home/sangwook/work/dataset/text/mnist/");

	//------------------------------------------------------------
	// Create a new Mlp.
	auto net = std::make_shared<Mlp>();

	//------------------------------------------------------------
	// Create a multi-threaded data loader for the MNIST dataset.
	auto data_loader = torch::data::make_data_loader(
		torch::data::datasets::MNIST(MNIST_data_path).map(torch::data::transforms::Stack<>()),
		/*batch_size=*/64
	);

	//------------------------------------------------------------
	// Instantiate an SGD optimization algorithm to update our Mlp's parameters.
	torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

	for (size_t epoch = 1; epoch <= 10; ++epoch)
	{
		size_t batch_index = 0;
		// Iterate the data loader to yield batches from the dataset.
		for (auto &batch: *data_loader)
		{
			// Reset gradients.
			optimizer.zero_grad();
			// Execute the model on the input data.
			torch::Tensor prediction = net->forward(batch.data);
			// Compute a loss value to judge the prediction of our model.
			torch::Tensor loss = torch::nll_loss(prediction, batch.target);
			// Compute gradients of the loss w.r.t. the parameters of our model.
			loss.backward();
			// Update the parameters based on the calculated gradients.
			optimizer.step();

			// Output the loss and checkpoint every 100 batches.
			if (++batch_index % 100 == 0)
			{
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;

				// Serialize your model periodically as a checkpoint.
				torch::save(net, "mlp.pt");
			}
		}
	}
}

struct DCGANGeneratorImpl: torch::nn::Module
{
	DCGANGeneratorImpl(int kNoiseSize)
	: conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
		batch_norm1(256),
		conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
		batch_norm2(128),
		conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
		batch_norm3(64),
		conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))
	{
		// Register_module() is needed if we want to use the parameters() method later on.
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);
		register_module("batch_norm1", batch_norm1);
		register_module("batch_norm2", batch_norm2);
		register_module("batch_norm3", batch_norm3);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(batch_norm1(conv1(x)));
		x = torch::relu(batch_norm2(conv2(x)));
		x = torch::relu(batch_norm3(conv3(x)));
		x = torch::tanh(conv4(x));
		return x;
	}

	torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
	torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANGenerator);

// REF [site] >> https://pytorch.org/tutorials/advanced/cpp_frontend.html
void dcgan_frontend_tutorial()
{
	const std::string MNIST_data_path("/home/sangwook/work/dataset/text/mnist/");

	const int kNoiseSize = 100;
	const size_t kBatchSize = 64, kNumberOfEpochs = 30;
	const size_t kCheckpointEvery = 1000;
	const bool kRestoreFromCheckpoint = false;
	const size_t num_workers = 2;

	const bool is_cuda_available = torch::cuda::is_available();
	const torch::Device device(is_cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (is_cuda_available ? "Device: CUDA." : "Device: CPU.") << std::endl;

	//------------------------------------------------------------
	// Build modules.
	DCGANGenerator generator(kNoiseSize);
	torch::nn::Sequential discriminator(
		// Layer 1.
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// Layer 2.
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(128),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// Layer 3.
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// Layer 4.
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
		torch::nn::Sigmoid()
	);

	generator->to(device);
	discriminator->to(device);

	//------------------------------------------------------------
	// Load data.
	auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());

	const auto num_train_samples = dataset.size().value();
	const size_t batches_per_epoch = (size_t)std::ceil(float(num_train_samples) / float(kBatchSize));

# if 0
	auto data_loader = torch::data::make_data_loader(std::move(dataset));
#else
	auto data_loader = torch::data::make_data_loader(
		std::move(dataset),
		torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(num_workers)
	);
#endif

	if (false)
		for (torch::data::Example<> &batch: *data_loader)
		{
			std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
			for (int64_t i = 0; i < batch.data.size(0); ++i)
				std::cout << batch.target[i].item<int64_t>() << " ";
			std::cout << std::endl;
		}

	//------------------------------------------------------------
	// Train.
	torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.999)));
	torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.999)));

	// Recover the training state.
	if (kRestoreFromCheckpoint)
	{
		torch::load(generator, "generator-checkpoint.pt");
		torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
		torch::load(discriminator, "discriminator-checkpoint.pt");
		torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
	}

	int64_t checkpoint_counter = 0;
	for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
	{
		int64_t batch_index = 0;
		for (torch::data::Example<> &batch: *data_loader)
		{
			// Train discriminator with real images.
			discriminator->zero_grad();
			torch::Tensor real_images = batch.data.to(device);
			torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0).to(device);
			torch::Tensor real_output = discriminator->forward(real_images);  // (64, 1, 1, 1).
			real_output = torch::squeeze(real_output);
			torch::Tensor dis_loss_real = torch::binary_cross_entropy(real_output, real_labels);
			dis_loss_real.backward();

			// Train discriminator with fake images.
			torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
			torch::Tensor fake_images = generator->forward(noise);
			torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
			torch::Tensor fake_output = discriminator->forward(fake_images.detach());  // (64, 1, 1, 1).
			fake_output = torch::squeeze(fake_output);
			torch::Tensor dis_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
			dis_loss_fake.backward();

			torch::Tensor dis_loss = dis_loss_real + dis_loss_fake;
			discriminator_optimizer.step();

			// Train generator.
			generator->zero_grad();
			fake_labels.fill_(1);
			fake_output = discriminator->forward(fake_images);  // (64, 1, 1, 1).
			fake_output = torch::squeeze(fake_output);
			torch::Tensor gen_loss = torch::binary_cross_entropy(fake_output, fake_labels);
			gen_loss.backward();
			generator_optimizer.step();

			std::printf(
				"\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
				epoch, kNumberOfEpochs,
				++batch_index, batches_per_epoch,
				dis_loss.item<float>(), gen_loss.item<float>()
			);

			// Checkpoint the training state.
			if (batch_index % kCheckpointEvery == 0)
			{
				// Checkpoint the model and optimizer state.
				torch::save(generator, "generator-checkpoint.pt");
				torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
				torch::save(discriminator, "discriminator-checkpoint.pt");
				torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");

				// Sample the generator and save the images.
				torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
				torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
				std::cout << "\n-> checkpoint " << ++checkpoint_counter << std::endl;
			}
		}
	}

	//------------------------------------------------------------
	// Inspect generated images.
	//	REF [file] >> ./inspect_dcgan_generated_images.py
}

// REF [site] >>
//	https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/convolutional_neural_network/inlcude/convnet.h
//	https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/convolutional_neural_network/src/convnet.cpp
class ConvNetImpl: public torch::nn::Module
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

	const std::string MNIST_data_path("/home/sangwook/work/dataset/text/mnist/");

	// MNIST dataset.
	// REF [site] >>
	//	https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/datasets/mnist.h
	//	https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp
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
void lenet_mnist_torch_script_example()
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

	const std::string MNIST_data_path("/home/sangwook/work/dataset/text/mnist/");

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
	// REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_torch_script.py
	const std::string script_module_filepath("./lenet_mnist_ts_model.pth");

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
	//local::simple_frontend_tutorial();
	local::mlp_frontend_tutorial();
	//local::dcgan_frontend_tutorial();

	//local::cnn_mnist_tutorial();
	//local::lenet_mnist_torch_script_example();
}

}  // namespace my_torch
