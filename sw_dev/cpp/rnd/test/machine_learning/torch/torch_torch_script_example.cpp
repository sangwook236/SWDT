#include <memory>
#include <chrono>
#include <vector>
#include <iostream>
#include <torch/script.h>

// LibTorch:
//	Download from https://pytorch.org/


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_torch {

// REF [site] >> https://pytorch.org/tutorials/advanced/cpp_export.html.
void torch_script_example()
{
	const torch::DeviceIndex gpu = -1;
	//const auto device(torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu) : torch::Device(torch::kCPU));
	const auto device(torch::hasCUDA() ? torch::Device(torch::kCUDA, gpu) : torch::Device(torch::kCPU));
	std::cout << (device.is_cuda() ? "Device: CUDA." : "Device: CPU.") << std::endl;

	//--------------------
	// Load a script module in C++.

	// REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_torch_script.py
	const std::string script_module_filepath("./resnet_ts_model.pth");
	const std::vector<int64_t> input_shape = {1, 3, 224, 224};

	std::cout << "Loading a model from " << script_module_filepath << "..." << std::endl;
	auto start_time = std::chrono::steady_clock::now();
	torch::jit::script::Module script_module;
	try
	{
		// Deserialize a ScriptModule from a file using torch::jit::load().
		script_module = torch::jit::load(script_module_filepath);
	}
	catch (const c10::Error &ex)
	{
		std::cerr << "Error: Script module not loaded, " << script_module_filepath << ": " << ex.what() << std::endl;
		return;
	}
	std::cout << "A model loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

	if (device.is_cuda())
		script_module.to(device);

	//--------------------
	// Execute the script module in C++.

	std::cout << "Inferring..." << std::endl;
	start_time = std::chrono::steady_clock::now();
	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	if (device.is_cuda())
		inputs.push_back(torch::ones(c10::IntArrayRef(input_shape)).to(device));
	else
		inputs.push_back(torch::ones(c10::IntArrayRef(input_shape)));

	// Execute the model and turn its output into a tensor.
	at::Tensor output = script_module.forward(inputs).toTensor();
	if (device.is_cuda())
		output = output.to(at::kCPU);
	std::cout << "Inferred: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;
	std::cout << "Output = " << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
}

}  // namespace my_torch
