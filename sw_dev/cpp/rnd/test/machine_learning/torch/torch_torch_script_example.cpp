#include <torch/script.h>
#include <vector>
#include <iostream>
#include <memory>

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
	const bool is_cuda_available = torch::hasCUDA();
	std::cout << (is_cuda_available ? "Device: CUDA." : "Device: CPU.") << std::endl;

	//--------------------
	// Load a script module in C++.

	// REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_torch_script.py
	const std::string script_module_filepath("./resnet_ts_model.pth");
	const std::vector<int64_t> input_shape = {1, 3, 224, 224};

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
	std::cout << "Info: A script module loaded from " << script_module_filepath << ": " << std::endl;

	if (is_cuda_available)
		script_module.to(at::kCUDA);

	//--------------------
	// Execute the script module in C++.

	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	if (is_cuda_available)
		inputs.push_back(torch::ones(c10::IntArrayRef(input_shape)).to(at::kCUDA));
	else
		inputs.push_back(torch::ones(c10::IntArrayRef(input_shape)));

	// Execute the model and turn its output into a tensor.
	at::Tensor output = script_module.forward(inputs).toTensor();
	if (is_cuda_available)
		output = output.to(at::kCPU);
	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
}

}  // namespace my_torch
