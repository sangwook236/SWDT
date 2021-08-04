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
	std::cout << (torch::hasCUDA() ? "CUDA available." : "No CUDA available.") << std::endl;

	//--------------------
	// Load a script module in C++.

	// REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_torch_script.py
	const std::string script_module_filepath("./resnet_ts_model.pth");
	const at::ArrayRef<int64_t> input_shape = {1, 3, 224, 224};

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

	//--------------------
	// Execute the script module in C++.

	if (torch::hasCUDA())
		script_module.to(at::kCUDA);

	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	if (torch::hasCUDA())
		inputs.push_back(torch::ones(input_shape).to(at::kCUDA));
	else
		inputs.push_back(torch::ones(input_shape));

	// Execute the model and turn its output into a tensor.
	at::Tensor output = script_module.forward(inputs).toTensor();
	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
}

}  // namespace my_torch
