//include "stdafx.h"
#include <cmath>
#include <chrono>
#include <string>
#include <iostream>
#include <onnxruntime_cxx_api.h>

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#pragma comment(lib, "onnxruntime.lib")
#endif


namespace {
namespace local {

template <typename T>
void softmax(T& input)
{
	const float rowmax = *std::max_element(input.begin(), input.end());
	std::vector<float> y(input.size());
	float sum = 0.0f;
	for (size_t i = 0; i != input.size(); ++i)
	{
		sum += y[i] = std::exp(input[i] - rowmax);
	}
	for (size_t i = 0; i != input.size(); ++i)
	{
		input[i] = y[i] / sum;
	}
}

// REF [site] >> https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/MNIST/MNIST.cpp
void onnx_runtime_mnist_example()
{
	const std::string onnx_filepath("model.onnx");
	//const std::wstring onnx_filepath(L"model.onnx");

	static constexpr const int width = 28;
	static constexpr const int height = 28;
	static constexpr const int channel = 1;
	static constexpr const int num_classes = 10;

	std::array<float, width * height> input_image{};
	std::array<float, 10> results{};

	// Can check the names of inputs & outputs in https://netron.app/.
	const char* input_names[] = {"Input3"};
	const char* output_names[] = {"Plus214_Output_0"};

	std::array<int64_t, 4> input_shape{1, channel, width, height};
	std::array<int64_t, 2> output_shape{1, num_classes};

	std::cout << "Loading a model from " << onnx_filepath << "..." << std::endl;
	auto start_time =std::chrono::steady_clock::now();
	Ort::Env env;
	//Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_log");
	Ort::Session session(env, onnx_filepath.c_str(), Ort::SessionOptions(nullptr));
	std::cout << "A model loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

	std::cout << "Inferring..." << std::endl;
	start_time = std::chrono::steady_clock::now();
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());
	Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), output_shape.data(), output_shape.size());

	session.Run(Ort::RunOptions(nullptr), input_names, &input_tensor, 1, output_names, &output_tensor, 1);
	std::cout << "Inferred: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

	softmax(results);
	const int64_t result = std::distance(results.begin(), std::max_element(results.begin(), results.end()));
	std::cout << "Result = " << result << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_onnx {

}  // namespace my_onnx

int onnx_main(int argc, char *argv[])
{
	local::onnx_runtime_mnist_example();

	return 0;
}
