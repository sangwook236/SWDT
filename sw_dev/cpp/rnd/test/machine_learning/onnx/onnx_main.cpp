//include "stdafx.h"
#include <cassert>
#include <cmath>
#include <chrono>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#pragma comment(lib, "onnxruntime.lib")
#endif


namespace {
namespace local {

template <typename T>
static void softmax(T& input)
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

// REF [site] >> https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/MNIST
void onnx_runtime_mnist_example()
{
	// REF [site] >> https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/MNIST
	const std::string onnx_filepath("path/to/mnist.onnx");
	//const std::wstring onnx_filepath(L"path/to/mnist.onnx");

	static constexpr const int image_width = 28;
	static constexpr const int image_height = 28;
	static constexpr const int image_channel = 1;
	static constexpr const int num_classes = 10;

	std::array<float, image_width * image_height> input_image{};
	{
		// REF [site] >> https://huggingface.co/datasets/ylecun/mnist
		const std::string image_path("path/to/mnist/0.jpg");
		//const std::string image_path("path/to/mnist1.jpg");
		//const std::string image_path("path/to/mnist2.jpg");
		//const std::string image_path("path/to/mnist4.jpg");
		//const std::string image_path("path/to/mnist7.jpg");

		const cv::Mat& gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (gray.empty())
		{
			std::cerr << "Failed to load image, " << image_path << std::endl;
			return;
		}
		//cv::resize(gray, gray, cv::Size(image_width, image_height));

		float* ptr = input_image.data();
		for (unsigned y = 0; y < image_height; ++y)
			for (unsigned x = 0; x < image_width; ++x, ++ptr)
				*ptr += gray.at<unsigned char>(y, x) == 0 ? 0.0f : 1.0f;
	}

	std::array<float, num_classes> results{};
	{
		std::cout << "Creating an ONNX session from " << onnx_filepath << "..." << std::endl;
		auto start_time(std::chrono::steady_clock::now());
		Ort::Env env;
		//Ort::Env env(ORT_LOGGING_LEVEL_WARNING, /*logid =*/ "onnx_log");
if 0
		// For CUDA
		Ort::SessionOptions session_options;
		{
			//session_options.SetIntraOpNumThreads(1);  // Set number of threads
			//session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);  // Enable graph optimizations

#if 0
			OrtCUDAProviderOptions cuda_options;
			cuda_options.device_id = 0;  // Specify GPU device ID
			//cuda_options.gpu_mem_limit = SIZE_MAX;  // Default: SIZE_MAX
			session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
			session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});  // Enable CUDA execution provider
#endif
		}
		Ort::Session session(env, onnx_filepath.c_str(), session_options);

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUOutput);
#elif 0
		// For TensorRT
		Ort::SessionOptions session_options;
		{
			//session_options.SetIntraOpNumThreads(1);  // Set number of threads
			//session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);  // Enable graph optimizations

#if 0
			OrtTensorRTProviderOptions trt_options;
			trt_options.device_id = 0;  // Specify GPU device ID
			//trt_options.trt_max_workspace_size = 1073741824;  // Default value: 1073741824 (1GB)
			trt_options.trt_max_partition_iterations = 1000;  // Default value: 1000
			trt_options.trt_min_subgraph_size = 1;  // Default value: 1
			session_options.AppendExecutionProvider_TensorRT(trt_options);  // Runtime error
#elif 0
			// Configure TensorRT settings (e.g., enable FP16, specify engine cache path)
			OrtTensorRTProviderOptions trt_options;
			trt_options.trt_fp16_enable = 1;
			//trt_options.trt_int8_enable = 1;
			//trt_options.trt_int8_use_native_calibration_table = 1;
			trt_options.trt_engine_cache_enable = 1;
			trt_options.trt_engine_cache_path = "path/to/engine/cache";
			session_options.AppendExecutionProvider_TensorRT(trt_options);  // Runtime error
#else
			session_options.AppendExecutionProvider_TensorRT(OrtTensorRTProviderOptions{});
#endif

			OrtCUDAProviderOptions cuda_options;  // Optional: for fallback to CUDA if TensorRT doesn't support a node
			cuda_options.device_id = 0;  // Specify GPU device ID
			//cuda_options.gpu_mem_limit = SIZE_MAX;  // Default: SIZE_MAX
			session_options.AppendExecutionProvider_CUDA(cuda_options);  // Register CUDA EP for fallback
		}
		Ort::Session session(env, onnx_filepath.c_str(), session_options);

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUOutput);
#else
		// For CPU
		Ort::Session session(env, onnx_filepath.c_str(), Ort::SessionOptions(nullptr));

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#endif
		std::cout << "An ONNX session created: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;
		{
			const size_t num_inputs = session.GetInputCount();
			const size_t num_outputs = session.GetOutputCount();
			const std::vector<std::string>& input_names = session.GetInputNames();
			const std::vector<std::string>& output_names = session.GetOutputNames();
			assert(num_inputs == std::size(input_names) && num_outputs == std::size(output_names));

			//std::cout << "Input names (#inputs = " << num_inputs << "): ";
			//std::copy(input_names.begin(), input_names.end(), std::ostream_iterator<std::string>(std::cout, ", "));
			//std::cout << std::endl;
			std::cout << "Input (#inputs = " << num_inputs << "):" << std::endl;
			for (size_t idx = 0; idx < num_inputs; ++idx)
			{
				const Ort::TypeInfo& type_info = session.GetInputTypeInfo(idx);
				//const Ort::TensorTypeAndShapeInfo& tensor_info = type_info.GetTensorTypeAndShapeInfo();
				const Ort::ConstTensorTypeAndShapeInfo& tensor_info = type_info.GetTensorTypeAndShapeInfo();
				const std::vector<int64_t>& shape = tensor_info.GetShape();

				std::cout << "\tInput #" << idx << ": " << input_names[idx] << std::endl;
				std::cout << "\t\tShape = ";
				std::copy(shape.begin(), shape.end(), std::ostream_iterator<int64_t>(std::cout, ", "));
				std::cout << std::endl;
			}

			//std::cout << "Output names (#outputs = " << num_outputs << "): ";
			//std::copy(output_names.begin(), output_names.end(), std::ostream_iterator<std::string>(std::cout, ", "));
			//std::cout << std::endl;
			std::cout << "Output (#outputs = " << num_outputs << "):" << std::endl;
			for (size_t idx = 0; idx < num_outputs; ++idx)
			{
				const Ort::TypeInfo& type_info = session.GetOutputTypeInfo(idx);
				//const Ort::TensorTypeAndShapeInfo& tensor_info = type_info.GetTensorTypeAndShapeInfo();
				const Ort::ConstTensorTypeAndShapeInfo& tensor_info = type_info.GetTensorTypeAndShapeInfo();
				const std::vector<int64_t>& shape = tensor_info.GetShape();

				std::cout << "\tOuput #" << idx << ": " << output_names[idx] << std::endl;
				std::cout << "\t\tShape = ";
				std::copy(shape.begin(), shape.end(), std::ostream_iterator<int64_t>(std::cout, ", "));
				std::cout << std::endl;
			}
		}

		//-----
		// Infer

		// Can check the names of inputs & outputs in https://netron.app/
		const char* input_names[] = {"Input3"};  // float32, [1, 1, 28, 28]
		const char* output_names[] = {"Plus214_Output_0"};  // float32, [1, 10]

		std::array<int64_t, 4> input_shape{1, image_channel, image_width, image_height};
		std::array<int64_t, 2> output_shape{1, num_classes};

		std::cout << "Inferring..." << std::endl;
		start_time = std::chrono::steady_clock::now();
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());
#if 1
		Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), output_shape.data(), output_shape.size());
		session.Run(Ort::RunOptions(nullptr), input_names, &input_tensor, input_names.size(), output_names, &output_tensor, output_names.size());
#else
		std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
		float* output_ptr = output_tensor[0].GetTensorMutableData<float>();
		std::copy(output_ptr, output_ptr + num_classes, results.data());
#endif
		softmax(results);
		std::cout << "Inferred: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;
		//std::cout << "Results dim = " << results.size() << std::endl;
	}

	const int64_t predicted = std::distance(results.begin(), std::max_element(results.begin(), results.end()));
	std::cout << "Predicted = " << predicted << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_onnx {

}  // namespace my_onnx

int onnx_main(int argc, char *argv[])
{
	// ONNX Runtime
	//	https://github.com/microsoft/onnxruntime/releases

	/*
	// REF [site] >> https://onnxruntime.ai/

	// Load the model and create InferenceSession
	Ort::Env env;
	const std::string model_path("path/to/your/onnx/model");
	Ort::Session session(env, model_path, Ort::SessionOptions{ nullptr });

	// Load and preprocess the input image to inputTensor
	...

	// Run inference
	std::vector<Ort::Value>& outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, inputNames.size(), outputNames.data(), outputNames.size());

	const float* outputDataPtr = outputTensors[0].GetTensorMutableData();
	std::cout << outputDataPtr[0] << std::endl;
	*/

	try
	{
		local::onnx_runtime_mnist_example();

		// Segment Anything (SAM)
		//	Refer to sam_onnx_runtime_test.cpp
	}
	catch (const Ort::Exception& ex)
	{
		std::cerr << "Ort::Exception caught: " << ex.what() << std::endl;
		//return 1;
	}

	//-----
	// ONNX on TensorRT
	//	Refer tensorrt_main.cpp

	return 0;
}
