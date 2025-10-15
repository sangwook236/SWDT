//include "stdafx.h"
#include <cassert>
#include <cmath>
#include <chrono>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <windows.h>
#include <stringapiset.h>
#endif
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


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
	const std::string mnist_dir_path("path/to/mnist");

	// REF [site] >> https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/MNIST
	const std::string onnx_filepath(mnist_dir_path + "/mnist.onnx");
	//const std::wstring onnx_filepath(mnist_dir_path + L"/mnist.onnx");

	static constexpr const int image_width(28);
	static constexpr const int image_height(28);
	static constexpr const int image_channel(1);
	static constexpr const int num_classes(10);

	std::array<float, image_width * image_height> input_image{};
	{
		// REF [site] >> https://huggingface.co/datasets/ylecun/mnist
		const std::string image_path(mnist_dir_path + "/0.jpg");
		//const std::string image_path(mnist_dir_path + "/1.jpg");
		//const std::string image_path(mnist_dir_path + "/2.jpg");
		//const std::string image_path(mnist_dir_path + "/4.jpg");
		//const std::string image_path(mnist_dir_path + "/7.jpg");

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
#if 0
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
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
		const auto len(MultiByteToWideChar(CP_ACP, 0, onnx_filepath.c_str(), (int)onnx_filepath.length(), NULL, NULL));
		wchar_t onnx_filepath_ws[MAX_PATH] = { L'\0', };
		MultiByteToWideChar(CP_ACP, 0, onnx_filepath.c_str(), (int)onnx_filepath.length(), onnx_filepath_ws, len);
		Ort::Session session(env, onnx_filepath_ws, session_options);
#else
		Ort::Session session(env, onnx_filepath.c_str(), session_options);
#endif

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
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
		const auto len(MultiByteToWideChar(CP_ACP, 0, onnx_filepath.c_str(), (int)onnx_filepath.length(), NULL, NULL));
		wchar_t onnx_filepath_ws[MAX_PATH] = { L'\0', };
		MultiByteToWideChar(CP_ACP, 0, onnx_filepath.c_str(), (int)onnx_filepath.length(), onnx_filepath_ws, len);
		Ort::Session session(env, onnx_filepath_ws, Ort::SessionOptions(nullptr));
#else
		Ort::Session session(env, onnx_filepath.c_str(), Ort::SessionOptions(nullptr));
#endif

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#endif
		std::cout << "An ONNX session created: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

		// Show session info
		{
			const size_t num_inputs(session.GetInputCount());
			const size_t num_outputs(session.GetOutputCount());
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

				std::cout << "\tInput #" << idx << ':' << std::endl;
				std::cout << "\t\tName: " << input_names[idx] << std::endl;
				std::cout << "\t\tShape (dimension = " << shape.size() << "): ";
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

				std::cout << "\tOuput #" << idx << ':' << std::endl;
				std::cout << "\t\tName: " << output_names[idx] << std::endl;
				std::cout << "\t\tShape (dimension = " << shape.size() << "): ";
				std::copy(shape.begin(), shape.end(), std::ostream_iterator<int64_t>(std::cout, ", "));
				std::cout << std::endl;
			}
		}

		//-----
		// Infer

		// "Input3": float32[1,1,28,28], "Plus214_Output_0": float32[1,10]
		//	Can check the names of inputs & outputs in https://netron.app/
#if 0
		const char* input_names[] = { "Input3" };
		const char* output_names[] = { "Plus214_Output_0" };

		std::array<int64_t, 4> input_shape{ 1, image_channel, image_width, image_height };
		std::array<int64_t, 2> output_shape{ 1, num_classes };

		const size_t num_inputs(1);
		const size_t num_outputs(1);

		assert(session.GetInputCount() == num_inputs);
		assert(session.GetOutputCount() == num_outputs);
#elif 1
		const std::vector<const char*> input_names = { "Input3" };
		const std::vector<const char*> output_names = { "Plus214_Output_0" };

		const std::vector<std::vector<int64_t>> input_shapes{ { 1, image_channel, image_width, image_height } };
		const std::vector<std::vector<int64_t>> output_shapes{ { 1, num_classes } };

		const size_t num_inputs(input_names.size());
		const size_t num_outputs(output_names.size());

		assert(session.GetInputCount() == num_inputs);
		assert(session.GetOutputCount() == num_outputs);
#else
		const size_t num_inputs(session.GetInputCount());
		const size_t num_outputs(session.GetOutputCount());

		//const std::vector<std::string>& input_names = session.GetInputNames();
		//const std::vector<std::string>& output_names = session.GetOutputNames();
		std::vector<const char*> input_names;
		input_names.reserve(num_inputs);
		for (auto& name : session.GetInputNames())
			input_names.push_back(name.c_str());  // Runtime error: input name cannot be empty
		std::vector<const char*> output_names;
		output_names.reserve(num_outputs);
		for (auto& name : session.GetOutputNames())
			output_names.push_back(name.c_str());  // Runtime error: output name cannot be empty

		std::vector<std::vector<int64_t>> input_shapes;
		input_shapes.reserve(num_inputs);
		for (size_t idx = 0; idx < num_inputs; ++idx)
		{
			const Ort::TypeInfo& type_info = session.GetInputTypeInfo(idx);
			//const Ort::TensorTypeAndShapeInfo& tensor_info = type_info.GetTensorTypeAndShapeInfo();
			const Ort::ConstTensorTypeAndShapeInfo& tensor_info = type_info.GetTensorTypeAndShapeInfo();
			const std::vector<int64_t>& shape = tensor_info.GetShape();
			input_shapes.push_back(shape);
		}
		std::vector<std::vector<int64_t>> output_shapes;
		output_shapes.reserve(num_outputs);
		for (size_t idx = 0; idx < num_outputs; ++idx)
		{
			const Ort::TypeInfo& type_info = session.GetOutputTypeInfo(idx);
			//const Ort::TensorTypeAndShapeInfo& tensor_info = type_info.GetTensorTypeAndShapeInfo();
			const Ort::ConstTensorTypeAndShapeInfo& tensor_info = type_info.GetTensorTypeAndShapeInfo();
			const std::vector<int64_t>& shape = tensor_info.GetShape();
			output_shapes.push_back(shape);
		}
#endif

		std::cout << "Inferring..." << std::endl;
		start_time = std::chrono::steady_clock::now();
		//Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.data(), input_image.size(), input_shapes[0].data(), input_shapes[0].size());
#if 1
		//Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), output_shape.data(), output_shape.size());
		Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), output_shapes[0].data(), output_shapes[0].size());
		//session.Run(Ort::RunOptions(nullptr), input_names, &input_tensor, num_inputs, output_names, &output_tensor, num_outputs);
		session.Run(Ort::RunOptions(nullptr), input_names.data(), &input_tensor, num_inputs, output_names.data(), &output_tensor, num_outputs);
#else
		std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions(nullptr), input_names, &input_tensor, num_inputs, output_names, num_outputs);
		//std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions(nullptr), input_names.data(), &input_tensor, num_inputs, output_names.data(), num_outputs);
		float* output_ptr = output_tensor[0].GetTensorMutableData<float>();
		std::copy(output_ptr, output_ptr + num_classes, results.data());
#endif
		softmax(results);
		std::cout << "Inferred: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;
		//std::cout << "Results dim = " << results.size() << std::endl;
	}

	const int64_t predicted(std::distance(results.begin(), std::max_element(results.begin(), results.end())));
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
	//	https://onnxruntime.ai/

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

		// ONNX on TensorRT
		//	Refer to tensorrt_main.cpp
	}
	catch (const Ort::Exception& ex)
	{
		std::cerr << "Ort::Exception caught: " << ex.what() << std::endl;
		//return 1;
	}

	return 0;
}
