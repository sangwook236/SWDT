#include <cassert>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>


namespace {
namespace local {

//#define __USE_CUDA_STREAM 0

template <typename T>
static void softmax(T& input)
{
	const float rowmax = *std::max_element(input.begin(), input.end());
	std::vector<float> y(input.size());
	float sum(0.0f);
	for (size_t i = 0; i != input.size(); ++i)
	{
		sum += y[i] = std::exp(input[i] - rowmax);
	}
	for (size_t i = 0; i != input.size(); ++i)
	{
		input[i] = y[i] / sum;
	}
}

#define CUDA_CHECK(status) \
	do \
	{ \
		auto ret = (status); \
		if (ret != 0) \
		{ \
			std::cerr << "CUDA error (code = " << std::to_string(ret) << "): " << cudaGetErrorString(ret) << std::endl; \
			abort(); \
		} \
	} while (0)
//#define CUDA_CHECK(status) \
//	do \
//	{ \
//		auto ret = (status); \
//		if (ret != 0) \
//		{ \
//			throw std::runtime_error("CUDA error (code = " + std::to_string(ret) + "): " + cudaGetErrorString(ret)); \
//		} \
//	} while (0)

// Logger for TensorRT info/warning/errors
class MyLogger : public nvinfer1::ILogger
{
	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
	{
		if (severity <= nvinfer1::ILogger::Severity::kINFO)
			std::cout << msg << std::endl;
	}
};

void mnist_onnx_tensorrt_test()
{
	const size_t image_width(28);
	const size_t image_height(28);
	const size_t image_channel(1);
	const size_t num_classes(10);

	//const std::string mnist_dir_path("path/to/mnist");
	const std::string mnist_dir_path("./mnist");

	// REF [site] >> https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/MNIST
	//const std::string plan_or_onnx_path(mnist_dir_path + "/mnist.onnx");
	//const std::string plan_or_onnx_path(mnist_dir_path + "/mnist.plan");
	const std::string plan_or_onnx_path(mnist_dir_path + "/mnist_fp16.plan");

	// REF [site] >> https://huggingface.co/datasets/ylecun/mnist
	const std::vector<std::string> image_paths{
		std::string(mnist_dir_path + "/0.jpg"),
		std::string(mnist_dir_path + "/1.jpg"),
		std::string(mnist_dir_path + "/2.jpg"),
		std::string(mnist_dir_path + "/4.jpg"),
		std::string(mnist_dir_path + "/7.jpg"),
	};

	//-----
	MyLogger logger;

	// Create a builder
	std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
	if (!builder)
	{
		std::cerr << "Failed to create TensorRT builder." << std::endl;
		return;
	}
	// Create a network
#if 0
	// Set network flags for explicit batch size
	const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
#else
	std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
#endif
	if (!network)
	{
		std::cerr << "Failed to create TensorRT network." << std::endl;
		return;
	}
	// Create a parser
	std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
	if (!parser)
	{
		std::cerr << "Failed to create TensorRT ONNX parser." << std::endl;
		return;
	}

	// Create an engine
	std::unique_ptr<nvinfer1::ICudaEngine> engine;
	{
		if (plan_or_onnx_path.ends_with(".onnx"))  // Too slow
		{
			std::cout << "Parsing an ONNX file, " << plan_or_onnx_path << "..." << std::endl;
			const auto start_time(std::chrono::steady_clock::now());
#if 0
			std::ifstream stream(plan_or_onnx_path, std::ios::binary | std::ios::ate);
			if (!stream)
			{
				std::cerr << "Failed to open an ONNX file, " << plan_or_onnx_path << std::endl;
				return;
			}
			const std::streamsize sz = stream.tellg();
			stream.seekg(0, std::ios::beg);
			std::vector<char> buf(sz, 0);
			if (!stream.read(buf.data(), buf.size()))
			{
				std::cerr << "Failed to read ONNX file." << std::endl;
				return nullptr;
			}

			if (parser->parse(buf.data(), buf.size()))
#else
			if (parser->parseFromFile(plan_or_onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
#endif
			{
				std::cout << "An ONNX file parsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;
			}
			else
			{
				std::cerr << "Failed to parse an ONNX file" << std::endl;
				return;
			}

			// Configure a builder config
			std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
			if (!config)
			{
				std::cerr << "Failed to create TensorRT builder config." << std::endl;
				return;
			}
			config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // Set maximum workspace size to 1 GiB

#if 0
			{
				nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

				const nvinfer1::Dims data_dims_min{ 3, { 256, 256, 3 } };
				const nvinfer1::Dims data_dims_opt{ 3, { 512, 512, 3 } };
				const nvinfer1::Dims data_dims_max{ 3, { 1024, 1024, 3 } };
				profile->setDimensions("data_name", nvinfer1::OptProfileSelector::kMIN, data_dims_min);
				profile->setDimensions("data_name", nvinfer1::OptProfileSelector::kOPT, data_dims_opt);
				profile->setDimensions("data_name", nvinfer1::OptProfileSelector::kMAX, data_dims_max);

#if 1
				const std::vector<int32_t> data_shape_min{ 256, 256 };
				const std::vector<int32_t> data_shape_opt{ 512, 512 };
				const std::vector<int32_t> data_shape_max{ 1024, 1024 };
				profile->setShapeValues("data_shape", nvinfer1::OptProfileSelector::kMIN, data_shape_min.data(), (int32_t)data_shape_min.size());
				profile->setShapeValues("data_shape", nvinfer1::OptProfileSelector::kOPT, data_shape_opt.data(), (int32_t)data_shape_opt.size());
				profile->setShapeValues("data_shape", nvinfer1::OptProfileSelector::kMAX, data_shape_max.data(), (int32_t)data_shape_max.size());
#else
				const std::vector<int64_t> data_shape_min{ 256, 256 };
				const std::vector<int64_t> data_shape_opt{ 512, 512 };
				const std::vector<int64_t> data_shape_max{ 1024, 1024 };
				profile->setShapeValuesV2("data_shape", nvinfer1::OptProfileSelector::kMIN, data_shape_min.data(), (int32_t)data_shape_min.size());
				profile->setShapeValuesV2("data_shape", nvinfer1::OptProfileSelector::kOPT, data_shape_opt.data(), (int32_t)data_shape_opt.size());
				profile->setShapeValuesV2("data_shape", nvinfer1::OptProfileSelector::kMAX, data_shape_max.data(), (int32_t)data_shape_max.size());
#endif

				config->addOptimizationProfile(profile);
			}
#endif

			// Build an engine
			engine.reset(builder->buildEngineWithConfig(*network, *config));
			if (!engine)
			{
				std::cerr << "Failed to build TensorRT engine." << std::endl;
				return;
			}
		}
		else  // .plan or .engine
		{
			std::cout << "Parsing a TensorRT plan file, " << plan_or_onnx_path << "..." << std::endl;
			const auto start_time(std::chrono::steady_clock::now());
			std::ifstream stream(plan_or_onnx_path, std::ios::binary | std::ios::ate);
			if (!stream)
			{
				std::cerr << "Failed to open TensorRT plan file, " << plan_or_onnx_path << std::endl;
				return;
			}
			const std::streamsize sz(stream.tellg());
			stream.seekg(0, std::ios::beg);
			std::vector<char> buf(sz, 0);
			if (!stream.read(buf.data(), buf.size()))
			{
				std::cerr << "Failed to read TensorRT plan file." << std::endl;
				return;
			}
			std::cout << "A TensorRT plan file parsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

			// Create a runtime
			std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
			if (!runtime)
			{
				std::cerr << "Failed to create TensorRT runtime." << std::endl;
				return;
			}

			// Deserialize an engine
			engine.reset(runtime->deserializeCudaEngine(buf.data(), buf.size()));
			if (!engine)
			{
				std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
				return;
			}
		}
	}

	// Show engine info
	{
		const auto num_tensors(engine->getNbIOTensors());
		std::cout << "Engine info:" << std::endl;
		for (int32_t idx = 0; idx < num_tensors; ++idx)
		{
			const char* tensor_name = engine->getIOTensorName(idx);
			const nvinfer1::Dims& tensor_dims = engine->getTensorShape(tensor_name);

			std::cout << "\tTensor #" << idx << ':' << std::endl;
			std::cout << "\t\tName: " << tensor_name << std::endl;
			std::cout << "\t\tShape (dimension = " << tensor_dims.nbDims << "): ";
			std::copy(tensor_dims.d, tensor_dims.d + tensor_dims.nbDims, std::ostream_iterator<int64_t>(std::cout, ", "));
			std::cout << std::endl;
		}
	}

	// Create execution context
	std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
	if (!context)
	{
		std::cerr << "Failed to create TensorRT execution context." << std::endl;
		return;
	}

	//const nvinfer1::ICudaEngine& engine = context->getEngine();

	//-----
	// Prepare input/output buffers
	//	"Input3": float32[1,1,28,28], "Plus214_Output_0": float32[1,10]
	//	Can check the names of inputs & outputs in https://netron.app/
#if 0
	const int input_id(engine->getBindingIndex("Input3"));  // Deprecated
	const int output_id(engine->getBindingIndex("Plus214_Output_0"));  // Deprecated
#else
	const int input_id(0);
	const int output_id(1);
#endif

	// Get input and output tensor names from the engine
	const char* input_name = engine->getIOTensorName(0);
	const char* output_name = engine->getIOTensorName(1);

	// Get input and output dimensions
	const nvinfer1::Dims& input_dims = engine->getTensorShape(input_name);
	const nvinfer1::Dims& output_dims = engine->getTensorShape(output_name);

	// Allocate host memory for input and output
	const int64_t input_size(std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>()));
	const int64_t output_size(std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int64_t>()));
	assert(num_classes == output_size);

	// Pointers to the input and output buffers on the GPU
	void* buffers[2] = { nullptr, };
	//std::vector<void*> buffers(3, nullptr);
	CUDA_CHECK(cudaMalloc(&buffers[input_id], input_size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&buffers[output_id], output_size * sizeof(float)));

	auto prepare_input_image = [](const std::string& image_path, std::vector<float>& input_data) -> void {
		const cv::Mat& gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (gray.empty())
		{
			std::cerr << "Failed to load image, " << image_path << std::endl;
			return;
		}
		//cv::resize(gray, gray, cv::Size(image_width, image_height));

		float* ptr = input_data.data();
		for (unsigned y = 0; y < image_height; ++y)
			for (unsigned x = 0; x < image_width; ++x, ++ptr)
				*ptr += gray.at<unsigned char>(y, x) == 0 ? 0.0f : 1.0f;
	};

	std::vector<float> input_data(1 * image_channel * image_height * image_width, 0.0f);  // [0, 1]
	assert(input_data.size() == input_size);
	std::vector<float> output_data(output_size, 0.0f);
	//assert(output_data.size() == output_size);
	for (const auto& image_path : image_paths)
	{
		// Prepare input image (1, 1, 28, 28)
		prepare_input_image(image_path, input_data);

		// Copy input data from host to device
		CUDA_CHECK(cudaMemcpy(buffers[input_id], input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

		// Run inference
		std::cout << "Inferring..." << std::endl;
		const auto start_time(std::chrono::steady_clock::now());
		if (context->executeV2(buffers))
		{
			std::cout << "Inferred: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

			// Copy output data from device to host
			CUDA_CHECK(cudaMemcpy(output_data.data(), buffers[output_id], output_size * sizeof(float), cudaMemcpyDeviceToHost));

			// Show results
			//softmax(output_data);
			const int64_t predicted(std::distance(output_data.begin(), std::max_element(output_data.begin(), output_data.end())));
			std::cout << "Predicted = " << predicted << std::endl;
		}
		else
		{
			std::cerr << "Inference failed." << std::endl;
		}
	}

	// Cleanup
	CUDA_CHECK(cudaFree(buffers[input_id]));
	CUDA_CHECK(cudaFree(buffers[output_id]));
	buffers[input_id] = nullptr;
	buffers[output_id] = nullptr;
}

void mnist_onnx_tensorrt_stream_test()
{
	const size_t image_width(28);
	const size_t image_height(28);
	const size_t image_channel(1);
	const size_t num_classes(10);

	//const std::string mnist_dir_path("path/to/mnist");
	const std::string mnist_dir_path("./mnist");

	// REF [site] >> https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/MNIST
	//const std::string plan_or_onnx_path(mnist_dir_path + "/mnist.onnx");
	//const std::string plan_or_onnx_path(mnist_dir_path + "/mnist.plan");
	const std::string plan_or_onnx_path(mnist_dir_path + "/mnist_fp16.plan");

	// REF [site] >> https://huggingface.co/datasets/ylecun/mnist
	const std::vector<std::string> image_paths{
		std::string(mnist_dir_path + "/0.jpg"),
		std::string(mnist_dir_path + "/1.jpg"),
		std::string(mnist_dir_path + "/2.jpg"),
		std::string(mnist_dir_path + "/4.jpg"),
		std::string(mnist_dir_path + "/7.jpg"),
	};

	//-----
	MyLogger logger;

	// Create a builder
	std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
	if (!builder)
	{
		std::cerr << "Failed to create TensorRT builder." << std::endl;
		return;
	}
	// Create a network
#if 0
	// Set network flags for explicit batch size
	const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
#else
	std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
#endif
	if (!network)
	{
		std::cerr << "Failed to create TensorRT network." << std::endl;
		return;
	}
	// Create a parser
	std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
	if (!parser)
	{
		std::cerr << "Failed to create TensorRT ONNX parser." << std::endl;
		return;
	}

	// Create an engine
	std::unique_ptr<nvinfer1::ICudaEngine> engine;
	{
		if (plan_or_onnx_path.ends_with(".onnx"))  // Too slow
		{
			std::cout << "Parsing an ONNX file, " << plan_or_onnx_path << "..." << std::endl;
			const auto start_time(std::chrono::steady_clock::now());
#if 0
			std::ifstream stream(plan_or_onnx_path, std::ios::binary | std::ios::ate);
			if (!stream)
			{
				std::cerr << "Failed to open an ONNX file, " << plan_or_onnx_path << std::endl;
				return;
			}
			const std::streamsize sz = stream.tellg();
			stream.seekg(0, std::ios::beg);
			std::vector<char> buf(sz, 0);
			if (!stream.read(buf.data(), buf.size()))
			{
				std::cerr << "Failed to read ONNX file." << std::endl;
				return nullptr;
			}

			if (parser->parse(buf.data(), buf.size()))
#else
			if (parser->parseFromFile(plan_or_onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
#endif
			{
				std::cout << "An ONNX file parsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;
			}
			else
			{
				std::cerr << "Failed to parse an ONNX file" << std::endl;
				return;
			}

			// Configure a builder config
			std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
			if (!config)
			{
				std::cerr << "Failed to create TensorRT builder config." << std::endl;
				return;
			}
			config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // Set maximum workspace size to 1 GiB

#if 0
			{
				nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

				const nvinfer1::Dims data_dims_min{ 3, { 256, 256, 3 } };
				const nvinfer1::Dims data_dims_opt{ 3, { 512, 512, 3 } };
				const nvinfer1::Dims data_dims_max{ 3, { 1024, 1024, 3 } };
				profile->setDimensions("data_name", nvinfer1::OptProfileSelector::kMIN, data_dims_min);
				profile->setDimensions("data_name", nvinfer1::OptProfileSelector::kOPT, data_dims_opt);
				profile->setDimensions("data_name", nvinfer1::OptProfileSelector::kMAX, data_dims_max);

#if 0
				const std::vector<int32_t> data_shape_min{ 256, 256 };
				const std::vector<int32_t> data_shape_opt{ 512, 512 };
				const std::vector<int32_t> data_shape_max{ 1024, 1024 };
				profile->setShapeValues("data_shape", nvinfer1::OptProfileSelector::kMIN, data_shape_min.data(), (int32_t)data_shape_min.size());
				profile->setShapeValues("data_shape", nvinfer1::OptProfileSelector::kOPT, data_shape_opt.data(), (int32_t)data_shape_opt.size());
				profile->setShapeValues("data_shape", nvinfer1::OptProfileSelector::kMAX, data_shape_max.data(), (int32_t)data_shape_max.size());
#else
				const std::vector<int64_t> data_shape_min{ 256, 256 };
				const std::vector<int64_t> data_shape_opt{ 512, 512 };
				const std::vector<int64_t> data_shape_max{ 1024, 1024 };
				profile->setShapeValuesV2("data_shape", nvinfer1::OptProfileSelector::kMIN, data_shape_min.data(), (int32_t)data_shape_min.size());
				profile->setShapeValuesV2("data_shape", nvinfer1::OptProfileSelector::kOPT, data_shape_opt.data(), (int32_t)data_shape_opt.size());
				profile->setShapeValuesV2("data_shape", nvinfer1::OptProfileSelector::kMAX, data_shape_max.data(), (int32_t)data_shape_max.size());
#endif

				config->addOptimizationProfile(profile);
			}
#endif

			// Build an engine
			engine.reset(builder->buildEngineWithConfig(*network, *config));
			if (!engine)
			{
				std::cerr << "Failed to build TensorRT engine." << std::endl;
				return;
			}
		}
		else  // .plan or .engine
		{
			std::cout << "Parsing a TensorRT plan file, " << plan_or_onnx_path << "..." << std::endl;
			const auto start_time(std::chrono::steady_clock::now());
			std::ifstream stream(plan_or_onnx_path, std::ios::binary | std::ios::ate);
			if (!stream)
			{
				std::cerr << "Failed to open TensorRT plan file, " << plan_or_onnx_path << std::endl;
				return;
			}
			const std::streamsize sz(stream.tellg());
			stream.seekg(0, std::ios::beg);
			std::vector<char> buf(sz, 0);
			if (!stream.read(buf.data(), buf.size()))
			{
				std::cerr << "Failed to read TensorRT plan file." << std::endl;
				return;
			}
			std::cout << "A TensorRT plan file parsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

			// Create a runtime
			std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
			if (!runtime)
			{
				std::cerr << "Failed to create TensorRT runtime." << std::endl;
				return;
			}

			// Deserialize an engine
			engine.reset(runtime->deserializeCudaEngine(buf.data(), buf.size()));
			if (!engine)
			{
				std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
				return;
			}
		}
	}

	// Show engine info
	{
		const auto num_tensors(engine->getNbIOTensors());
		std::cout << "Engine info:" << std::endl;
		for (int32_t idx = 0; idx < num_tensors; ++idx)
		{
			const char* tensor_name = engine->getIOTensorName(idx);
			const nvinfer1::Dims& tensor_dims = engine->getTensorShape(tensor_name);

			std::cout << "\tTensor #" << idx << ':' << std::endl;
			std::cout << "\t\tName: " << tensor_name << std::endl;
			std::cout << "\t\tShape (dimension = " << tensor_dims.nbDims << "): ";
			std::copy(tensor_dims.d, tensor_dims.d + tensor_dims.nbDims, std::ostream_iterator<int64_t>(std::cout, ", "));
			std::cout << std::endl;
		}
	}

	// Create execution context
	std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
	if (!context)
	{
		std::cerr << "Failed to create TensorRT execution context." << std::endl;
		return;
	}

	//const nvinfer1::ICudaEngine& engine = context->getEngine();

	//-----
	// Prepare input/output buffers
	//	"Input3": float32[1,1,28,28], "Plus214_Output_0": float32[1,10]
	//	Can check the names of inputs & outputs in https://netron.app/
#if 0
	const int input_id(engine->getBindingIndex("Input3"));  // Deprecated
	const int output_id(engine->getBindingIndex("Plus214_Output_0"));  // Deprecated
#else
	const int input_id(0);
	const int output_id(1);
#endif

	// Get input and output tensor names from the engine
	const char* input_name = engine->getIOTensorName(0);
	const char* output_name = engine->getIOTensorName(1);

	// Get input and output dimensions
	const nvinfer1::Dims& input_dims = engine->getTensorShape(input_name);
	const nvinfer1::Dims& output_dims = engine->getTensorShape(output_name);

	// Allocate host memory for input and output
	const int64_t input_size(std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>()));
	const int64_t output_size(std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int64_t>()));
	assert(num_classes == output_size);

	// Create CUDA stream for inference
	const size_t num_streams(image_paths.size());
	//std::vector<std::unique_ptr<cudaStream_t>> streams;
	//streams.reserve(num_streams);
	//for (size_t idx = 0; idx < num_streams; ++idx)
	//{
	//	auto stream = std::make_unique<cudaStream_t>();
	//	CUDA_CHECK(cudaStreamCreate(stream.get()));
	//	streams.push_back(std::move(stream));
	//}
	std::vector<cudaStream_t*> streams;
	streams.reserve(num_streams);
	{
		cudaStream_t* stream = new cudaStream_t;
		CUDA_CHECK(cudaStreamCreate(stream));
		streams.push_back(stream);
	}

	// Pointers to the input and output buffers on the GPU
	// TODO [check] >>
	//void* buffers[2] = { nullptr, };
	//std::vector<void*> buffers(2, nullptr);
	float* buffers[2] = { nullptr, };
	{
		//CUDA_CHECK(cudaMallocAsync(&buffers[input_id], input_size * sizeof(float), stream));
		//CUDA_CHECK(cudaMallocAsync(&buffers[output_id], output_size * sizeof(float), stream));
		// TODO [check] >>
		//CUDA_CHECK(cudaMalloc(&buffers[input_id], input_size * sizeof(float)* num_streams));
		//CUDA_CHECK(cudaMalloc(&buffers[output_id], output_size * sizeof(float)* num_streams));
		CUDA_CHECK(cudaMalloc((void**)&buffers[input_id], input_size * sizeof(float) * num_streams));
		CUDA_CHECK(cudaMalloc((void**)&buffers[output_id], output_size * sizeof(float) * num_streams));

		context->setTensorAddress(input_name, buffers[input_id]);
		context->setTensorAddress(output_name, buffers[output_id]);
	}

	auto prepare_input_image = [](const std::string& image_path, float* input_data) -> void {
		const cv::Mat& gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (gray.empty())
		{
			std::cerr << "Failed to load image, " << image_path << std::endl;
			return;
		}
		//cv::resize(gray, gray, cv::Size(image_width, image_height));

		float* ptr = input_data;
		for (unsigned y = 0; y < image_height; ++y)
			for (unsigned x = 0; x < image_width; ++x, ++ptr)
				*ptr += gray.at<unsigned char>(y, x) == 0 ? 0.0f : 1.0f;
	};

#if 1
	//std::vector<float> input_data(1 * image_channel * image_height * image_width, 0.0f);  // [0, 1]
	//assert(input_data.size() == input_size);
	std::vector<float> input_data(1 * image_channel * image_height * image_width * num_streams, 0.0f);  // [0, 1]
	assert(input_data.size() == input_size * num_streams);
	//std::vector<float> output_data(output_size, 0.0f);
	//assert(output_data.size() == output_size);
	std::vector<float> output_data(output_size * num_streams);
	//assert(output_data.size() == output_size * num_streams);
#else
	float* input_data = nullptr;
	float* output_data = nullptr;
	CUDA_CHECK(cudaMallocHost((void**)&input_data, input_size * sizeof(float) * num_streams));
	CUDA_CHECK(cudaMallocHost((void**)&output_data, output_size * sizeof(float) * num_streams));
#endif
	for (size_t idx = 0; idx < num_streams; ++idx)
	{
		// Prepare input image (1, 1, 28, 28)
		prepare_input_image(image_paths[idx], input_data.data() + input_size);

		// Copy input data from host to device
		CUDA_CHECK(cudaMemcpyAsync((void*)(buffers[input_id] + input_size * idx), (void*)(input_data.data() + input_size * idx), input_size * sizeof(float), cudaMemcpyHostToDevice, *(streams[idx])));
	}

	for (auto& stream : streams)
	{
		// Run inference
		std::cout << "Inferring..." << std::endl;
		const auto start_time(std::chrono::steady_clock::now());
		//if (context->executeAsyncV2(buffers, stream, nullptr))  // Compile-time error: 'executeAsyncV2': is not a member of 'nvinfer1::IExecutionContext'
		if (context->enqueueV3(*stream))
			std::cout << "Inferred: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;
		else
		{
			std::cerr << "Inference failed." << std::endl;
		}
	}

	for (size_t idx = 0; idx < num_streams; ++idx)
	{
		// Copy output data from device to host
		CUDA_CHECK(cudaMemcpyAsync((void*)(output_data.data() + output_size * idx), (void*)((float*)buffers[output_id] + output_size * idx), output_size * sizeof(float), cudaMemcpyDeviceToHost, *streams[idx]));

		// Show results
		//softmax(output_data);
		const int64_t predicted(std::distance(output_data.data() + output_size * idx, std::max_element(output_data.data() + output_size * idx, output_data.data() + output_size * (idx + 1))));
		std::cout << "Predicted = " << predicted << std::endl;
	}

	// Synchronize stream
	for (auto& stream : streams)
		CUDA_CHECK(cudaStreamSynchronize(*stream));

	// Cleanup
	//CUDA_CHECK(cudaFreeHost((void*)input_data));
	//CUDA_CHECK(cudaFreeHost((void*)output_data));

	//CUDA_CHECK(cudaFreeAsync(buffers[input_id], stream));
	//CUDA_CHECK(cudaFreeAsync(buffers[output_id], stream));
	CUDA_CHECK(cudaFree(buffers[input_id]));
	CUDA_CHECK(cudaFree(buffers[output_id]));
	buffers[input_id] = nullptr;
	buffers[output_id] = nullptr;

	for (auto& stream : streams)
	{
		CUDA_CHECK(cudaStreamDestroy(*stream));
		delete stream;
		stream = nullptr;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_tensorrt {

}  // namespace my_tensorrt

int tensorrt_main(int argc, char *argv[])
{
	// TensorRT
	//	https://github.com/NVIDIA/TensorRT/releases
	//	https://developer.nvidia.com/tensorrt

	// ONNX to TensorRT plan
	//	https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html
	//
	//	trtexec --onnx=path/to/mnist.onnx --saveEngine=path/to/mnist.plan
	//	trtexec --onnx=path/to/mnist.onnx --saveEngine=path/to/mnist.plan --workspace=4096 --shapes=input:1024x1024x3
	//	trtexec --onnx=path/to/mnist.onnx --saveEngine=path/to/mnist_fp16.plan --fp16
	//	trtexec --onnx=path/to/mnist.onnx --saveEngine=path/to/mnist_fp16.plan --fp16 --workspace=4096 --shapes=input:1024x1024x3

	local::mnist_onnx_tensorrt_test();
	//local::mnist_onnx_tensorrt_stream_test();  // Uses CUDA Stream. Runtime error

	// Segment Anything (SAM)
	//	Refer to sam_tensorrt_test.cpp

	return 0;
}
