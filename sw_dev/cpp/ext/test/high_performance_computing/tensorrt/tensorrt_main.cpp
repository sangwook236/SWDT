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

#define CUDA_CHECK(status) \
	do \
	{ \
		auto ret = (status); \
		if (ret != 0) \
		{ \
			std::cerr << "CUDA failure: " << ret << std::endl; \
			abort(); \
		} \
	} while (0)

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
	const size_t image_width = 28;
	const size_t image_height = 28;
	const size_t image_channel = 1;
	const size_t num_classes = 10;

	// REF [site] >> https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/MNIST
	const std::string onnx_path("./mnist/mnist.onnx");

	// Prepare input image (1, 1, 28, 28)
	std::vector<float> input_data(1 * image_channel * image_height * image_width, 0.0f);  // [0, 1]
	{
		// REF [site] >> https://huggingface.co/datasets/ylecun/mnist
		const std::string image_path("./mnist/0.jpg");
		//const std::string image_path("./mnist/1.jpg");
		//const std::string image_path("./mnist/2.jpg");
		//const std::string image_path("./mnist/4.jpg");
		//const std::string image_path("./mnist/7.jpg");
	
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
	}

	{
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

		// Parse an ONNX model
		{
			std::cout << "Parsing an ONNX file, " << onnx_path << "..." << std::endl;
			const auto start_time(std::chrono::steady_clock::now());
#if 0
			std::ifstream stream(onnx_path, std::ios::binary | std::ios::ate);
			if (!stream)
			{
				std::cerr << "Failed to open an ONNX file, " << onnx_path << std::endl;
				return;
			}
			const std::streamsize sz = stream.tellg();
			stream.seekg(0, std::ios::beg);
			std::vector<char> buf(sz);
			stream.read(buf.data(), sz);
			if (parser->parse(buf.data(), buf.size()))
#else
			if (parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
#endif
			{
				std::cout << "An ONNX file parsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;
			}
			else
			{
				std::cerr << "Failed to parse an ONNX file, " << onnx_path << std::endl;
				return;
			}
		}

		// Configure a builder config
		std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
		if (!config)
		{
			std::cerr << "Failed to create TensorRT builder config." << std::endl;
			return;
		}
		config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // Set maximum workspace size to 1 GiB

		// Build an engine
		std::unique_ptr<nvinfer1::ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config));
		if (!engine)
		{
			std::cerr << "Failed to build TensorRT engine." << std::endl;
			return;
		}

		//-----
		// Infer

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
		const int input_idx = engine->getBindingIndex("Input3");  // Deprecated
		const int output_idx = engine->getBindingIndex("Plus214_Output_0");  // Deprecated
#else
		const int input_idx = 0;
		const int output_idx = 1;
#endif

		// Get input and output tensor names from the engine
		const char* input_name = engine->getIOTensorName(0);
		const char* output_name = engine->getIOTensorName(1);

		// Get input and output dimensions
		const nvinfer1::Dims input_dims = engine->getTensorShape(input_name);
		const nvinfer1::Dims output_dims = engine->getTensorShape(output_name);

		{
			// Show input & output
			std::cout << "Input:" << std::endl;
			std::cout << "\tname = " << input_name << std::endl;
			std::cout << "\tShape (dimension = " << input_dims.nbDims << "): ";
			std::copy(input_dims.d, input_dims.d + input_dims.nbDims, std::ostream_iterator<int64_t>(std::cout, ", "));
			std::cout << std::endl;
			std::cout << "Output:" << std::endl;
			std::cout << "\tname = " << output_name << std::endl;
			std::cout << "\tShape (dimension = " << output_dims.nbDims << "): ";
			std::copy(output_dims.d, output_dims.d + output_dims.nbDims, std::ostream_iterator<int64_t>(std::cout, ", "));
			std::cout << std::endl;
		}

		// Allocate host memory for input and output
		const int64_t input_size = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>());
		const int64_t output_size = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int64_t>());
		assert(input_data.size() == input_size);
		assert(num_classes == output_size);

		// Pointers to the input and output buffers on the GPU
		void* buffers[2] = { nullptr, };
		CUDA_CHECK(cudaMalloc(&buffers[input_idx], input_size * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&buffers[output_idx], output_size * sizeof(float)));

		context->setTensorAddress(input_name, buffers[input_idx]);
		context->setTensorAddress(output_name, buffers[output_idx]);

#if 1
		CUDA_CHECK(cudaMemcpy(buffers[input_idx], input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

		// Run inference
		std::cout << "Inferring..." << std::endl;
		const auto start_time(std::chrono::steady_clock::now());
		if (context->executeV2(buffers))
		{
			std::cout << "Inferred: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

			// Copy output data from device to host
			std::vector<float> output_data(output_size);
			CUDA_CHECK(cudaMemcpy(output_data.data(), buffers[output_idx], output_size * sizeof(float), cudaMemcpyDeviceToHost));

			// Print output
			//softmax(output_data);
			const int64_t predicted = std::distance(output_data.begin(), std::max_element(output_data.begin(), output_data.end()));
			std::cout << "Predicted = " << predicted << std::endl;
		}
		else
		{
			std::cerr << "Inference failed." << std::endl;
		}

		// Cleanup
		CUDA_CHECK(cudaFree(buffers[input_idx]));
		CUDA_CHECK(cudaFree(buffers[output_idx]));
#else
		// Infer async

		// Create CUDA stream for inference
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		// Copy input data from host to device
		CUDA_CHECK(cudaMemcpyAsync(buffers[input_idx], input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

		// Execute the inference
		std::cout << "Inferring..." << std::endl;
		const auto start_time(std::chrono::steady_clock::now());
		//if (context->executeAsyncV2(buffers, stream, nullptr))  // Compile-time error: 'executeAsyncV2': is not a member of 'nvinfer1::IExecutionContext'
		if (context->enqueueV3(stream))
		{
			std::cout << "Inferred: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << " msec." << std::endl;

			// Copy output data from device to host
			std::vector<float> output_data(output_size);
			CUDA_CHECK(cudaMemcpyAsync(output_data.data(), buffers[output_idx], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

			// Print output
			//softmax(output_data);
			const int64_t predicted = std::distance(output_data.begin(), std::max_element(output_data.begin(), output_data.end()));
			std::cout << "Predicted = " << predicted << std::endl;
		}
		else
		{
			std::cerr << "Inference failed." << std::endl;
		}

		// Synchronize stream
		CUDA_CHECK(cudaStreamSynchronize(stream));

		// Free GPU memory
		CUDA_CHECK(cudaFree(buffers[input_idx]));
		CUDA_CHECK(cudaFree(buffers[output_idx]));
		CUDA_CHECK(cudaStreamDestroy(stream));
#endif
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_tensorrt {

}  // namespace my_tensorrt

int tensorrt_main(int argc, char *argv[])
{
	local::mnist_onnx_tensorrt_test();

	return 0;
}
