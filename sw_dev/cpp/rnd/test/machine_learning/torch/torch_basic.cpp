//#include "stdafx.h"
#include <cassert>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <torch/torch.h>


namespace torch {

Tensor addmm2(
	Scalar beta, const Tensor &self,
	Scalar alpha, const Tensor &mat1,
	const Tensor &mat2
)
{
	return addmm(self, mat1, mat2, beta, alpha);
}

Scalar sum2(const Tensor &self)
{
	return sum(self).item();
}

}  // namespace torch

namespace {
namespace local {

void basic_operation()
{
	// PyTorch version.
	std::cout << "PyTorch version: " <<
		TORCH_VERSION_MAJOR << '.' <<
		TORCH_VERSION_MINOR << '.' <<
		TORCH_VERSION_PATCH << std::endl;

	// Device.
	const torch::DeviceIndex gpu = -1;
	const auto is_cuda_available = torch::cuda::is_available();
	//const auto is_cuda_available = torch::hasCUDA();

	//const torch::Device device(is_cuda_available ? torch::kCUDA : torch::kCPU);
	const auto device(is_cuda_available ? torch::Device(torch::kCUDA, gpu) : torch::Device(torch::kCPU));
	std::cout << (std::string("Device: ") + device.str()) << std::endl;

	//------------------------------------------------------------
	// Tensor basics.
	//	https://pytorch.org/cppdocs/notes/tensor_basics.html

	{
		const torch::Tensor tensor = torch::eye(3);
		std::cout << tensor << std::endl;

		std::cout << "Tensor: dim = " << tensor.dim() << ", sizes = " << tensor.sizes() << std::endl;
		std::cout << "\tsize0 = " << tensor.size(0) << ", size1 = " << tensor.size(1) << std::endl;
		std::cout << "Tensor: min = " << torch::min(tensor).item<float>() << ", max = " << torch::max(tensor).item<float>() << std::endl;
		std::cout << "Tensor: vector size = " << std::vector(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel()).size() << std::endl;
	}

	// Efficient access to tensor elements.
	{
		// CPU accessors.
		torch::Tensor foo = torch::rand({12, 12});

		// Assert foo is 2-dimensional and holds floats.
		auto foo_a = foo.accessor<float, 2>();
		float trace = 0;

		for (int i = 0; i < foo_a.size(0); ++i)
		{
			// Use the accessor foo_a to get tensor data.
			trace += foo_a[i][i];
		}
	}
#if 0
	{
		// CUDA accessors.
		//	In a CUDA source file.

		__global__ void packed_accessor_kernel(torch::PackedTensorAccessor64<float, 2> foo, float *trace)
		{
			int i = threadIdx.x;
			gpuAtomicAdd(trace, foo[i][i]);
		}

		torch::Tensor foo = torch::rand({12, 12});

		// Assert foo is 2-dimensional and holds floats.
		auto foo_a = foo.packed_accessor64<float, 2>();
		float trace = 0;

		packed_accessor_kernel<<<1, 12>>>(foo_a, &trace);
	}
#endif

	// Using externally created data.
	{
		float data[] = {
			1, 2, 3,
			4, 5, 6
		};
		torch::Tensor f = torch::from_blob(data, {2, 3});
		//torch::Tensor f = torch::from_blob(data, {2, 3}, torch::kByte);

		std::cout << "f = " << f << std::endl;
	}

	// Scalars and zero-dimensional tensors.
	{
		torch::Tensor a = torch::rand({2, 4});
		torch::Tensor b = torch::rand({2, 3});
		torch::Tensor c = torch::rand({3, 4});
		torch::Tensor r = torch::addmm2(1.0, a, 0.5, b, c);
		torch::Scalar s = torch::sum2(a);

		std::cout << "r = " << r << std::endl;
		std::cout << "s.type() = " << s.type() << ", s = " << s.toDouble() << std::endl;

		torch::Tensor two = torch::rand({10, 20});
		two[1][2] = 4;  // two[1][2]: a zero-dimensional tensor.
	}

	//------------------------------------------------------------
	// Tensor creation.
	//	https://pytorch.org/cppdocs/notes/tensor_creation.html

	// Specifying a size.
	{
		torch::Tensor tensor1 = torch::ones(5);

		torch::Tensor tensor2 = torch::randn({3, 4, 5});
		assert(tensor2.sizes() == std::vector<int64_t>({3, 4, 5}));

		// Passing function-specific parameters.
		torch::Tensor tensor3 = torch::randint(/*high=*/10, {5, 5});
		torch::Tensor tensor4 = torch::randint(/*low=*/3, /*high=*/10, {5, 5});
	}

	// Configuring properties of the tensor.
	//	For dtype: {kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat32, kFloat64}.
	//	For layout: {kStrided, kSparse}.
	//	For device: {kCPU, or kCUDA} (which accepts an optional device index).
	//	For requires_grad: {true, false}.
	{
		auto options = torch::TensorOptions()
			.dtype(torch::kFloat32)
			.layout(torch::kStrided)
			.device(torch::kCUDA, 1)
			.requires_grad(true);
		//auto options = torch::TensorOptions().device(torch::kCUDA, 1).requires_grad(true);
		//torch::TensorOptions options;

		torch::Tensor tensor1 = torch::full({3, 4}, /*value=*/123, options);

		assert(tensor1.dtype() == torch::kFloat32);
		assert(tensor1.layout() == torch::kStrided);
		assert(tensor1.device().type() == torch::kCUDA);
		assert(tensor1.device().is_cuda());
		assert(tensor1.device().index() == 1);
		assert(tensor1.requires_grad());

		// A 32-bit float, strided, CPU tensor that does not require a gradient.
		torch::Tensor tensor2 = torch::randn({3, 4});
		torch::Tensor range = torch::arange(5, 10);

		torch::Tensor tensor3 = torch::ones(10, torch::TensorOptions().dtype(torch::kFloat32));
		torch::Tensor tensor4 = torch::ones(10, torch::dtype(torch::kFloat32));
		torch::Tensor tensor5 = torch::ones(10, torch::kFloat32);
		torch::Tensor tensor6 = torch::ones(10, torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided));
		torch::Tensor tensor7 = torch::ones(10, torch::dtype(torch::kFloat32).layout(torch::kStrided));
		// tensor = torch.randn(3, 4, dtype=torch.float32, device=torch.device('cuda', 1), requires_grad=True);
		torch::Tensor tensor8 = torch::randn({3, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 1).requires_grad(true));
	}

	// Conversion.
	{
		torch::Tensor float_tensor = torch::randn({2, 3}, torch::kFloat32);
		torch::Tensor int_tensor = float_tensor.to(torch::kInt64);

		torch::Tensor gpu_tensor = float_tensor.to(torch::kCUDA);
		torch::Tensor gpu_two_tensor = float_tensor.to(torch::Device(torch::kCUDA, 1));

		torch::Tensor async_cpu_tensor = gpu_tensor.to(torch::kCPU, /*non_blocking=*/true);
	}

	//------------------------------------------------------------
	// Tensor CUDA Stream.
	//	A CUDA Stream is a linear sequence of execution that belongs to a specific CUDA device.
	//	https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html

	//------------------------------------------------------------
	// Tensor indexing.
	//	https://pytorch.org/cppdocs/notes/tensor_indexing.html

	/*
	Python					C++
	-------------------------------------------------------------------------------
	None					None
	Ellipsis				Ellipsis
	...						"..."
	123						123
	True					true
	False					false
	: or ::					Slice() or Slice(None, None) or Slice(None, None, None)
	1: or 1::				Slice(1, None) or Slice(1, None, None)
	:3 or :3:				Slice(None, 3) or Slice(None, 3, None)
	::2						Slice(None, None, 2)
	1:3						Slice(1, 3)
	1::2					Slice(1, None, 2)
	:3:2					Slice(None, 3, 2)
	1:3:2					Slice(1, 3, 2)
	torch.tensor([1, 2])	torch::tensor({1, 2})
	*/

	// Getter.
	{
		torch::Tensor tensor = torch::randn({3, 4, 5});

		tensor.index({torch::indexing::None});  // tensor[None].
		tensor.index({torch::indexing::Ellipsis, "..."});  // tensor[Ellipsis, ...].
		tensor.index({1, 2});  // tensor[1, 2].
		tensor.index({true, false});  // tensor[True, False].
		tensor.index({torch::indexing::Slice(1, torch::indexing::None, 2)});  // tensor[1::2].
		tensor.index({torch::tensor({1, 2})});  // tensor[torch.tensor([1, 2])].
		tensor.index({"...", 0, true, torch::indexing::Slice(1, torch::indexing::None, 2), torch::tensor({1, 2})});  // tensor[..., 0, True, 1::2, torch.tensor([1, 2])].
	}

	// Setter.
	{
		torch::Tensor tensor = torch::randn({3, 4, 5});

		tensor.index_put_({torch::indexing::None}, 1);  // tensor[None] = 1.
		tensor.index_put_({torch::indexing::Ellipsis, "..."}, 1);  // tensor[Ellipsis, ...] = 1.
		tensor.index_put_({1, 2}, 1);  // tensor[1, 2] = 1.
		tensor.index_put_({true, false}, 1);  // tensor[True, False] = 1.
		tensor.index_put_({torch::indexing::Slice(1, torch::indexing::None, 2)}, 1);  // tensor[1::2] = 1.
		tensor.index_put_({torch::tensor({1, 2})}, 1);  // tensor[torch.tensor([1, 2])] = 1.
		tensor.index_put_({"...", 0, true, torch::indexing::Slice(1, torch::indexing::None, 2), torch::tensor({1, 2})}, 1);  // tensor[..., 0, True, 1::2, torch.tensor([1, 2])] = 1.
	}
}

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
		//return torch::addmm(bias, input, weight);
		return torch::randn({1, 2});
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

#if 0
#include <opencv2/opencv.hpp>

// REF [site] >> https://discuss.pytorch.org/t/how-to-convert-an-opencv-image-into-libtorch-tensor/90818/2
void opencv_example()
{
	const std::string image_filepath("/path/to/image.png");
	const cv::Mat img(cv::imread(image_filepath));
	if (img.empty())
	{
		std::cerr << "Image file not found, " << image_filepath << std::endl;
		return;
	}

	at::Tensor tensor_uint8 = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, at::kByte);
	std::cout << "Sizes = " << tensor_uint8.sizes() << std::endl;

	at::Tensor tensor_float = tensor_uint8.toType(c10::kFloat).div(255);

#if 0
	cv::Mat img_uint8(cv::Size{img.cols, img.rows}, img.type(), tensor_uint8.data_ptr<unsigned char>());
	cv::Mat img_float(cv::Size{img.cols, img.rows}, (img.channels() == 3 ? CV_32FC3 : CV_32FC1), tensor_float.data_ptr<float>());
	cv::imshow("Image (original)", img);
	cv::imshow("Image (uint8)", img_uint8);
	cv::imshow("Image (float)", img_float);
	cv::waitKey(0);
	cv::destroyAllWindows();
#endif

	//-----
	tensor_uint8 = tensor_uint8.permute({2, 0, 1});  // H x W x C -> C x H x W.
	std::cout << "Sizes = " << tensor_uint8.sizes() << std::endl;
	tensor_uint8.unsqueeze_(/*dim=*/0);  // 1 x C x H x W.
	std::cout << "Sizes (unsqueezed) = " << tensor_uint8.sizes() << std::endl;

	//tensor_float = tensor_float.clamp_max(c10::Scalar(0.1236));
	tensor_float.clamp_min_(c10::Scalar(0.1236));
	tensor_float.clamp_max_(c10::Scalar(0.6321));

	std::cout << "Min = " << torch::min(tensor_float).item<float>() << ", max = " << torch::max(tensor_float).item<float>() << std::endl;
	std::cout << "Max (member function) = " << tensor_float.max().item<float>() << std::endl;
	const auto &results = tensor_float.max(/*dim=*/0);
	std::cout << "Max value sizes = " << std::get<0>(results).sizes() << ", max index sizes = " << std::get<1>(results).sizes() << std::endl;
	std::cout << "Max value at (0, 0) = " << std::get<0>(results)[0][0].item().toFloat() << ", max index at (0, 0) = " << std::get<1>(results)[0][0].item().toLong() << std::endl;
	std::cout << "Argmax = " << tensor_float.argmax().item().toInt() << std::endl;
	std::cout << "Argmax(dim=0) sizes = " << tensor_float.argmax(/*dim=*/0).sizes() << std::endl;

	const auto tensor_sliced = tensor_float.slice(/*dim=*/2, /*start=*/0, /*end=*/1);  // H x W x 1.
	std::cout << "Sliced sizes = " << tensor_sliced.sizes() << std::endl;
	tensor_sliced.squeeze_(/*dim=*/2);  // H x W.
	std::cout << "Sliced sizes (squeezed) = " << tensor_sliced.sizes() << std::endl;
}
#endif

}  // namespace local
}  // unnamed namespace

namespace my_torch {

void training_example()
{
	local::basic_operation();
	//local::simple_frontend_tutorial();

	//local::opencv_example();
}

}  // namespace my_torch

int main()
{
	my_torch::training_example();
	return 0;
}
